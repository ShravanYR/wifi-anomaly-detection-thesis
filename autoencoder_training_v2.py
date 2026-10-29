#!/usr/bin/env python3
"""
autoencoder_training.py
=======================
Trains two Autoencoder models on normal WiFi telemetry data.
Mirrors the exact same two-model architecture as Isolation Forest:

    Idle AE   — trained on idle rows (is_active=0)
                features: rssi, tx_throughput_mbps, rx_throughput_mbps
                architecture: 3 → 8 → 4 → 8 → 3

    Active AE — trained on active rows (is_active=1)
                features: all 15 traffic + MCS features
                architecture: 15 → 32 → 16 → 8 → 16 → 32 → 15

HOW ANOMALY DETECTION WORKS:
    AE is trained to RECONSTRUCT normal input.
    Anomaly score = MSE between input X and reconstruction X_hat.
    Normal data: low error. Anomaly data: HIGH error.
    Threshold = 99th percentile of training reconstruction errors.

WHY THIS IMPROVES ON ISOLATION FOREST:
    IF scores each feature against its marginal distribution independently.
    AE learns the JOINT distribution of ALL features together.
    retry_rate=0.45 + tx_success=0.72 + mcs_entropy=1.8 together
    is a joint pattern the AE has never seen — high reconstruction error.

USAGE:
    pip install torch
    python autoencoder_training.py

INPUT:  ml_ready_data.csv
OUTPUT: models/ae_*.pt, models/ae_*.npy, models/ae_*.pkl, models/ae_metadata.json
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("[ERROR] PyTorch not installed. Run: pip install torch")
    exit(1)

from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE      = "ml_ready_data_raw_active.csv"   # new active dataset
MODELS_DIR      = "models_active"                  # separate folder — old models untouched
RESULTS_DIR     = "results_active"                 # separate results folder
EXCLUDE_MACS    = ['a8:74:1d:be:20:48', 'a8:74:1d:47:e6:4e']
VALIDATION_MAC  = 'a8:74:1d:47:e6:46'
LEARNING_RATE   = 0.001
BATCH_SIZE      = 64
MAX_EPOCHS      = 200
PATIENCE        = 20
VAL_SPLIT       = 0.15
RANDOM_SEED     = 42

# Minimum idle rows required to train a meaningful idle model.
# If dataset has fewer idle rows than this, idle training is skipped
# and the previously trained idle model should be retained.
# 8 idle rows (one per client) is far too few for BatchNorm and a
# meaningful validation split — we need at least ~100.
MIN_IDLE_ROWS   = 100
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Suggestion 2 — threshold sweep
# All three percentiles computed and saved simultaneously.
# No retraining needed to switch — change which threshold the
# evaluator loads from ae_metadata.json.
#   95 = more sensitive (catches more, higher false alarm rate)
#   97 = middle ground  (recommended after outlier removal)
#   99 = conservative   (fewer false alarms, may miss subtle anomalies)
THRESHOLD_PERCENTILES = [95, 97, 99]
THRESHOLD_PCT         = 99   # primary threshold saved to .npy files

# Suggestions 3 & 4 — outlier removal before training
# Rows where any feature exceeds OUTLIER_IQR_FACTOR * IQR above/below
# the quartiles are treated as corrupted and removed before training.
# An error of 4962 in normal data means a feature was ~70 std devs off —
# caused by driver garbage, mid-interval reconnection counter reset, or
# timestamp ordering errors. Remove before the model learns from them.
# Factor 10 is conservative — only catches genuinely extreme values.
OUTLIER_IQR_FACTOR = 10.0
REMOVE_OUTLIERS    = True   # set False to compare with/without removal

IDLE_FEATURES = ['rssi','tx_success_rate',
    'retry_rate']

ACTIVE_FEATURES = [
    'tx_packet_rate', 'rx_packet_rate', 'tx_byte_rate', 'rx_byte_rate',
    'tx_success_rate', 'retry_rate', 'tx_failed_rate',
    'rssi', 'tx_throughput_mbps', 'rx_throughput_mbps',
    'low_mcs_ratio', 'mid_mcs_ratio', 'high_mcs_ratio',
    'mcs_weighted_score', 'mcs_entropy',
]

# ── Model ─────────────────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        # Encoder
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        enc += [nn.Linear(prev, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        # Decoder (mirror)
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        with torch.no_grad():
            return torch.mean((x - self.forward(x)) ** 2, dim=1).cpu().numpy()

# ── Train ─────────────────────────────────────────────────────────────────────
def train_ae(X_tr, X_val, input_dim, hidden_dims, latent_dim, name):
    model = Autoencoder(input_dim, hidden_dims, latent_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    crit  = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(X_tr)),
                        batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.FloatTensor(X_val)

    best_loss, best_state, patience_ctr = float('inf'), None, 0
    arch = f"{input_dim}→{hidden_dims}→{latent_dim}→{list(reversed(hidden_dims))}→{input_dim}"
    print(f"\n  [{name}] Architecture: {arch}")
    print(f"  [{name}] Train rows: {len(X_tr)}  Val rows: {len(X_val)}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_losses = []
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = np.mean(tr_losses)

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_val_t), X_val_t).item()

        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] Epoch {epoch+1:3d}  train={tr_loss:.6f}  "
                  f"val={val_loss:.6f}  patience={patience_ctr}/{PATIENCE}")
        if patience_ctr >= PATIENCE:
            print(f"  [{name}] Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  [{name}] Done. Best val_loss: {best_loss:.6f}")
    return model, best_loss

def compute_threshold(model, X_scaled):
    """
    Compute reconstruction errors and thresholds at all configured percentiles.
    Returns (primary_threshold, errors, threshold_dict).
    threshold_dict contains all percentile thresholds for the metadata.
    """
    model.eval()
    errors = model.reconstruction_error(torch.FloatTensor(X_scaled))
    thr_dict = {f"threshold_pct{p}": float(np.percentile(errors, p))
                for p in THRESHOLD_PERCENTILES}
    primary_threshold = float(np.percentile(errors, THRESHOLD_PCT))
    return primary_threshold, errors, thr_dict

def remove_outliers(df, features, name, log_fn):
    """
    Suggestions 3 & 4 — Remove rows with extreme feature values before training.

    Method: IQR-based detection per feature.
    A row is removed if ANY feature exceeds:
        Q3 + OUTLIER_IQR_FACTOR * IQR  (upper bound)
    or falls below:
        Q1 - OUTLIER_IQR_FACTOR * IQR  (lower bound)

    Why IQR and not std-dev: IQR is itself robust to outliers.
    If std-dev is used, one extreme value inflates std-dev and
    the threshold becomes useless (the outlier hides itself).

    Why this matters: a row with MSE=4962 during training pushes
    the 99th percentile threshold UP, making it harder to detect
    real anomalies. Removing it first gives a cleaner threshold.
    """
    if not REMOVE_OUTLIERS:
        log_fn(f"  [{name}] Outlier removal SKIPPED (REMOVE_OUTLIERS=False)")
        return df, 0, []

    avail = [f for f in features if f in df.columns]
    original_len = len(df)
    outlier_mask = pd.Series(False, index=df.index)
    flagged_features = []

    for feat in avail:
        col = df[feat]
        Q1  = col.quantile(0.25)
        Q3  = col.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue   # constant feature — no outliers possible
        lower = Q1 - OUTLIER_IQR_FACTOR * IQR
        upper = Q3 + OUTLIER_IQR_FACTOR * IQR
        feat_outliers = (col < lower) | (col > upper)
        n_feat = feat_outliers.sum()
        if n_feat > 0:
            outlier_mask |= feat_outliers
            flagged_features.append((feat, int(n_feat), float(lower), float(upper),
                                     float(col[feat_outliers].abs().max())))

    n_removed = outlier_mask.sum()
    df_clean  = df[~outlier_mask].copy()

    log_fn(f"  [{name}] Outlier removal (IQR factor = {OUTLIER_IQR_FACTOR}):")
    log_fn(f"    Rows before : {original_len:,}")
    log_fn(f"    Rows removed: {n_removed:,}  ({100*n_removed/original_len:.2f}%)")
    log_fn(f"    Rows after  : {len(df_clean):,}")

    if flagged_features:
        log_fn(f"    Flagged features (with outlier counts):")
        for feat, n, lo, hi, max_val in sorted(flagged_features,
                                                key=lambda x: -x[1]):
            log_fn(f"      {feat:35s}: {n:4d} rows  "
                   f"bounds=[{lo:.4f}, {hi:.4f}]  "
                   f"max_val_seen={max_val:.4f}")

    return df_clean, n_removed, flagged_features


def section(t): print(f"\n{'='*55}\n  {t}\n{'='*55}")

def patch_scaler_for_constant_features(scaler, feature_names, log_fn):
    """
    Fix zero-variance features in a fitted StandardScaler.

    StandardScaler computes (x - mean) / std.  If std == 0 (constant
    feature) the result is NaN, which silently corrupts model training.

    This function replaces any zero scale_ value with 1e-6 so that
    a constant feature always normalises to (0 - 0) / 1e-6 = 0.
    The 15-feature architecture is preserved unchanged.

    Affected features in the new active dataset:
        tx_failed_rate  — always 0 (no failed packets in this setup)
        low_mcs_ratio   — always 0 (all transmissions above MCS index 3)
    """
    zero_idx = np.where(scaler.scale_ == 0)[0]
    if len(zero_idx) == 0:
        return scaler
    const_names = [feature_names[i] for i in zero_idx]
    log_fn(f"  [SCALER] Constant features detected (std=0): {const_names}")
    log_fn(f"  [SCALER] Patching scale_ from 0 → 1e-6 to prevent NaN scaling.")
    log_fn(f"  [SCALER] These features will normalise to 0.0 for all rows.")
    scaler.scale_ = np.where(scaler.scale_ == 0, 1e-6, scaler.scale_)
    return scaler

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_lines = []
    def log(m=''):
        print(m); log_lines.append(m)

    start = datetime.now()

    section("1. LOADING DATA")
    df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'])
    df = df[~df['client_mac'].isin(EXCLUDE_MACS)].copy()
    df_train = df[df['client_mac'] != VALIDATION_MAC].copy()
    df_val_c = df[df['client_mac'] == VALIDATION_MAC].copy()
    log(f"  Training rows : {len(df_train):,}")
    log(f"  Val client    : {VALIDATION_MAC} ({len(df_val_c):,} rows)")

    section("2. SPLITTING BY ACTIVITY")
    df_idle   = df_train[df_train['is_active'] == 0]
    df_active = df_train[df_train['is_active'] == 1]
    log(f"  Idle   : {len(df_idle):,} rows")
    log(f"  Active : {len(df_active):,} rows")

    section("3. TRAINING IDLE AUTOENCODER")

    # Guard: the new active dataset has very few idle rows (8 total).
    # Training BatchNorm layers on fewer than MIN_IDLE_ROWS rows produces
    # unreliable running statistics and a meaningless validation split.
    # In this case, skip idle training and retain the previously trained
    # idle model from the original two-day normal collection.
    if len(df_idle) < MIN_IDLE_ROWS:
        log(f"\n  [IDLE] WARNING: Only {len(df_idle)} idle rows found.")
        log(f"  [IDLE] Minimum required: {MIN_IDLE_ROWS} rows.")
        log(f"  [IDLE] Idle model training SKIPPED.")
        log(f"  [IDLE] ACTION REQUIRED: Keep the idle model from the original")
        log(f"  [IDLE] two-day training run (ae_idle_model.pt, ae_scaler_idle.pkl).")
        log(f"  [IDLE] Only the active model will be updated by this run.")
        # Provide placeholder values so metadata saves without error
        thr_idle      = float('nan')
        thr_idle_dict = {f"threshold_pct{p}": float('nan')
                         for p in THRESHOLD_PERCENTILES}
        idle_best     = float('nan')
        n_removed_idle        = 0
        idle_outlier_feats    = []
        df_idle_clean         = df_idle.copy()
    else:
        # Suggestions 3 & 4 — remove outliers before fitting scaler and training
        df_idle_clean, n_removed_idle, idle_outlier_feats = remove_outliers(
            df_idle, IDLE_FEATURES, "IDLE", log)

        X_idle  = df_idle_clean[IDLE_FEATURES].values.astype(np.float32)
        sc_idle = StandardScaler()
        sc_idle.fit(X_idle)
        sc_idle = patch_scaler_for_constant_features(sc_idle, IDLE_FEATURES, log)
        Xs_idle = sc_idle.transform(X_idle)
        n_val   = max(1, int(len(Xs_idle) * VAL_SPLIT))
        idx     = np.random.permutation(len(Xs_idle))
        ae_idle, idle_best = train_ae(
            Xs_idle[idx[n_val:]], Xs_idle[idx[:n_val]],
            input_dim=3, hidden_dims=[8], latent_dim=4, name="IDLE"
        )
        thr_idle, errs_idle, thr_idle_dict = compute_threshold(ae_idle, Xs_idle)
        flagged_idle = (errs_idle > thr_idle).sum()

        # Suggestion 2 — report all percentile thresholds
        log(f"\n  Threshold sweep (idle):")
        for pct in THRESHOLD_PERCENTILES:
            key = f"threshold_pct{pct}"
            t   = thr_idle_dict[key]
            fa  = (errs_idle > t).sum()
            log(f"    {pct}th pct: threshold={t:.6f}  "
                f"flagged={fa} ({100*fa/len(errs_idle):.2f}%)")
        log(f"  Primary threshold ({THRESHOLD_PCT}th pct): {thr_idle:.6f}")

        torch.save(ae_idle.state_dict(), f"{MODELS_DIR}/ae_idle_model.pt")
        np.save(f"{MODELS_DIR}/ae_idle_threshold.npy", thr_idle)
        with open(f"{MODELS_DIR}/ae_scaler_idle.pkl", 'wb') as f:
            pickle.dump(sc_idle, f)
        log(f"  Saved -> {MODELS_DIR}/ae_idle_model.pt")

    section("4. TRAINING ACTIVE AUTOENCODER")

    # Suggestions 3 & 4 — remove outliers (this is where the MSE=4962 rows live)
    df_active_clean, n_removed_active, active_outlier_feats = remove_outliers(
        df_active, ACTIVE_FEATURES, "ACTIVE", log)

    X_active  = df_active_clean[ACTIVE_FEATURES].values.astype(np.float32)
    sc_active = StandardScaler()
    sc_active.fit(X_active)
    # Fix zero-variance features (tx_failed_rate, low_mcs_ratio in new dataset).
    # Must be done before transform — see patch_scaler_for_constant_features().
    sc_active  = patch_scaler_for_constant_features(sc_active, ACTIVE_FEATURES, log)
    Xs_active  = sc_active.transform(X_active)
    n_val = max(1, int(len(Xs_active) * VAL_SPLIT))
    idx   = np.random.permutation(len(Xs_active))
    ae_active, active_best = train_ae(
        Xs_active[idx[n_val:]], Xs_active[idx[:n_val]],
        input_dim=15, hidden_dims=[32, 16], latent_dim=8, name="ACTIVE"
    )
    thr_active, errs_active, thr_active_dict = compute_threshold(ae_active, Xs_active)
    flagged_active = (errs_active > thr_active).sum()

    # Suggestion 2 — report all percentile thresholds
    log(f"\n  Threshold sweep (active):")
    for pct in THRESHOLD_PERCENTILES:
        key = f"threshold_pct{pct}"
        t   = thr_active_dict[key]
        fa  = (errs_active > t).sum()
        log(f"    {pct}th pct: threshold={t:.6f}  "
            f"flagged={fa} ({100*fa/len(errs_active):.2f}%)")
    log(f"  Primary threshold ({THRESHOLD_PCT}th pct): {thr_active:.6f}")

    torch.save(ae_active.state_dict(), f"{MODELS_DIR}/ae_active_model.pt")
    np.save(f"{MODELS_DIR}/ae_active_threshold.npy", thr_active)
    with open(f"{MODELS_DIR}/ae_scaler_active.pkl", 'wb') as f: pickle.dump(sc_active, f)
    log(f"  Saved -> {MODELS_DIR}/ae_active_model.pt")

    section("5. SAVING METADATA")
    meta = {
        'trained_at'              : start.isoformat(),
        'idle_architecture'       : '3->[8]->4->[8]->3',
        'active_architecture'     : '15->[32,16]->8->[16,32]->15',
        'idle_features'           : IDLE_FEATURES,
        'active_features'         : ACTIVE_FEATURES,
        'outlier_removal_applied' : REMOVE_OUTLIERS,
        'outlier_iqr_factor'      : OUTLIER_IQR_FACTOR,
        'idle_rows_before_clean'  : int(len(df_idle)),
        'idle_rows_after_clean'   : int(len(df_idle_clean)),
        'idle_outliers_removed'   : int(n_removed_idle),
        'active_rows_before_clean': int(len(df_active)),
        'active_rows_after_clean' : int(len(df_active_clean)),
        'active_outliers_removed' : int(n_removed_active),
        # Suggestion 2 — all three thresholds saved
        # To switch: update 'idle_threshold' and 'active_threshold' below
        # with values from thr_idle_dict / thr_active_dict
        'threshold_percentile'    : THRESHOLD_PCT,
        'idle_threshold'          : float(thr_idle),
        'active_threshold'        : float(thr_active),
        'idle_thresholds_all'     : thr_idle_dict,
        'active_thresholds_all'   : thr_active_dict,
        'learning_rate'           : LEARNING_RATE,
        'batch_size'              : BATCH_SIZE,
        'max_epochs'              : MAX_EPOCHS,
        'early_stopping_patience' : PATIENCE,
        'best_idle_val_loss'      : float(idle_best),
        'best_active_val_loss'    : float(active_best),
    }
    with open(f"{MODELS_DIR}/ae_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    log(f"\n  Threshold sweep summary:")
    log(f"  {'Percentile':>12}  {'Idle thr':>12}  {'Active thr':>12}")
    log(f"  {'-'*12}  {'-'*12}  {'-'*12}")
    for pct in THRESHOLD_PERCENTILES:
        k = f"threshold_pct{pct}"
        log(f"  {pct:>12}  {thr_idle_dict[k]:>12.6f}  {thr_active_dict[k]:>12.6f}")
    log(f"\n  To use 95th pct threshold: update ae_metadata.json:")
    log(f"    idle_threshold   -> {thr_idle_dict['threshold_pct95']:.6f}")
    log(f"    active_threshold -> {thr_active_dict['threshold_pct95']:.6f}")

    section("TRAINING COMPLETE")
    log(f"  Duration : {(datetime.now()-start).total_seconds():.1f}s")
    log(f"  Models   : {MODELS_DIR}/ae_*.pt")
    log()
    log("  NEXT STEP: python autoencoder_evaluator.py <experiment_csv>")

    with open(
        f"{RESULTS_DIR}/ae_training_report.txt",
        'w',
        encoding='utf-8'
    ) as f:
        f.write('\n'.join(log_lines))

if __name__ == '__main__':
    main()