#!/usr/bin/env python3
"""
isolation_forest_training.py
============================
Isolation Forest anomaly detection for WiFi node telemetry.

WHY TWO MODELS:
    98.23% of rows in this dataset are idle (is_active=0).
    A single model trained on all rows learns "idle=normal" and would
    flag ANY activity as an anomaly. Instead we train two specialised models:

    Model 1 — IDLE STATE (78,234 rows)
        Trained on idle rows. Features: RSSI, throughput averages.
        Detects: RSSI drops, sustained throughput changes during idle.

    Model 2 — ACTIVE STATE (1,413 rows)
        Trained on active rows. Features: all 17 traffic + MCS features.
        Detects: retry spikes, MCS degradation, tx_success drops,
                 unusual traffic volumes, security errors.

USAGE:
    python isolation_forest_training.py

INPUT:
    ml_ready_data.csv  (output of feature_engineering.py)

OUTPUTS:
    models/if_idle_model.pkl       — trained idle-state IF model
    models/if_active_model.pkl     — trained active-state IF model
    models/scaler_idle.pkl         — scaler for idle features
    models/scaler_active.pkl       — scaler for active features
    models/model_metadata.json     — thresholds, feature lists, stats
    results/if_scores_all.csv      — full dataset with anomaly scores
    results/if_anomalies.csv       — rows flagged as anomalies
    results/if_training_report.txt — human-readable summary
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_FILE   = "ml_ready_data.csv"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"

# Clients to exclude (transient — less than 100 rows)
EXCLUDE_MACS = ['a8:74:1d:be:20:48', 'a8:74:1d:47:e6:4e']

# Client to hold out for validation (intermittent — joined mid-session)
VALIDATION_MAC = 'a8:74:1d:47:e6:46'

# Isolation Forest hyperparameters
# contamination: expected fraction of anomalies in training data
# Start at 0.01 (1%). Increase if too few alerts; decrease if too many.
IF_CONTAMINATION  = 0.01
IF_N_ESTIMATORS   = 200    # more trees = more stable scores
IF_RANDOM_STATE   = 42

# Anomaly score threshold — scores below this are flagged as anomalies
# IsolationForest predict() uses -1=anomaly, 1=normal
# score_samples() returns raw scores: lower = more anomalous
# We use the model's built-in threshold (contamination parameter drives this)
# but also save the raw score for custom thresholding later

# Features for each model
IDLE_FEATURES = [
    'rssi'
]

ACTIVE_FEATURES = [
    'tx_packet_rate',
    'rx_packet_rate',
    'tx_byte_rate',
    'rx_byte_rate',
    'tx_success_rate',
    'retry_rate',
    'tx_failed_rate',
    'rssi',
    'tx_throughput_mbps',
    'rx_throughput_mbps',
    'low_mcs_ratio',
    'mid_mcs_ratio',
    'high_mcs_ratio',
    'mcs_weighted_score',
    'mcs_entropy',
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"    Saved: {path}")


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    make_dirs()
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    start_time = datetime.now()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    section("1. LOADING DATA")
    df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'])

    log(f"  Raw rows loaded   : {len(df):,}")
    log(f"  Unique clients    : {df['client_mac'].nunique()}")

    # Exclude transient clients
    df = df[~df['client_mac'].isin(EXCLUDE_MACS)].copy()
    log(f"  After exclusions  : {len(df):,} rows, {df['client_mac'].nunique()} clients")

    # Split training vs held-out validation client
    df_val   = df[df['client_mac'] == VALIDATION_MAC].copy()
    df_train = df[df['client_mac'] != VALIDATION_MAC].copy()
    log(f"  Training clients  : {df_train['client_mac'].nunique()}")
    log(f"  Validation client : {VALIDATION_MAC} ({len(df_val):,} rows)")

    # ── 2. Split by activity state ────────────────────────────────────────────
    section("2. SPLITTING BY ACTIVITY STATE")

    df_idle   = df_train[df_train['is_active'] == 0].copy()
    df_active = df_train[df_train['is_active'] == 1].copy()

    log(f"  Idle rows   : {len(df_idle):,}  ({100*len(df_idle)/len(df_train):.1f}%)")
    log(f"  Active rows : {len(df_active):,}  ({100*len(df_active)/len(df_train):.1f}%)")

    if len(df_active) < 100:
        log("  [WARN] Very few active rows — active model may be unreliable.")
        log("         Collect more data during active usage periods.")

    # ── 3. Print active-row feature stats (sanity check) ─────────────────────
    section("3. ACTIVE ROW SANITY CHECK")
    stats = df_active[ACTIVE_FEATURES].describe().round(4)
    log("  Feature stats for ACTIVE rows (is_active=1):")
    log(stats.to_string())
    log("")
    log(f"  tx_success_rate mean on active rows: "
        f"{df_active['tx_success_rate'].mean():.4f}  (should be close to 1.0)")
    log(f"  rssi mean on active rows           : "
        f"{df_active['rssi'].mean():.1f}")

    # ── 4. Train IDLE model ───────────────────────────────────────────────────
    section("4. TRAINING IDLE-STATE MODEL")

    X_idle = df_idle[IDLE_FEATURES].values
    scaler_idle = StandardScaler()
    X_idle_scaled = scaler_idle.fit_transform(X_idle)

    if_idle = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    if_idle.fit(X_idle_scaled)

    idle_scores = if_idle.score_samples(X_idle_scaled)
    idle_preds  = if_idle.predict(X_idle_scaled)
    n_idle_anom = (idle_preds == -1).sum()

    log(f"  Training rows     : {len(X_idle):,}")
    log(f"  Features used     : {IDLE_FEATURES}")
    log(f"  Anomalies flagged : {n_idle_anom} ({100*n_idle_anom/len(X_idle):.2f}%)")
    log(f"  Score range       : [{idle_scores.min():.4f}, {idle_scores.max():.4f}]")
    log(f"  Score mean        : {idle_scores.mean():.4f}")

    save_pickle(if_idle,     f"{MODELS_DIR}/if_idle_model.pkl")
    save_pickle(scaler_idle, f"{MODELS_DIR}/scaler_idle.pkl")

    # ── 5. Train ACTIVE model ─────────────────────────────────────────────────
    section("5. TRAINING ACTIVE-STATE MODEL")

    X_active = df_active[ACTIVE_FEATURES].values
    scaler_active = StandardScaler()
    X_active_scaled = scaler_active.fit_transform(X_active)

    # With only ~1413 active rows, use lower contamination
    # to avoid flagging too many of the limited normal samples
    active_contamination = min(IF_CONTAMINATION, 0.05)

    if_active = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=active_contamination,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    if_active.fit(X_active_scaled)

    active_scores = if_active.score_samples(X_active_scaled)
    active_preds  = if_active.predict(X_active_scaled)
    n_active_anom = (active_preds == -1).sum()

    log(f"  Training rows     : {len(X_active):,}")
    log(f"  Features used     : {len(ACTIVE_FEATURES)} features")
    log(f"  Anomalies flagged : {n_active_anom} ({100*n_active_anom/len(X_active):.2f}%)")
    log(f"  Score range       : [{active_scores.min():.4f}, {active_scores.max():.4f}]")
    log(f"  Score mean        : {active_scores.mean():.4f}")

    save_pickle(if_active,     f"{MODELS_DIR}/if_active_model.pkl")
    save_pickle(scaler_active, f"{MODELS_DIR}/scaler_active.pkl")

    # ── 6. Score entire dataset ───────────────────────────────────────────────
    section("6. SCORING FULL DATASET")

    df_out = df.copy()
    df_out['anomaly_score'] = np.nan
    df_out['anomaly_flag']  = 0      # 0=normal, 1=anomaly
    df_out['model_used']    = 'none'

    # Score idle rows
    idle_mask = df_out['is_active'] == 0
    if idle_mask.sum() > 0:
        X_s = scaler_idle.transform(df_out.loc[idle_mask, IDLE_FEATURES].values)
        df_out.loc[idle_mask, 'anomaly_score'] = if_idle.score_samples(X_s)
        df_out.loc[idle_mask, 'anomaly_flag']  = (if_idle.predict(X_s) == -1).astype(int)
        df_out.loc[idle_mask, 'model_used']    = 'idle'

    # Score active rows
    active_mask = df_out['is_active'] == 1
    if active_mask.sum() > 0:
        X_s = scaler_active.transform(df_out.loc[active_mask, ACTIVE_FEATURES].values)
        df_out.loc[active_mask, 'anomaly_score'] = if_active.score_samples(X_s)
        df_out.loc[active_mask, 'anomaly_flag']  = (if_active.predict(X_s) == -1).astype(int)
        df_out.loc[active_mask, 'model_used']    = 'active'

    total_anomalies = df_out['anomaly_flag'].sum()
    log(f"  Total rows scored    : {len(df_out):,}")
    log(f"  Total anomalies      : {total_anomalies} ({100*total_anomalies/len(df_out):.2f}%)")

    # Per-client anomaly summary
    log("\n  Anomalies per client:")
    summary = df_out.groupby('client_mac')['anomaly_flag'].agg(['sum','count'])
    summary['pct'] = (100 * summary['sum'] / summary['count']).round(2)
    summary.columns = ['anomalies', 'total_rows', 'anomaly_pct']
    log(summary.to_string())

    # ── 7. Save results ───────────────────────────────────────────────────────
    section("7. SAVING RESULTS")

    df_out.to_csv(f"{RESULTS_DIR}/if_scores_all.csv", index=False)
    log(f"  Full scored dataset : {RESULTS_DIR}/if_scores_all.csv")

    df_anomalies = df_out[df_out['anomaly_flag'] == 1].copy()
    df_anomalies.to_csv(f"{RESULTS_DIR}/if_anomalies.csv", index=False)
    log(f"  Anomalies only      : {RESULTS_DIR}/if_anomalies.csv  ({len(df_anomalies):,} rows)")

    # ── 8. Save metadata ──────────────────────────────────────────────────────
    metadata = {
        'trained_at'           : start_time.isoformat(),
        'input_file'           : INPUT_FILE,
        'contamination'        : IF_CONTAMINATION,
        'n_estimators'         : IF_N_ESTIMATORS,
        'idle_features'        : IDLE_FEATURES,
        'active_features'      : ACTIVE_FEATURES,
        'idle_train_rows'      : int(len(df_idle)),
        'active_train_rows'    : int(len(df_active)),
        'idle_anomalies_train' : int(n_idle_anom),
        'active_anomalies_train': int(n_active_anom),
        'exclude_macs'         : EXCLUDE_MACS,
        'validation_mac'       : VALIDATION_MAC,
    }
    with open(f"{MODELS_DIR}/model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    log(f"  Metadata            : {MODELS_DIR}/model_metadata.json")

    # ── 9. Validation client scoring ──────────────────────────────────────────
    section("8. VALIDATION CLIENT SCORING")
    if len(df_val) > 0:
        df_val_out = df_val.copy()
        df_val_out['anomaly_score'] = np.nan
        df_val_out['anomaly_flag']  = 0
        df_val_out['model_used']    = 'none'

        idle_v  = df_val_out['is_active'] == 0
        active_v = df_val_out['is_active'] == 1

        if idle_v.sum() > 0:
            X_s = scaler_idle.transform(df_val_out.loc[idle_v, IDLE_FEATURES].values)
            df_val_out.loc[idle_v, 'anomaly_score'] = if_idle.score_samples(X_s)
            df_val_out.loc[idle_v, 'anomaly_flag']  = (if_idle.predict(X_s) == -1).astype(int)
            df_val_out.loc[idle_v, 'model_used']    = 'idle'

        if active_v.sum() > 0:
            X_s = scaler_active.transform(df_val_out.loc[active_v, ACTIVE_FEATURES].values)
            df_val_out.loc[active_v, 'anomaly_score'] = if_active.score_samples(X_s)
            df_val_out.loc[active_v, 'anomaly_flag']  = (if_active.predict(X_s) == -1).astype(int)
            df_val_out.loc[active_v, 'model_used']    = 'active'

        val_anom = df_val_out['anomaly_flag'].sum()
        log(f"  Validation client rows    : {len(df_val_out):,}")
        log(f"  Anomalies flagged         : {val_anom} ({100*val_anom/len(df_val_out):.2f}%)")
        log(f"  (Expected: similar rate to training clients if behaviour is normal)")

        df_val_out.to_csv(f"{RESULTS_DIR}/if_validation_client.csv", index=False)
        log(f"  Saved: {RESULTS_DIR}/if_validation_client.csv")

    # ── 10. Final report ──────────────────────────────────────────────────────
    section("TRAINING COMPLETE")
    duration = (datetime.now() - start_time).total_seconds()
    log(f"  Duration            : {duration:.1f} seconds")
    log(f"  Models saved in     : {MODELS_DIR}/")
    log(f"  Results saved in    : {RESULTS_DIR}/")
    log("")
    log("  NEXT STEPS:")
    log("  1. Check results/if_anomalies.csv — do flagged rows make sense?")
    log("  2. Check anomaly rate per client — any client >5% is suspicious")
    log("     OR your contamination parameter needs adjustment")
    log("  3. Adjust IF_CONTAMINATION up/down and retrain if needed:")
    log("     Too many alerts -> lower contamination (e.g. 0.005)")
    log("     Too few alerts  -> raise contamination (e.g. 0.02)")
    log("  4. Once satisfied with IF results -> proceed to Autoencoder")

    with open(f"{RESULTS_DIR}/if_training_report.txt", 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report saved: {RESULTS_DIR}/if_training_report.txt")


if __name__ == '__main__':
    main()
