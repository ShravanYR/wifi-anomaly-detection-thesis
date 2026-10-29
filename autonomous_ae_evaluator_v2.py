"""
autonomous_ae_evaluator.py  (v2 — updated for models_active)
=============================================================
Changes from v1:
  1. MODELS_DIR  → "models_active"  (new active model trained on 78,265 rows)
  2. RESULTS_DIR → "results_active" (keeps results separate from old experiments)
  3. np.nan_to_num guard added after each .transform() call (safety net for
     constant features tx_failed_rate / low_mcs_ratio whose scaler.scale_
     was patched to 1e-6 during training)

PRE-RUN (run once in PowerShell before executing this script):
  copy models\\ae_idle_model.pt       models_active\\ae_idle_model.pt
  copy models\\ae_idle_threshold.npy  models_active\\ae_idle_threshold.npy
  copy models\\ae_scaler_idle.pkl     models_active\\ae_scaler_idle.pkl

USAGE:
  python autonomous_ae_evaluator.py <ml_ready_csv>
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("[ERROR] PyTorch not installed. Run: pip install torch")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

# CHANGE 1 — point to new model folder
MODELS_DIR   = "models_active"

# CHANGE 2 — separate results folder
RESULTS_DIR  = "results_active"

MIN_ROWS          = 20
BASELINE_MAX      = 30
BASELINE_FRACTION = 0.30

THRESHOLD_WARNING_DEFAULT  = 0.05
THRESHOLD_CRITICAL_DEFAULT = 0.20

PELT_PENALTY = 10

IDLE_FEATURES = ['rssi', 'tx_success_rate', 'retry_rate']

ACTIVE_FEATURES = [
    'tx_packet_rate', 'rx_packet_rate', 'tx_byte_rate', 'rx_byte_rate',
    'tx_success_rate', 'retry_rate', 'tx_failed_rate',
    'rssi', 'tx_throughput_mbps', 'rx_throughput_mbps',
    'low_mcs_ratio', 'mid_mcs_ratio', 'high_mcs_ratio',
    'mcs_weighted_score', 'mcs_entropy',
]

EXPLAIN_FEATURES = [
    'retry_rate', 'tx_failed_rate', 'tx_success_rate',
    'tx_packet_rate', 'rx_packet_rate',
    'rssi', 'mcs_entropy', 'low_mcs_ratio',
    'tx_throughput_mbps', 'rx_throughput_mbps',
]


# ── AE model definition (must match autoencoder_training.py exactly) ──────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                    nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        enc += [nn.Linear(prev, latent_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                    nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        self.eval()
        with torch.no_grad():
            return torch.mean((x - self.forward(x)) ** 2, dim=1).cpu().numpy()


# ── Load models and thresholds ────────────────────────────────────────────────

def load_models():
    def load_pkl(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    meta_path = f"{MODELS_DIR}/ae_metadata.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"ae_metadata.json not found in {MODELS_DIR}/\n"
            f"Run autoencoder_training_v2.py first, then copy idle "
            f"model files into {MODELS_DIR}/."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    thr_idle   = meta['idle_threshold']
    thr_active = meta['active_threshold']
    thr_pct    = meta.get('threshold_percentile', 99)

    ae_idle = Autoencoder(input_dim=3, hidden_dims=[8], latent_dim=4)
    ae_idle.load_state_dict(
        torch.load(f"{MODELS_DIR}/ae_idle_model.pt", map_location='cpu'))
    ae_idle.eval()

    ae_active = Autoencoder(input_dim=15, hidden_dims=[32, 16], latent_dim=8)
    ae_active.load_state_dict(
        torch.load(f"{MODELS_DIR}/ae_active_model.pt", map_location='cpu'))
    ae_active.eval()

    print(f"[INFO] AE model loaded — trained at: {meta.get('trained_at','unknown')}")
    print(f"[INFO] Thresholds ({thr_pct}th percentile from training):")
    print(f"         idle   = {thr_idle:.6f}")
    print(f"         active = {thr_active:.6f}")

    if 'active_thresholds_all' in meta:
        print(f"[INFO] All available active thresholds:")
        for k, v in sorted(meta['active_thresholds_all'].items()):
            marker = ' ← active' if abs(v - thr_active) < 1e-9 else ''
            print(f"         {k:20s} = {v:.6f}{marker}")
        print(f"         To switch: update 'active_threshold' in ae_metadata.json")

    if meta.get('outlier_removal_applied'):
        print(f"[INFO] Outlier removal applied during training:")
        print(f"         idle   removed: {meta.get('idle_outliers_removed', 0)}")
        print(f"         active removed: {meta.get('active_outliers_removed', 0)}")

    print(f"[INFO] Training rows — idle: {meta.get('idle_train_rows','?')}  "
          f"active: {meta.get('active_train_rows','?')}")

    thr_warn = meta.get('status_threshold_warning',  THRESHOLD_WARNING_DEFAULT)
    thr_crit = meta.get('status_threshold_critical', THRESHOLD_CRITICAL_DEFAULT)
    if 'status_threshold_warning' in meta:
        print(f"[INFO] Status thresholds: calibrated "
              f"(warning={thr_warn:.2%}, critical={thr_crit:.2%})")
    else:
        print(f"[INFO] Status thresholds: using defaults "
              f"(warning={thr_warn:.2%}, critical={thr_crit:.2%})")
        print(f"         Calibrate after experiments by adding "
              f"'status_threshold_warning' to ae_metadata.json")

    return {
        'idle_model'        : ae_idle,
        'active_model'      : ae_active,
        'scaler_idle'       : load_pkl(f"{MODELS_DIR}/ae_scaler_idle.pkl"),
        'scaler_active'     : load_pkl(f"{MODELS_DIR}/ae_scaler_active.pkl"),
        'threshold_idle'    : thr_idle,
        'threshold_active'  : thr_active,
        'threshold_warning' : thr_warn,
        'threshold_critical': thr_crit,
        'meta'              : meta,
    }


# ── Score all rows through AE ─────────────────────────────────────────────────

def score_rows_ae(df, models):
    df = df.copy()
    df['anomaly_score'] = np.nan
    df['anomaly_flag']  = 0
    df['model_used']    = 'none'

    idle_mask   = df['is_active'] == 0
    active_mask = df['is_active'] == 1
    avail_idle   = [f for f in IDLE_FEATURES   if f in df.columns]
    avail_active = [f for f in ACTIVE_FEATURES if f in df.columns]

    if idle_mask.sum() > 0 and len(avail_idle) == len(IDLE_FEATURES):
        X = models['scaler_idle'].transform(
            df.loc[idle_mask, avail_idle].values.astype(np.float32))
        # CHANGE 3 — safety net for constant features with scale_=1e-6
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        errors = models['idle_model'].reconstruction_error(
            torch.FloatTensor(X))
        df.loc[idle_mask, 'anomaly_score'] = errors
        df.loc[idle_mask, 'anomaly_flag']  = (
            errors > models['threshold_idle']).astype(int)
        df.loc[idle_mask, 'model_used'] = 'idle_ae'

    if active_mask.sum() > 0 and len(avail_active) > 0:
        X = models['scaler_active'].transform(
            df.loc[active_mask, avail_active].values.astype(np.float32))
        # CHANGE 3 — safety net: constant features (tx_failed_rate, low_mcs_ratio)
        # have scale_=1e-6 in the patched scaler. If experiment data has any
        # non-zero value for these, the scaled result is very large but finite.
        # We only guard against true NaN/Inf which should not occur.
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        errors = models['active_model'].reconstruction_error(
            torch.FloatTensor(X))
        df.loc[active_mask, 'anomaly_score'] = errors
        df.loc[active_mask, 'anomaly_flag']  = (
            errors > models['threshold_active']).astype(int)
        df.loc[active_mask, 'model_used'] = 'active_ae'

    return df


# ── Composite stress signal ───────────────────────────────────────────────────

def build_stress_signal(df_client):
    signal = pd.Series(0.0, index=df_client.index)
    for col, invert in [
        ('retry_rate',         False),
        ('tx_failed_rate',     False),
        ('tx_success_rate',    True),
        ('tx_throughput_mbps', True),
    ]:
        if col in df_client.columns:
            v  = df_client[col].fillna(0).astype(float)
            mx = v.abs().max()
            if mx > 0:
                signal += (1 - v / mx) if invert else (v / mx)
    return signal.rolling(window=3, min_periods=1, center=True).mean().values


# ── Change point detection ────────────────────────────────────────────────────

def find_change_point(signal, min_size=5):
    if len(signal) < min_size * 2:
        return None, 'insufficient_data', 'none'

    split  = None
    method = 'sliding_window'

    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        algo.fit(signal.reshape(-1, 1))
        bps = [b for b in algo.predict(pen=PELT_PENALTY) if b < len(signal)]
        if bps:
            split  = bps[0]
            method = 'PELT'
    except ImportError:
        pass

    if split is None:
        best_diff = -1
        for i in range(min_size, len(signal) - min_size):
            d = abs(signal[i:].mean() - signal[:i].mean())
            if d > best_diff:
                best_diff = d
                split = i

    if split is None:
        return None, 'no_change', 'none'

    shift      = abs(signal[split:].mean() - signal[:split].mean())
    confidence = ('high'   if shift > 0.3 else
                  'medium' if shift > 0.1 else 'low')
    return split, method, confidence


# ── Explain what changed ──────────────────────────────────────────────────────

def explain_anomaly_ae(df_baseline, df_anomaly, models):
    explanations  = []
    base_active   = df_baseline[df_baseline['is_active'] == 1]
    anom_active   = df_anomaly[df_anomaly['is_active'] == 1]

    if len(base_active) == 0 or len(anom_active) == 0:
        base_active = df_baseline
        anom_active = df_anomaly

    avail = [f for f in EXPLAIN_FEATURES
             if f in base_active.columns and f in anom_active.columns]

    for feat in avail:
        b_mean = float(base_active[feat].mean())
        a_mean = float(anom_active[feat].mean())
        if b_mean == 0:
            if abs(a_mean) > 0.001:
                explanations.append((feat, b_mean, a_mean,
                                     float('inf'), 'appeared'))
        else:
            pct = 100 * (a_mean - b_mean) / abs(b_mean)
            if abs(pct) > 5:
                direction = 'increased' if pct > 0 else 'decreased'
                explanations.append((feat, b_mean, a_mean, pct, direction))

    explanations.sort(
        key=lambda x: abs(x[3]) if x[3] != float('inf') else 9999,
        reverse=True)
    return explanations[:5]


# ── Evaluate one client ───────────────────────────────────────────────────────

def evaluate_client_ae(mac, df_client, models):
    df_client = df_client.sort_values('timestamp').reset_index(drop=True)

    result = {
        'client_mac'             : mac,
        'total_rows'             : len(df_client),
        'active_rows'            : int(df_client['is_active'].sum()),
        'status'                 : 'NORMAL',
        'anomaly_rate'           : 0.0,
        'anomaly_rate_all_rows'  : 0.0,
        'anomaly_start_ts'       : None,
        'change_point_method'    : None,
        'change_confidence'      : None,
        'score_baseline_median'  : None,
        'score_anomaly_median'   : None,
        'score_ratio'            : None,
        'flagged_rows'           : 0,
        'n_baseline'             : 0,
        'top_features'           : [],
    }

    if len(df_client) < MIN_ROWS:
        result['status'] = 'INSUFFICIENT_DATA'
        return result

    n_baseline = min(BASELINE_MAX,
                     max(3, int(len(df_client) * BASELINE_FRACTION)))

    df_scored              = score_rows_ae(df_client, models)
    result['flagged_rows'] = int(df_scored['anomaly_flag'].sum())
    result['anomaly_rate'] = round(result['flagged_rows'] / len(df_scored), 4)
    result['anomaly_rate_all_rows'] = result['anomaly_rate']
    result['n_baseline']   = n_baseline

    active_rows = df_scored[df_scored['is_active'] == 1]
    signal_src  = active_rows if len(active_rows) >= MIN_ROWS else df_scored
    signal      = build_stress_signal(signal_src)

    split_in_src, method, confidence = find_change_point(signal)
    result['change_point_method'] = method
    result['change_confidence']   = confidence

    if split_in_src is not None and method not in ('no_change',
                                                    'insufficient_data'):
        src_indices  = signal_src.index.tolist()
        global_split = src_indices[min(split_in_src, len(src_indices) - 1)]
        split_ts     = df_scored.loc[global_split, 'timestamp']
        result['anomaly_start_ts'] = str(split_ts)

        df_before = df_scored[df_scored.index <  global_split]
        df_after  = df_scored[df_scored.index >= global_split]
        df_baseline = df_scored.iloc[:n_baseline]

        df_before_active = df_before[df_before['is_active'] == 1]
        df_after_active  = df_after[df_after['is_active'] == 1]

        if len(df_before_active) > 0:
            result['score_baseline_median'] = round(
                float(df_before_active['anomaly_score'].median()), 6)
        if len(df_after_active) > 0:
            result['score_anomaly_median'] = round(
                float(df_after_active['anomaly_score'].median()), 6)

        if (result['score_baseline_median'] and
                result['score_baseline_median'] > 0 and
                result['score_anomaly_median']):
            result['score_ratio'] = round(
                result['score_anomaly_median'] /
                result['score_baseline_median'], 3)

        after_active = df_after[df_after['is_active'] == 1]
        if len(after_active) > 0:
            result['anomaly_rate'] = round(
                after_active['anomaly_flag'].sum() / len(after_active), 4)
        result['anomaly_rate_all_rows'] = round(
            df_after['anomaly_flag'].sum() / len(df_after),
            4) if len(df_after) > 0 else 0.0

        result['top_features'] = explain_anomaly_ae(df_baseline, df_after, models)
    else:
        df_baseline = df_scored.iloc[:n_baseline]
        rest        = df_scored.iloc[n_baseline:]

        df_baseline_active = df_baseline[df_baseline['is_active'] == 1]
        rest_active        = rest[rest['is_active'] == 1]

        if len(df_baseline_active) > 0:
            result['score_baseline_median'] = round(
                float(df_baseline_active['anomaly_score'].median()), 6)
        if len(rest_active) > 0:
            result['score_anomaly_median'] = round(
                float(rest_active['anomaly_score'].median()), 6)

        if (result['score_baseline_median'] and
                result['score_baseline_median'] > 0 and
                result['score_anomaly_median']):
            result['score_ratio'] = round(
                result['score_anomaly_median'] /
                result['score_baseline_median'], 3)

    thr_warn = models['threshold_warning']
    thr_crit = models['threshold_critical']
    anom_rate = result['anomaly_rate']
    ratio     = result['score_ratio'] or 1.0

    if anom_rate >= thr_crit:
        result['status'] = 'CRITICAL'
    elif anom_rate >= thr_warn:
        result['status'] = 'WARNING'
    elif ratio > 1.5 and confidence in ('medium', 'high'):
        result['status'] = 'WATCH'
    else:
        result['status'] = 'NORMAL'

    result['_df_scored'] = df_scored
    return result


# ── Print client report ───────────────────────────────────────────────────────

STATUS_ICONS = {
    'CRITICAL'         : '🔴  CRITICAL',
    'WARNING'          : '🟡  WARNING',
    'WATCH'            : '🔵  WATCH',
    'NORMAL'           : '✅  NORMAL',
    'INSUFFICIENT_DATA': '⚪  SKIP (too few rows)',
}

def print_client_report(r, thr_active):
    icon   = STATUS_ICONS.get(r['status'], r['status'])
    total  = r['total_rows']
    active = r['active_rows']

    print(f"\n  {'─'*54}")
    print(f"  Client : {r['client_mac']}")
    print(f"  Status : {icon}")
    print(f"  Rows   : {total} total  |  "
          f"{active} active ({100*active/total:.1f}%)")
    print(f"  Flagged: {r['flagged_rows']} rows  |  "
          f"Anomaly rate (active): {100*r['anomaly_rate']:.1f}%  |  "
          f"(all rows): {100*r.get('anomaly_rate_all_rows', r['anomaly_rate']):.1f}%")

    if r['anomaly_start_ts']:
        print(f"  Anomaly detected at  : {r['anomaly_start_ts']}")
        print(f"  Detection method     : {r['change_point_method']}  "
              f"(confidence: {r['change_confidence']})")

    if r['score_baseline_median'] is not None:
        print(f"  Reconstruction error (MSE):")
        print(f"    Baseline median: {r['score_baseline_median']:.6f}")
        if r['score_anomaly_median'] is not None:
            print(f"    Anomaly median : {r['score_anomaly_median']:.6f}")
            if r['score_ratio'] is not None:
                direction = ('AE senses anomaly'
                             if (r['score_ratio'] or 0) > 1.2
                             else 'no clear shift')
                print(f"    Error ratio    : {r['score_ratio']:.3f}x  ({direction})")
            else:
                print(f"    Error ratio    : n/a")
        else:
            print(f"    Anomaly median : n/a")
        print(f"    AE threshold   : {thr_active:.6f}  "
              f"(rows with MSE above this are flagged)")

    if r['top_features']:
        print(f"  What changed (active rows):")
        for feat, b, a, pct, direction in r['top_features']:
            if pct == float('inf'):
                print(f"    {feat:35s}: 0 → {a:.4f}  (appeared)")
            else:
                arrow = '↑' if direction == 'increased' else '↓'
                print(f"    {feat:35s}: {b:.4f} → {a:.4f}  "
                      f"{arrow} {pct:+.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python autonomous_ae_evaluator.py <ml_ready_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  AUTONOMOUS AUTOENCODER ANOMALY EVALUATOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}\n")

    print(f"[INFO] Loading: {input_file}")
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    df['client_mac'] = df['client_mac'].str.lower().str.strip()
    clients = sorted(df['client_mac'].unique())

    print(f"[INFO] Rows    : {len(df):,}")
    print(f"[INFO] Clients : {len(clients)}")
    print(f"[INFO] Period  : {df['timestamp'].min()} → {df['timestamp'].max()}")

    print(f"\n[INFO] Loading trained AE models...")
    models = load_models()
    thr_active = models['threshold_active']

    print(f"\n[INFO] Evaluating {len(clients)} client(s) autonomously...\n")
    all_results       = []
    all_scored_frames = []

    for mac in clients:
        df_client = df[df['client_mac'] == mac].copy()
        result    = evaluate_client_ae(mac, df_client, models)
        df_scored = result.pop('_df_scored', None)

        if df_scored is not None:
            df_scored['detected_status']        = result['status']
            df_scored['anomaly_start_detected'] = result.get('anomaly_start_ts', '')
            all_scored_frames.append(df_scored)

        all_results.append(result)

    order = {'CRITICAL': 0, 'WARNING': 1, 'WATCH': 2,
             'NORMAL': 3, 'INSUFFICIENT_DATA': 4}
    all_results.sort(
        key=lambda r: (order.get(r['status'], 5), -r['anomaly_rate']))

    print(f"\n{'='*55}")
    print(f"  RESULTS — ALL CLIENTS RANKED BY SEVERITY")
    print(f"{'='*55}")
    for r in all_results:
        print_client_report(r, thr_active)

    critical = [r for r in all_results if r['status'] == 'CRITICAL']
    warnings = [r for r in all_results if r['status'] == 'WARNING']
    watches  = [r for r in all_results if r['status'] == 'WATCH']
    normal   = [r for r in all_results if r['status'] == 'NORMAL']

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  🔴  CRITICAL : {len(critical)} client(s)")
    print(f"  🟡  WARNING  : {len(warnings)} client(s)")
    print(f"  🔵  WATCH    : {len(watches)} client(s)")
    print(f"  ✅  NORMAL   : {len(normal)} client(s)")

    if critical or warnings or watches:
        print(f"\n  ⚠️  ANOMALOUS / SUSPICIOUS CLIENTS:")
        for r in critical + warnings + watches:
            ts    = r['anomaly_start_ts'] or 'not detected'
            ratio = f"{r['score_ratio']:.2f}x" if r['score_ratio'] else 'N/A'
            print(f"     {r['client_mac']}  →  {r['status']}")
            print(f"       started ~{ts}  |  "
                  f"error ratio {ratio}  |  "
                  f"flagged {100*r['anomaly_rate']:.1f}% of rows")
    else:
        print(f"\n  ✅  No anomalous clients detected.")

    base    = os.path.splitext(os.path.basename(input_file))[0]
    run_ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    thr_pct = models['meta'].get('threshold_percentile', 99)
    prefix  = f"ae_auto_{base}_{run_ts}"

    report_rows = []
    for r in all_results:
        report_rows.append({
            'run_timestamp'            : run_ts,
            'input_file'               : os.path.basename(input_file),
            'ae_threshold_pct_used'    : thr_pct,
            'ae_threshold_active'      : thr_active,
            'threshold_warning_used'   : models['threshold_warning'],
            'threshold_critical_used'  : models['threshold_critical'],
            'client_mac'               : r['client_mac'],
            'status'                   : r['status'],
            'total_rows'               : r['total_rows'],
            'active_rows'              : r['active_rows'],
            'flagged_rows'             : r['flagged_rows'],
            'anomaly_rate_active_pct'  : round(100 * r['anomaly_rate'], 2),
            'anomaly_rate_all_rows_pct': round(
                100 * r.get('anomaly_rate_all_rows', r['anomaly_rate']), 2),
            'baseline_rows_used'       : r.get('n_baseline', ''),
            'anomaly_start'            : r['anomaly_start_ts'] or '',
            'change_method'            : r['change_point_method'] or '',
            'confidence'               : r['change_confidence'] or '',
            'mse_baseline'             : r['score_baseline_median'] or '',
            'mse_anomaly_period'       : r['score_anomaly_median'] or '',
            'error_ratio'              : r['score_ratio'] or '',
            'top_changed_features'     : ' | '.join(
                [f"{f}({d})" for f, _, _, _, d in r['top_features']]),
        })

    rpt_path  = f"{RESULTS_DIR}/{prefix}_report.csv"
    full_path = f"{RESULTS_DIR}/{prefix}_scored.csv"
    flag_path = f"{RESULTS_DIR}/{prefix}_flagged.csv"

    pd.DataFrame(report_rows).to_csv(rpt_path, index=False)
    print(f"\n[INFO] Report saved  : {rpt_path}")

    if all_scored_frames:
        full = pd.concat(all_scored_frames, ignore_index=True)
        full['run_timestamp']       = run_ts
        full['ae_threshold_active'] = thr_active
        full['threshold_pct_used']  = thr_pct
        full.to_csv(full_path, index=False)
        print(f"[INFO] Scored data   : {full_path}")

        flagged = full[full['anomaly_flag'] == 1]
        if len(flagged) > 0:
            flagged.to_csv(flag_path, index=False)
            print(f"[INFO] Flagged rows  : {flag_path}  ({len(flagged)} rows)")

    print(f"\n[INFO] Run ID: {run_ts}  (use this to match report/scored/flagged files)")
    print(f"[INFO] Done.\n")


if __name__ == '__main__':
    main()
