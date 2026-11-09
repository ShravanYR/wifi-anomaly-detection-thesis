#!/usr/bin/env python3
"""
autonomous_detector.py
======================
Autonomous WiFi anomaly detector — no manual input required.

Give it a feature-engineered CSV file. It scores every client,
identifies which clients are behaving abnormally, finds when the
anomaly started, and explains what changed.

This replaces the manual experiment_mac input in the evaluator script.

HOW IT WORKS:
    1. Score every row for every client through the trained IF models.
    2. For each client, build a per-client baseline from its first rows.
    3. Apply PELT per client to detect if/when behaviour shifted.
    4. Rank all clients by anomaly intensity.
    5. Flag clients above threshold as WARNING or CRITICAL.
    6. For flagged clients: show WHEN anomaly started, WHAT features changed.

KEY DESIGN DECISION — PER-CLIENT BASELINE:
    Each client is compared against its OWN early behaviour, not against
    other clients. What is normal for client A may be unusual for client B.
    The first BASELINE_ROWS rows of each client define its normal baseline.

USAGE:
    python autonomous_detector.py <ml_ready_csv>

    python autonomous_detector.py ml_ready_data_ex_signalD.csv
    python autonomous_detector.py live_collection.csv

INPUT:
    Any feature-engineered CSV (output of feature_engineering.py).
    Can contain any number of clients — all are evaluated automatically.

OUTPUT:
    Printed report: ranked list of clients with anomaly status
    results/autonomous_<filename>_report.csv   — full per-client results
    results/autonomous_<filename>_flagged.csv  — flagged rows only
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
RESULTS_DIR = "results"

# How many rows to use as each client's personal normal baseline
# 30 rows at 10s interval = 5 minutes of baseline per client
BASELINE_ROWS = 30

# Anomaly rate thresholds (fraction of rows flagged in anomaly period)
THRESHOLD_WARNING  = 0.05   # 5%  of rows flagged = WARNING
THRESHOLD_CRITICAL = 0.20   # 20% of rows flagged = CRITICAL

# Minimum rows a client must have to be evaluated
MIN_ROWS = 20

# PELT penalty — higher = less sensitive to small shifts
PELT_PENALTY = 10

# Features used by each model (must match training)
IDLE_FEATURES = [
    'rssi'
]

ACTIVE_FEATURES = [
    'tx_packet_rate', 'rx_packet_rate',
    'tx_byte_rate',   'rx_byte_rate',
    'tx_success_rate', 'retry_rate', 'tx_failed_rate',
    'rssi', 'tx_throughput_mbps', 'rx_throughput_mbps',
    'low_mcs_ratio', 'mid_mcs_ratio', 'high_mcs_ratio',
    'mcs_weighted_score', 'mcs_entropy',
]

# Features to show in the "what changed" explanation
EXPLAIN_FEATURES = [
    'retry_rate', 'tx_failed_rate', 'tx_success_rate',
    'tx_packet_rate', 'rx_packet_rate',
    'rssi', 'mcs_entropy', 'low_mcs_ratio',
    'tx_throughput_mbps', 'rx_throughput_mbps',
]


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    def load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run isolation_forest_training.py first."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {
        'idle_model'    : load(f"{MODELS_DIR}/if_idle_model.pkl"),
        'active_model'  : load(f"{MODELS_DIR}/if_active_model.pkl"),
        'scaler_idle'   : load(f"{MODELS_DIR}/scaler_idle.pkl"),
        'scaler_active' : load(f"{MODELS_DIR}/scaler_active.pkl"),
    }


# ── Score all rows ────────────────────────────────────────────────────────────

def score_all_rows(df, models):
    """
    Score every row in the dataframe using the correct model per row.
    Returns df with anomaly_score and anomaly_flag columns added.
    """
    df = df.copy()
    df['anomaly_score'] = np.nan
    df['anomaly_flag']  = 0
    df['model_used']    = 'none'

    idle_mask   = df['is_active'] == 0
    active_mask = df['is_active'] == 1

    avail_idle   = [f for f in IDLE_FEATURES   if f in df.columns]
    avail_active = [f for f in ACTIVE_FEATURES if f in df.columns]

    if idle_mask.sum() > 0 and len(avail_idle) == len(IDLE_FEATURES):
        X = models['scaler_idle'].transform(df.loc[idle_mask, avail_idle].values)
        df.loc[idle_mask, 'anomaly_score'] = models['idle_model'].score_samples(X)
        df.loc[idle_mask, 'anomaly_flag']  = (
            models['idle_model'].predict(X) == -1).astype(int)
        df.loc[idle_mask, 'model_used'] = 'idle'

    if active_mask.sum() > 0 and len(avail_active) > 0:
        X = models['scaler_active'].transform(
            df.loc[active_mask, avail_active].values)
        df.loc[active_mask, 'anomaly_score'] = models['active_model'].score_samples(X)
        df.loc[active_mask, 'anomaly_flag']  = (
            models['active_model'].predict(X) == -1).astype(int)
        df.loc[active_mask, 'model_used'] = 'active'

    return df


# ── Build composite stress signal per client ──────────────────────────────────

def build_stress_signal(df_client):
    """
    Build a single per-row signal that rises when the network is stressed.
    Used by PELT to find the change point automatically.
    """
    signal = pd.Series(0.0, index=df_client.index)

    for col, weight in [
        ('retry_rate',      1.0),
        ('tx_failed_rate',  1.0),
        ('tx_success_rate', -1.0),   # inverted: lower success = higher stress
        ('tx_throughput_mbps', -0.5) # inverted: lower throughput = higher stress
    ]:
        if col in df_client.columns:
            vals = df_client[col].fillna(0).astype(float)
            max_val = vals.abs().max()
            if max_val > 0:
                if weight > 0:
                    signal += weight * (vals / max_val)
                else:
                    # Invert: normalise then flip
                    signal += abs(weight) * (1 - (vals / max_val))

    # Smooth with rolling average to reduce noise
    signal = signal.rolling(window=3, min_periods=1, center=True).mean()
    return signal.values.astype(float)


# ── Automatic change point detection per client ───────────────────────────────

def find_change_point(signal, min_size=5):
    """
    Find where signal distribution shifts using PELT (primary)
    or sliding window (fallback).
    Returns (split_index, method, confidence).
    """
    if len(signal) < min_size * 2:
        return None, 'insufficient_data', 'none'

    # Try PELT
    try:
        import ruptures as rpt
        algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        algo.fit(signal.reshape(-1, 1))
        result = algo.predict(pen=PELT_PENALTY)
        breakpoints = [b for b in result if b < len(signal)]
        if breakpoints:
            split = breakpoints[0]
            left_mean  = signal[:split].mean()
            right_mean = signal[split:].mean()
            shift = abs(right_mean - left_mean)
            confidence = 'high' if shift > 0.3 else ('medium' if shift > 0.1 else 'low')
            return split, 'PELT', confidence
    except ImportError:
        pass

    # Sliding window fallback
    best_split, best_diff = None, -1
    for i in range(min_size, len(signal) - min_size):
        diff = abs(signal[i:].mean() - signal[:i].mean())
        if diff > best_diff:
            best_diff = diff
            best_split = i

    if best_split is None:
        return None, 'no_change', 'none'

    shift = best_diff
    confidence = 'high' if shift > 0.3 else ('medium' if shift > 0.1 else 'low')
    return best_split, 'sliding_window', confidence


# ── Explain what changed for a flagged client ─────────────────────────────────

def explain_anomaly(df_baseline, df_anomaly_period):
    """
    Compare feature means between baseline and anomaly period.
    Returns list of (feature, baseline_mean, anomaly_mean, change_pct, direction)
    sorted by magnitude of change, top 5 only.
    """
    explanations = []
    avail = [f for f in EXPLAIN_FEATURES
             if f in df_baseline.columns and f in df_anomaly_period.columns]

    # Use active rows only for comparison — idle rows have no MCS signal
    base_active = df_baseline[df_baseline['is_active'] == 1]
    anom_active = df_anomaly_period[df_anomaly_period['is_active'] == 1]

    if len(base_active) == 0 or len(anom_active) == 0:
        # Fall back to all rows if no active rows available
        base_active = df_baseline
        anom_active = df_anomaly_period

    for feat in avail:
        b_mean = base_active[feat].mean()
        a_mean = anom_active[feat].mean()
        if b_mean == 0:
            if a_mean != 0:
                explanations.append((feat, b_mean, a_mean, float('inf'),
                                     'appeared'))
        else:
            pct = 100 * (a_mean - b_mean) / abs(b_mean)
            if abs(pct) > 5:   # only report meaningful changes
                direction = 'increased' if pct > 0 else 'decreased'
                explanations.append((feat, b_mean, a_mean, pct, direction))

    # Sort by absolute magnitude of change
    explanations.sort(key=lambda x: abs(x[3]) if x[3] != float('inf') else 9999,
                      reverse=True)
    return explanations[:5]   # top 5 most changed features


# ── Evaluate one client autonomously ─────────────────────────────────────────

def evaluate_client(mac, df_client, models):
    """
    Fully autonomous evaluation of a single client.
    Returns a result dict with all findings.
    """
    df_client = df_client.sort_values('timestamp').reset_index(drop=True)

    result = {
        'client_mac'        : mac,
        'total_rows'        : len(df_client),
        'active_rows'       : int(df_client['is_active'].sum()),
        'status'            : 'NORMAL',
        'anomaly_rate'      : 0.0,
        'anomaly_start_ts'  : None,
        'change_point_idx'  : None,
        'change_method'     : None,
        'change_confidence' : None,
        'score_baseline'    : None,
        'score_anomaly'     : None,
        'score_shift'       : None,
        'top_features'      : [],
        'flagged_rows'      : 0,
    }

    if len(df_client) < MIN_ROWS:
        result['status'] = 'INSUFFICIENT_DATA'
        return result

    # Score all rows
    df_scored = score_all_rows(df_client, models)

    total_flagged    = int(df_scored['anomaly_flag'].sum())
    overall_anom_rate = total_flagged / len(df_scored)
    result['flagged_rows']  = total_flagged
    result['anomaly_rate']  = round(overall_anom_rate, 4)

    # ── Build stress signal and find change point ──────────────────────────
    # Use active rows for the signal if enough exist, else all rows
    active_rows = df_scored[df_scored['is_active'] == 1]
    signal_source = active_rows if len(active_rows) >= MIN_ROWS else df_scored
    signal = build_stress_signal(signal_source)

    split_in_source, method, confidence = find_change_point(signal)
    result['change_method']     = method
    result['change_confidence'] = confidence

    if split_in_source is not None and method not in ('no_change', 'insufficient_data'):
        # Map split back to full client dataframe
        source_indices = signal_source.index.tolist()
        global_split   = source_indices[min(split_in_source,
                                            len(source_indices) - 1)]

        result['change_point_idx'] = int(global_split)
        result['anomaly_start_ts'] = str(df_scored.loc[global_split, 'timestamp'])

        # Split into baseline (before change point) and anomaly period (after)
        df_before = df_scored[df_scored.index < global_split]
        df_after  = df_scored[df_scored.index >= global_split]

        # Use first BASELINE_ROWS rows as the personal baseline
        df_baseline = df_scored.iloc[:BASELINE_ROWS]

        # Score statistics for both periods
        if len(df_before) > 0:
            result['score_baseline'] = round(
                float(df_before['anomaly_score'].mean()), 4)
        if len(df_after) > 0:
            result['score_anomaly']  = round(
                float(df_after['anomaly_score'].mean()), 4)
        if result['score_baseline'] and result['score_anomaly']:
            result['score_shift'] = round(
                result['score_anomaly'] - result['score_baseline'], 4)

        # Anomaly rate in the detected anomaly period only
        if len(df_after) > 0:
            result['anomaly_rate'] = round(
                df_after['anomaly_flag'].sum() / len(df_after), 4)

        # Explain what changed
        result['top_features'] = explain_anomaly(df_baseline, df_after)
    else:
        # No change point found — use full history as baseline
        df_baseline = df_scored.iloc[:BASELINE_ROWS]
        result['score_baseline'] = round(
            float(df_baseline['anomaly_score'].mean()), 4)
        result['score_anomaly']  = round(
            float(df_scored.iloc[BASELINE_ROWS:]['anomaly_score'].mean()), 4) \
            if len(df_scored) > BASELINE_ROWS else result['score_baseline']
        result['score_shift'] = round(
            result['score_anomaly'] - result['score_baseline'], 4) \
            if result['score_baseline'] and result['score_anomaly'] else 0.0

    # ── Assign status ──────────────────────────────────────────────────────
    anom_rate = result['anomaly_rate']
    score_shift = result['score_shift'] or 0

    if anom_rate >= THRESHOLD_CRITICAL:
        result['status'] = 'CRITICAL'
    elif anom_rate >= THRESHOLD_WARNING:
        result['status'] = 'WARNING'
    elif score_shift > 0.05 and confidence in ('medium', 'high'):
        # Score drifted meaningfully even without crossing flag threshold
        result['status'] = 'WATCH'
    else:
        result['status'] = 'NORMAL'

    # Attach scored dataframe for saving
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

def print_client_report(r):
    status_str = STATUS_ICONS.get(r['status'], r['status'])
    print(f"\n  {'─'*52}")
    print(f"  Client : {r['client_mac']}")
    print(f"  Status : {status_str}")
    print(f"  Rows   : {r['total_rows']} total  |  "
          f"{r['active_rows']} active "
          f"({100*r['active_rows']/r['total_rows']:.1f}%)")
    print(f"  Flagged: {r['flagged_rows']} rows  |  "
          f"Anomaly rate: {100*r['anomaly_rate']:.1f}%")

    if r['anomaly_start_ts']:
        print(f"  Anomaly detected at : {r['anomaly_start_ts']}")
        print(f"  Detection method    : {r['change_method']}  "
              f"(confidence: {r['change_confidence']})")

    if r['score_baseline'] is not None:
        print(f"  Score — baseline: {r['score_baseline']:.4f}  |  "
              f"anomaly period: {r['score_anomaly']:.4f}  |  "
              f"shift: {r['score_shift']:+.4f}")

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
        print("Usage: python autonomous_detector.py <ml_ready_csv>")
        print("Example: python autonomous_detector.py ml_ready_data_ex_signalD.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  AUTONOMOUS WiFi ANOMALY DETECTOR")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")
    print(f"\n[INFO] Loading: {input_file}")

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    df['client_mac'] = df['client_mac'].str.lower().str.strip()
    clients = sorted(df['client_mac'].unique())

    print(f"[INFO] Rows    : {len(df):,}")
    print(f"[INFO] Clients : {len(clients)}")
    print(f"[INFO] Period  : {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── Load models ───────────────────────────────────────────────────────
    print(f"\n[INFO] Loading trained models...")
    models = load_models()
    print(f"[INFO] Models loaded.")

    # ── Evaluate every client ──────────────────────────────────────────────
    print(f"\n[INFO] Evaluating {len(clients)} client(s) autonomously...")
    all_results = []
    all_scored_frames = []

    for mac in clients:
        df_client = df[df['client_mac'] == mac].copy()
        result = evaluate_client(mac, df_client, models)
        df_scored = result.pop('_df_scored', None)

        if df_scored is not None:
            df_scored['detected_status'] = result['status']
            df_scored['anomaly_start_detected'] = result.get('anomaly_start_ts', '')
            all_scored_frames.append(df_scored)

        all_results.append(result)

    # ── Sort by severity ───────────────────────────────────────────────────
    severity_order = {'CRITICAL': 0, 'WARNING': 1, 'WATCH': 2,
                      'NORMAL': 3, 'INSUFFICIENT_DATA': 4}
    all_results.sort(key=lambda r: (severity_order.get(r['status'], 5),
                                    -r['anomaly_rate']))

    # ── Print results ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RESULTS — ALL CLIENTS RANKED BY SEVERITY")
    print(f"{'='*55}")

    for r in all_results:
        print_client_report(r)

    # ── Summary ────────────────────────────────────────────────────────────
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

    if critical or warnings:
        print(f"\n  ⚠️  ANOMALOUS CLIENTS DETECTED:")
        for r in critical + warnings:
            ts = r['anomaly_start_ts'] or 'unknown'
            print(f"     {r['client_mac']}  →  {r['status']}  "
                  f"(started ~{ts})")

    # ── Save outputs ───────────────────────────────────────────────────────
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Summary report CSV
    report_rows = []
    for r in all_results:
        report_rows.append({
            'client_mac'        : r['client_mac'],
            'status'            : r['status'],
            'total_rows'        : r['total_rows'],
            'active_rows'       : r['active_rows'],
            'flagged_rows'      : r['flagged_rows'],
            'anomaly_rate_pct'  : round(100 * r['anomaly_rate'], 2),
            'anomaly_start'     : r['anomaly_start_ts'] or '',
            'change_method'     : r['change_method'] or '',
            'confidence'        : r['change_confidence'] or '',
            'score_baseline'    : r['score_baseline'] or '',
            'score_anomaly'     : r['score_anomaly'] or '',
            'score_shift'       : r['score_shift'] or '',
            'top_changed_features': ' | '.join(
                [f"{f}({d})" for f, _, _, _, d in r['top_features']]
            ),
        })

    report_path = f"{RESULTS_DIR}/autonomous_{base_name}_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    print(f"\n[INFO] Report saved : {report_path}")

    # Full scored data
    if all_scored_frames:
        full_scored = pd.concat(all_scored_frames, ignore_index=True)
        full_path = f"{RESULTS_DIR}/autonomous_{base_name}_scored.csv"
        full_scored.to_csv(full_path, index=False)
        print(f"[INFO] Scored data  : {full_path}")

        # Flagged rows only
        flagged = full_scored[full_scored['anomaly_flag'] == 1]
        if len(flagged) > 0:
            flagged_path = f"{RESULTS_DIR}/autonomous_{base_name}_flagged.csv"
            flagged.to_csv(flagged_path, index=False)
            print(f"[INFO] Flagged rows : {flagged_path}  ({len(flagged)} rows)")

    print(f"\n[INFO] Done.\n")


if __name__ == '__main__':
    main()
