#!/usr/bin/env python3
"""
real_experiment_evaluator.py  —  v2  (automatic boundary detection)
====================================================================
Evaluates trained Isolation Forest models against real experiment data.
No manual timestamp required — the split point between normal and anomaly
periods is detected automatically from the data.

HOW BOUNDARY DETECTION WORKS:
    A composite stress signal is built per row from key features:
        stress = retry_rate + tx_failed_rate + (1 - tx_success_rate)
                 + normalised_throughput_drop

    Then one of two methods finds where this signal shifts permanently:

    PRIMARY — PELT change point detection (ruptures library)
        Statistically rigorous algorithm that finds the exact row
        where the signal distribution changes. Used when ruptures
        is installed and enough rows are available.
        Install: pip install ruptures

    FALLBACK — Sliding window comparison (no extra library needed)
        Divides the timeline into overlapping windows and finds the
        window boundary where the signal mean shifts most sharply.
        Used automatically if ruptures is not installed.

    Both methods produce the same output — a split timestamp —
    with no manual input needed.

USAGE:
    pip install ruptures          (optional but recommended)
    python real_experiment_evaluator.py

INPUT:
    Edit the EXPERIMENTS list below with your experiment CSV files.
    No anomaly_start time needed — just the file and device MAC.

OUTPUT:
    results/real_<name>_evaluation.csv     scored rows with period label
    results/real_experiment_summary.txt    combined report for thesis
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
RESULTS_DIR = "results"

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

COMPARISON_FEATURES = [
    'retry_rate', 'tx_failed_rate', 'tx_success_rate',
    'tx_packet_rate', 'rx_packet_rate',
    'rssi', 'mcs_entropy', 'low_mcs_ratio', 'high_mcs_ratio',
    'tx_throughput_mbps', 'rx_throughput_mbps',
]


# ── EXPERIMENTS — edit this list, no timestamp needed ─────────────────────────

EXPERIMENTS = [
    {
        'name'          : 'signal_degradation',
        'csv_file'      : 'ml_ready_data_ex_signalD.csv',
        'experiment_mac': 'a8:74:1d:47:e6:3e',
        'description'   : 'WiFi signal degradation experiment',
    },
    # Add the remaining experiments as you collect them:
    # {
    #     'name'          : 'signal_degradation',
    #     'csv_file'      : 'data/exp_signal_degradation_ml_ready.csv',
    #     'experiment_mac': 'a8:74:1d:47:e6:8e',
    #     'description'   : 'device moved 15m from AP, blocked by wall',
    # },
    # {
    #     'name'          : 'link_failure',
    #     'csv_file'      : 'data/exp_link_failure_ml_ready.csv',
    #     'experiment_mac': 'a8:74:1d:47:e6:8e',
    #     'description'   : 'WiFi toggled off/on 8 times over 10 minutes',
    # },
    # {
    #     'name'          : 'traffic_burst',
    #     'csv_file'      : 'data/exp_traffic_burst_ml_ready.csv',
    #     'experiment_mac': 'a8:74:1d:47:e6:8e',
    #     'description'   : 'large file download sustained for 10 minutes',
    # },
]


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    def load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
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


# ── Score rows ────────────────────────────────────────────────────────────────

def score_rows(df, models):
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


# ── Composite stress signal ───────────────────────────────────────────────────

def build_stress_signal(df):
    """
    Build a single composite signal per row that rises when the
    network is under stress. Higher value = more stressed.

    Components:
        retry_rate          — direct stress indicator
        tx_failed_rate      — direct stress indicator
        (1-tx_success_rate) — inverted success = failure signal
        throughput_drop     — normalised drop in combined throughput

    All components normalised to [0,1] range before combining
    so no single feature dominates.
    """
    signal = pd.Series(0.0, index=df.index)

    # Component 1 — retry rate (already 0-1 range)
    if 'retry_rate' in df.columns:
        r = df['retry_rate'].fillna(0)
        max_r = r.max()
        if max_r > 0:
            signal += r / max_r

    # Component 2 — tx_failed_rate (already 0-1 range)
    if 'tx_failed_rate' in df.columns:
        f = df['tx_failed_rate'].fillna(0)
        max_f = f.max()
        if max_f > 0:
            signal += f / max_f

    # Component 3 — inverted tx_success_rate
    # (0 = perfect, 1 = complete failure)
    if 'tx_success_rate' in df.columns:
        fail = (1 - df['tx_success_rate'].fillna(1)).clip(0, 1)
        max_fail = fail.max()
        if max_fail > 0:
            signal += fail / max_fail

    # Component 4 — throughput drop (inverted, higher = more drop)
    if 'tx_throughput_mbps' in df.columns:
        tput = df['tx_throughput_mbps'].fillna(0)
        # Invert: high throughput = low stress, low throughput = high stress
        tput_max = tput.max()
        if tput_max > 0:
            tput_drop = 1 - (tput / tput_max)
            signal += tput_drop

    # Smooth with a small rolling window (3 rows) to reduce noise
    signal = signal.rolling(window=3, min_periods=1, center=True).mean()

    return signal.values.astype(float)


# ── Automatic boundary detection ──────────────────────────────────────────────

def detect_boundary_pelt(signal, min_normal_rows=5):
    """
    Use PELT (Pruned Exact Linear Time) change point detection.
    Finds where the signal distribution changes most significantly.
    Returns the index of the detected change point.
    """
    import ruptures as rpt

    if len(signal) < min_normal_rows * 2:
        return None

    # Reshape to 2D as required by ruptures
    sig_2d = signal.reshape(-1, 1)

    # PELT with RBF cost — best for detecting mean shifts
    # pen=10 is conservative (avoids detecting noise fluctuations as changes)
    # Lower pen = more sensitive, higher pen = less sensitive
    algo = rpt.Pelt(model="rbf", min_size=max(3, min_normal_rows), jump=1)
    algo.fit(sig_2d)
    result = algo.predict(pen=10)

    # result is list of breakpoint indices, last entry is always len(signal)
    # Remove the last entry, take the first real change point
    breakpoints = [b for b in result if b < len(signal)]

    if not breakpoints:
        return None

    # If multiple change points, take the first one
    # (first significant shift = where anomaly started)
    return breakpoints[0]


def detect_boundary_sliding_window(signal, min_normal_rows=5):
    """
    Fallback: sliding window comparison.
    Splits the signal at every possible position and finds the split
    where the difference in mean between left and right is largest.
    Returns the index of the best split point.
    """
    n = len(signal)
    best_split = None
    best_diff  = -1

    # Ensure we have at least min_normal_rows on each side
    for split in range(min_normal_rows, n - min_normal_rows):
        left_mean  = signal[:split].mean()
        right_mean = signal[split:].mean()
        diff = abs(right_mean - left_mean)
        if diff > best_diff:
            best_diff  = diff
            best_split = split

    return best_split


def detect_boundary(df_device, method_log):
    """
    Automatically detect the normal/anomaly boundary.
    Returns (split_index, split_timestamp, method_used, confidence).

    Uses active rows only for signal — idle rows carry no stress signal.
    Then maps back to the full dataframe index.
    """
    df_device = df_device.sort_values('timestamp').reset_index(drop=True)

    # Use active rows only for the signal — idle rows carry no stress signal
    active_idx = df_device[df_device['is_active'] == 1].index.tolist()

    if len(active_idx) < 10:
        # Not enough active rows — fall back to first 30% as normal
        split_idx = max(1, int(len(df_device) * 0.30))
        method_log.append(
            f"  [BOUNDARY] Too few active rows ({len(active_idx)}) for signal-based detection.")
        method_log.append(
            f"  [BOUNDARY] Using first 30% of rows as normal period.")
        split_ts = df_device.loc[split_idx, 'timestamp']
        return split_idx, split_ts, 'first_30pct', 'low'

    # Build stress signal on active rows
    df_active = df_device.loc[active_idx].copy().reset_index(drop=True)
    signal = build_stress_signal(df_active)

    split_in_active = None
    method_used     = None

    # Try PELT first
    try:
        import ruptures
        split_in_active = detect_boundary_pelt(signal, min_normal_rows=5)
        method_used = 'PELT'
        method_log.append(f"  [BOUNDARY] Method: PELT change point detection (ruptures)")
    except ImportError:
        method_log.append(
            f"  [BOUNDARY] ruptures not installed — using sliding window fallback")
        method_log.append(
            f"             Install for better accuracy: pip install ruptures")

    # Fallback to sliding window
    if split_in_active is None:
        split_in_active = detect_boundary_sliding_window(signal, min_normal_rows=5)
        method_used = 'sliding_window'
        method_log.append(f"  [BOUNDARY] Method: sliding window comparison")

    if split_in_active is None:
        split_idx = int(len(df_device) * 0.5)
        method_log.append(f"  [BOUNDARY] Detection failed — using 50% split")
        split_ts = df_device.loc[split_idx, 'timestamp']
        return split_idx, split_ts, 'fallback_50pct', 'low'

    # Map split index in active rows back to full device dataframe
    global_split_idx = active_idx[min(split_in_active, len(active_idx) - 1)]
    split_ts         = df_device.loc[global_split_idx, 'timestamp']

    # Confidence: how large is the signal shift across the boundary?
    left_mean  = signal[:split_in_active].mean()
    right_mean = signal[split_in_active:].mean()
    shift      = abs(right_mean - left_mean)

    if shift > 0.3:
        confidence = 'high'
    elif shift > 0.1:
        confidence = 'medium'
    else:
        confidence = 'low'

    method_log.append(
        f"  [BOUNDARY] Split at row {global_split_idx} of {len(df_device)} "
        f"| signal shift: {shift:.4f} | confidence: {confidence}")
    method_log.append(
        f"  [BOUNDARY] Detected split timestamp: {split_ts}")

    return global_split_idx, split_ts, method_used, confidence


# ── Feature comparison table ──────────────────────────────────────────────────

def feature_comparison(df_normal_active, df_anomaly_active, name):
    lines = []
    lines.append(f"\n  Feature comparison — {name}")
    lines.append(
        f"  {'Feature':35s}  {'Normal avg':>12}  {'Anomaly avg':>12}  {'Change':>10}")
    lines.append(f"  {'-'*35}  {'-'*12}  {'-'*12}  {'-'*10}")

    available = [f for f in COMPARISON_FEATURES
                 if f in df_normal_active.columns and f in df_anomaly_active.columns]

    for feat in available:
        n_mean = df_normal_active[feat].mean()
        a_mean = df_anomaly_active[feat].mean()
        if n_mean == 0:
            change_str = 'N/A'
        else:
            pct = 100 * (a_mean - n_mean) / abs(n_mean)
            arrow = '↑' if pct > 5 else ('↓' if pct < -5 else '~')
            change_str = f"{arrow} {pct:+.0f}%"
        lines.append(
            f"  {feat:35s}  {n_mean:>12.4f}  {a_mean:>12.4f}  {change_str:>10}")

    return '\n'.join(lines)


# ── Evaluate one experiment ───────────────────────────────────────────────────

def evaluate_experiment(exp, models, all_report_lines):
    name        = exp['name']
    csv_file    = exp['csv_file']
    mac         = exp['experiment_mac'].lower()
    description = exp['description']

    def log(msg=''):
        print(msg)
        all_report_lines.append(msg)

    log(f"\n{'='*55}")
    log(f"  EXPERIMENT: {name.upper()}")
    log(f"  {description}")
    log(f"  Device : {mac}")
    log(f"{'='*55}")

    if not os.path.exists(csv_file):
        log(f"  [SKIP] File not found: {csv_file}")
        log(f"         Run feature_engineering.py on your experiment data first.")
        return

    # ── Load and filter to experiment device ──────────────────────────────────
    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    df['client_mac'] = df['client_mac'].str.lower().str.strip()
    df_device = df[df['client_mac'] == mac].sort_values(
        'timestamp').reset_index(drop=True)

    if len(df_device) == 0:
        log(f"  [ERROR] MAC {mac} not found. Available: {list(df['client_mac'].unique())}")
        return

    log(f"  Total rows loaded  : {len(df):,}  ({df['client_mac'].nunique()} clients)")
    log(f"  Rows for device    : {len(df_device):,}")
    log(f"  Time range         : {df_device['timestamp'].min()} -> "
        f"{df_device['timestamp'].max()}")
    log(f"  Active rows        : {df_device['is_active'].sum()} "
        f"({100*df_device['is_active'].mean():.1f}%)")

    # ── Automatic boundary detection ──────────────────────────────────────────
    log(f"\n  AUTOMATIC BOUNDARY DETECTION:")
    boundary_log = []
    split_idx, split_ts, method, confidence = detect_boundary(
        df_device, boundary_log)

    for line in boundary_log:
        log(line)

    log(f"  Detected split     : {split_ts}")
    log(f"  Method             : {method}  |  Confidence: {confidence}")

    if confidence == 'low':
        log(f"  [WARN] Low confidence split — the stress condition may not have")
        log(f"         changed features enough for automatic detection.")
        log(f"         Consider: longer experiment, stronger stress, or more data.")

    # ── Split into normal and anomaly periods ─────────────────────────────────
    df_normal  = df_device[df_device['timestamp'] <  split_ts].copy()
    df_anomaly = df_device[df_device['timestamp'] >= split_ts].copy()

    log(f"\n  Period split:")
    log(f"    Normal period  : {len(df_normal):>4} rows  "
        f"({df_normal['timestamp'].min() if len(df_normal) > 0 else 'N/A'} -> {split_ts})")
    log(f"    Anomaly period : {len(df_anomaly):>4} rows  "
        f"({split_ts} -> {df_device['timestamp'].max()})")

    if len(df_normal) < 5:
        log(f"  [WARN] Very few normal period rows — collect at least 10 min of")
        log(f"         normal data before starting the stress condition.")

    # ── Score both periods ────────────────────────────────────────────────────
    df_normal_scored  = score_rows(df_normal,  models)
    df_anomaly_scored = score_rows(df_anomaly, models)

    # ── Normal period — false alarm check ─────────────────────────────────────
    log(f"\n  NORMAL PERIOD (false alarm check):")
    if len(df_normal_scored) > 0:
        n_total   = len(df_normal_scored)
        n_flagged = int(df_normal_scored['anomaly_flag'].sum())
        fa_rate   = 100 * n_flagged / n_total
        log(f"    Rows         : {n_total}")
        log(f"    Active rows  : {int(df_normal_scored['is_active'].sum())}")
        log(f"    Flagged      : {n_flagged} ({fa_rate:.1f}%)  "
            f"[training baseline: 1.5%]")
    else:
        log("    No normal period rows to evaluate.")
        fa_rate = 0

    # ── Anomaly period — detection check ──────────────────────────────────────
    log(f"\n  ANOMALY PERIOD (detection check):")
    a_total   = len(df_anomaly_scored)
    a_flagged = int(df_anomaly_scored['anomaly_flag'].sum())
    a_active  = df_anomaly_scored[df_anomaly_scored['is_active'] == 1]
    a_active_flagged = int(a_active['anomaly_flag'].sum())
    det_rate_all    = 100 * a_flagged / a_total if a_total > 0 else 0
    det_rate_active = 100 * a_active_flagged / len(a_active) if len(a_active) > 0 else 0

    log(f"    Rows              : {a_total}")
    log(f"    Active rows       : {len(a_active)}")
    log(f"    Flagged (all)     : {a_flagged} ({det_rate_all:.1f}%)")
    log(f"    Flagged (active)  : {a_active_flagged}/{len(a_active)} "
        f"({det_rate_active:.1f}%)")
    log(f"")
    log(f"    ** DETECTION RATE: {det_rate_active:.1f}% (on active rows) **")

    # ── Anomaly score distribution ────────────────────────────────────────────
    log(f"\n  ANOMALY SCORE DISTRIBUTION:")
    log(f"    Normal period  — mean: {df_normal_scored['anomaly_score'].mean():.4f}  "
        f"max: {df_normal_scored['anomaly_score'].max():.4f}")
    log(f"    Anomaly period — mean: {df_anomaly_scored['anomaly_score'].mean():.4f}  "
        f"max: {df_anomaly_scored['anomaly_score'].max():.4f}")
    log(f"    (Less negative = more anomalous)")

    # Score shift — key diagnostic
    n_mean = df_normal_scored['anomaly_score'].mean()
    a_mean = df_anomaly_scored['anomaly_score'].mean()
    shift  = a_mean - n_mean
    log(f"    Score shift     : {shift:+.4f}  "
        f"({'model sensed change' if shift > 0 else 'no signal shift'})")

    # ── Feature comparison ────────────────────────────────────────────────────
    normal_active  = df_normal_scored[df_normal_scored['is_active'] == 1]
    anomaly_active = df_anomaly_scored[df_anomaly_scored['is_active'] == 1]

    if len(normal_active) > 0 and len(anomaly_active) > 0:
        log(feature_comparison(normal_active, anomaly_active, name))
    else:
        log(f"\n  [SKIP] Feature comparison needs active rows in both periods.")
        log(f"         Normal active: {len(normal_active)}, "
            f"Anomaly active: {len(anomaly_active)}")

    # ── Thesis result line ────────────────────────────────────────────────────
    log(f"\n  THESIS RESULT LINE:")
    log(f"    {name}: IF detection = {det_rate_active:.1f}% | "
        f"false alarm = {fa_rate:.1f}% | "
        f"boundary method = {method} | confidence = {confidence}")

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_out = pd.concat([df_normal_scored, df_anomaly_scored], ignore_index=True)
    df_out['period'] = 'normal'
    df_out.loc[df_out['timestamp'] >= split_ts, 'period'] = 'anomaly'
    df_out['detected_boundary'] = str(split_ts)
    df_out['boundary_method']   = method
    df_out['boundary_confidence'] = confidence

    out_path = f"{RESULTS_DIR}/real_{name}_evaluation.csv"
    df_out.to_csv(out_path, index=False)
    log(f"\n  Saved: {out_path}")
    log(f"  (period column: normal/anomaly — use for thesis plots)")

    return det_rate_active, fa_rate


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[INFO] Loading trained models...")
    models = load_models()
    print("[INFO] Models loaded.")

    all_report_lines = [
        "REAL EXPERIMENT EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "(Automatic boundary detection — no manual timestamps needed)",
        "=" * 55,
    ]

    if not EXPERIMENTS:
        print("[WARN] No experiments configured in EXPERIMENTS list.")
        return

    summary_rows = []

    for exp in EXPERIMENTS:
        result = evaluate_experiment(exp, models, all_report_lines)
        if result:
            det_rate, fa_rate = result
            summary_rows.append({
                'experiment': exp['name'],
                'detection_rate': det_rate,
                'false_alarm_rate': fa_rate,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    all_report_lines.append(f"\n{'='*55}")
    all_report_lines.append("  SUMMARY TABLE FOR THESIS")
    all_report_lines.append(f"  {'Experiment':28s}  "
                             f"{'IF Detection':>14}  {'False Alarm':>12}")
    all_report_lines.append(f"  {'-'*28}  {'-'*14}  {'-'*12}")

    for row in summary_rows:
        all_report_lines.append(
            f"  {row['experiment']:28s}  "
            f"{row['detection_rate']:>13.1f}%  "
            f"{row['false_alarm_rate']:>11.1f}%")

    all_report_lines.append(f"\n  Synthetic testing reference:")
    all_report_lines.append(f"  {'retry_storm (synthetic)':28s}  {'16.0%':>14}  {'1.5%':>12}")
    all_report_lines.append(f"  {'signal_degradation (synth)':28s}  {'12.0%':>14}  {'1.5%':>12}")
    all_report_lines.append(f"  {'link_failure (synthetic)':28s}  {'100.0%':>14}  {'1.5%':>12}")
    all_report_lines.append(f"  {'traffic_burst (synthetic)':28s}  {'100.0%':>14}  {'1.5%':>12}")

    report_path = f"{RESULTS_DIR}/real_experiment_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_report_lines))

    print(f"\n[INFO] Report saved: {report_path}")
    print(f"[INFO] Done.")


if __name__ == '__main__':
    main()
