#!/usr/bin/env python3
"""
real_experiment_evaluator.py
============================
Evaluates trained Isolation Forest models against real-world experiment data.
One script handles all four experiment types.

HOW IT WORKS:
    Each experiment file has TWO periods:
        Period 1 — normal behaviour    (first ~10 minutes)
        Period 2 — anomaly behaviour   (last ~10 minutes, e.g. hrping stress)

    The script splits by a timestamp you provide, scores both periods
    separately, and reports:
        - False alarm rate on normal period  (should be ~1.5%, same as training)
        - Detection rate on anomaly period   (this is your result)
        - Feature comparison table           (normal avg vs anomaly avg)

USAGE:
    Edit the EXPERIMENTS list at the bottom of this file, then run:
        python real_experiment_evaluator.py

INPUT FILES NEEDED:
    - ml_ready_data.csv                    (your normal training data)
    - models/if_idle_model.pkl
    - models/if_active_model.pkl
    - models/scaler_idle.pkl
    - models/scaler_active.pkl
    - one feature-engineered CSV per experiment
      (parse raw txt -> feature_engineering.py -> gives you this CSV)

OUTPUT:
    results/real_<experiment_name>_evaluation.csv   scored rows
    results/real_experiment_summary.txt             combined report

IMPORTANT — HOW TO NOTE YOUR TIMESTAMPS:
    When running the collection, note the exact time you started
    the stress condition (e.g. started hrping). That is your
    ANOMALY_START_TIME. Everything before it = normal period.
    Everything from it onwards = anomaly period.

    Example: collection started 14:00, you started hrping at 14:10
        ANOMALY_START_TIME = '2025-05-10 14:10:00'
"""

import os
import json
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

# Features to show in the comparison table
# These are the ones most relevant to each anomaly type
COMPARISON_FEATURES = [
    'retry_rate',
    'tx_failed_rate',
    'tx_success_rate',
    'tx_packet_rate',
    'rx_packet_rate',
    'rssi',
    'mcs_entropy',
    'low_mcs_ratio',
    'high_mcs_ratio',
    'tx_throughput_mbps',
    'rx_throughput_mbps',
]

# ── Experiment definitions ────────────────────────────────────────────────────
# Edit these before running. Add one entry per experiment you have collected.
#
# Fields:
#   name              : short label used in output files and reports
#   csv_file          : path to your feature-engineered experiment CSV
#   experiment_mac    : MAC address of the device you stressed
#   anomaly_start     : timestamp when you started the stress condition
#                       format: 'YYYY-MM-DD HH:MM:SS'
#   description       : one line describing what you did (goes into report)

EXPERIMENTS = [
    {
        'name'           : 'retry_storm',
        'csv_file'       : 'ml_ready_data_ex_RStorm.csv',
        'experiment_mac' : 'a8:74:1d:47:e6:3e',   # <- replace with your device MAC
        'anomaly_start'  : '2024-04-19 16:48:19',  # <- replace with your actual time
        'description'    : 'Strong hrping retry storm with retries up to 144 and tx_failed up to 18',
    },
    # Uncomment and fill in as you collect each experiment:
    # {
    #     'name'           : 'signal_degradation',
    #     'csv_file'       : 'data/exp_signal_degradation_ml_ready.csv',
    #     'experiment_mac' : 'a8:74:1d:47:e6:8e',
    #     'anomaly_start'  : '2025-05-10 15:10:00',
    #     'description'    : 'device moved 15m from AP, blocked by wall',
    # },
    # {
    #     'name'           : 'link_failure',
    #     'csv_file'       : 'data/exp_link_failure_ml_ready.csv',
    #     'experiment_mac' : 'a8:74:1d:47:e6:8e',
    #     'anomaly_start'  : '2025-05-10 16:10:00',
    #     'description'    : 'WiFi toggled off/on 8 times over 10 minutes',
    # },
    # {
    #     'name'           : 'traffic_burst',
    #     'csv_file'       : 'data/exp_traffic_burst_ml_ready.csv',
    #     'experiment_mac' : 'a8:74:1d:47:e6:8e',
    #     'anomaly_start'  : '2025-05-10 17:10:00',
    #     'description'    : 'large file download ~2GB sustained for 10 minutes',
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


# ── Score a dataframe through the right model per row ─────────────────────────

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
        df.loc[idle_mask, 'anomaly_flag']  = (models['idle_model'].predict(X) == -1).astype(int)
        df.loc[idle_mask, 'model_used']    = 'idle'

    if active_mask.sum() > 0 and len(avail_active) > 0:
        # Use available features only — experiment file may have fewer columns
        X = models['scaler_active'].transform(df.loc[active_mask, avail_active].values)
        df.loc[active_mask, 'anomaly_score'] = models['active_model'].score_samples(X)
        df.loc[active_mask, 'anomaly_flag']  = (models['active_model'].predict(X) == -1).astype(int)
        df.loc[active_mask, 'model_used']    = 'active'

    return df


# ── Feature comparison table ──────────────────────────────────────────────────

def feature_comparison(df_normal_active, df_anomaly_active, exp_name):
    """
    Print side-by-side feature averages: normal period vs anomaly period.
    This is the most important diagnostic — shows exactly what changed.
    """
    lines = []
    lines.append(f"\n  Feature comparison — {exp_name}")
    lines.append(f"  {'Feature':35s}  {'Normal avg':>12}  {'Anomaly avg':>12}  {'Change':>10}")
    lines.append(f"  {'-'*35}  {'-'*12}  {'-'*12}  {'-'*10}")

    available = [f for f in COMPARISON_FEATURES if f in df_normal_active.columns
                 and f in df_anomaly_active.columns]

    for feat in available:
        n_mean = df_normal_active[feat].mean()
        a_mean = df_anomaly_active[feat].mean()

        if n_mean == 0:
            change_str = 'N/A'
        else:
            change_pct = 100 * (a_mean - n_mean) / abs(n_mean)
            arrow = '↑' if change_pct > 5 else ('↓' if change_pct < -5 else '~')
            change_str = f"{arrow} {change_pct:+.0f}%"

        lines.append(
            f"  {feat:35s}  {n_mean:>12.4f}  {a_mean:>12.4f}  {change_str:>10}"
        )

    return '\n'.join(lines)


# ── Evaluate one experiment ───────────────────────────────────────────────────

def evaluate_experiment(exp, models, all_report_lines):
    name         = exp['name']
    csv_file     = exp['csv_file']
    mac          = exp['experiment_mac'].lower()
    anomaly_start = pd.Timestamp(exp['anomaly_start'])
    description  = exp['description']

    print(f"\n{'='*55}")
    print(f"  EXPERIMENT: {name.upper()}")
    print(f"{'='*55}")

    def log(msg=''):
        print(msg)
        all_report_lines.append(msg)

    log(f"\n{'='*55}")
    log(f"  EXPERIMENT: {name.upper()}")
    log(f"  {description}")
    log(f"  Device : {mac}")
    log(f"  Anomaly starts : {anomaly_start}")
    log(f"{'='*55}")

    # ── Load experiment CSV ───────────────────────────────────────────────────
    if not os.path.exists(csv_file):
        log(f"  [SKIP] File not found: {csv_file}")
        log(f"         Run: python feature_engineering.py on your experiment data first")
        return

    df = pd.read_csv(csv_file, parse_dates=['timestamp'])
    log(f"  Total rows loaded : {len(df):,}")
    log(f"  Unique clients    : {df['client_mac'].nunique()}")

    # Normalise MAC addresses
    df['client_mac'] = df['client_mac'].str.lower().str.strip()

    # ── Filter to experiment device ───────────────────────────────────────────
    df_device = df[df['client_mac'] == mac].copy()

    if len(df_device) == 0:
        log(f"  [ERROR] MAC {mac} not found in file.")
        log(f"  Available MACs: {list(df['client_mac'].unique())}")
        return

    df_device = df_device.sort_values('timestamp').reset_index(drop=True)
    log(f"  Rows for device   : {len(df_device):,}")
    log(f"  Time range        : {df_device['timestamp'].min()} -> {df_device['timestamp'].max()}")

    # ── Split into normal and anomaly periods ─────────────────────────────────
    df_normal  = df_device[df_device['timestamp'] <  anomaly_start].copy()
    df_anomaly = df_device[df_device['timestamp'] >= anomaly_start].copy()

    log(f"\n  Period split:")
    log(f"    Normal period  : {len(df_normal):,} rows  "
        f"({df_normal['timestamp'].min() if len(df_normal) > 0 else 'N/A'} -> {anomaly_start})")
    log(f"    Anomaly period : {len(df_anomaly):,} rows  "
        f"({anomaly_start} -> {df_device['timestamp'].max()})")

    if len(df_normal) == 0:
        log("  [WARN] No normal period rows — check your anomaly_start timestamp")
        log("         It may be set before any data was collected")
    if len(df_anomaly) == 0:
        log("  [WARN] No anomaly period rows — check your anomaly_start timestamp")
        log("         It may be set after all data was collected")
        return

    # ── Score both periods ────────────────────────────────────────────────────
    df_normal_scored  = score_rows(df_normal,  models)
    df_anomaly_scored = score_rows(df_anomaly, models)

    # ── Results for normal period (false alarm check) ─────────────────────────
    log(f"\n  NORMAL PERIOD RESULTS (false alarm check):")
    if len(df_normal_scored) > 0:
        n_total   = len(df_normal_scored)
        n_flagged = df_normal_scored['anomaly_flag'].sum()
        n_active  = df_normal_scored['is_active'].sum()
        fa_rate   = 100 * n_flagged / n_total if n_total > 0 else 0
        log(f"    Total rows      : {n_total}")
        log(f"    Active rows     : {n_active} ({100*n_active/n_total:.1f}%)")
        log(f"    Flagged         : {n_flagged} ({fa_rate:.1f}%)")
        log(f"    False alarm rate: {fa_rate:.1f}%  (training baseline was 1.5%)")
        if fa_rate > 5:
            log(f"    [WARN] False alarm rate higher than training baseline — check normal period data")
    else:
        log("    No normal period rows to evaluate")
        fa_rate = 0

    # ── Results for anomaly period (detection check) ──────────────────────────
    log(f"\n  ANOMALY PERIOD RESULTS (detection check):")
    a_total   = len(df_anomaly_scored)
    a_flagged = df_anomaly_scored['anomaly_flag'].sum()
    a_active  = df_anomaly_scored['is_active'].sum()
    det_rate  = 100 * a_flagged / a_total if a_total > 0 else 0

    # Detection rate on active rows only (more meaningful — idle rows are harder to flag)
    a_active_rows  = df_anomaly_scored[df_anomaly_scored['is_active'] == 1]
    a_active_flagged = a_active_rows['anomaly_flag'].sum()
    det_rate_active = 100 * a_active_flagged / len(a_active_rows) if len(a_active_rows) > 0 else 0

    log(f"    Total rows        : {a_total}")
    log(f"    Active rows       : {a_active} ({100*a_active/a_total:.1f}% of anomaly period)")
    log(f"    Flagged (all)     : {a_flagged} ({det_rate:.1f}%)")
    log(f"    Flagged (active)  : {a_active_flagged}/{len(a_active_rows)} ({det_rate_active:.1f}%)")
    log(f"")
    log(f"    ** DETECTION RATE (active rows): {det_rate_active:.1f}% **")

    if det_rate_active == 0:
        log(f"    [NOTE] 0% detection on active rows.")
        log(f"           Possible reasons:")
        log(f"           1. anomaly_start timestamp is wrong — adjust it")
        log(f"           2. active rows are too few in anomaly period — stress condition")
        log(f"              may not have generated enough traffic (need >10 active rows)")
        log(f"           3. The anomaly is too subtle for IF — expected for retry_storm")
        log(f"              (synthetic result was also 16%) -> Autoencoder should do better")
    elif det_rate_active < 20:
        log(f"    [NOTE] Low detection rate — consistent with IF's known weakness on subtle")
        log(f"           degradation. This is your finding, not a failure.")
        log(f"           Document this and compare with Autoencoder results.")

    # ── Feature comparison ────────────────────────────────────────────────────
    normal_active  = df_normal_scored[df_normal_scored['is_active'] == 1]
    anomaly_active = df_anomaly_scored[df_anomaly_scored['is_active'] == 1]

    if len(normal_active) > 0 and len(anomaly_active) > 0:
        comparison = feature_comparison(normal_active, anomaly_active, name)
        log(comparison)
    else:
        log(f"\n  [SKIP] Feature comparison requires active rows in both periods")
        log(f"         Normal active: {len(normal_active)}, Anomaly active: {len(anomaly_active)}")

    # ── Score distribution ────────────────────────────────────────────────────
    log(f"\n  ANOMALY SCORE DISTRIBUTION:")
    log(f"    Normal period  — mean: {df_normal_scored['anomaly_score'].mean():.4f}  "
        f"min: {df_normal_scored['anomaly_score'].min():.4f}  "
        f"max: {df_normal_scored['anomaly_score'].max():.4f}")
    log(f"    Anomaly period — mean: {df_anomaly_scored['anomaly_score'].mean():.4f}  "
        f"min: {df_anomaly_scored['anomaly_score'].min():.4f}  "
        f"max: {df_anomaly_scored['anomaly_score'].max():.4f}")
    log(f"    (Lower score = more anomalous in Isolation Forest)")

    # ── Thesis result line ────────────────────────────────────────────────────
    log(f"\n  THESIS RESULT LINE:")
    log(f"    {name}: IF detection rate = {det_rate_active:.1f}% on active rows  |  "
        f"false alarm rate = {fa_rate:.1f}%")

    # ── Save scored output ────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_all = pd.concat([df_normal_scored, df_anomaly_scored], ignore_index=True)
    df_all['period'] = 'normal'
    df_all.loc[df_all['timestamp'] >= anomaly_start, 'period'] = 'anomaly'
    out_path = f"{RESULTS_DIR}/real_{name}_evaluation.csv"
    df_all.to_csv(out_path, index=False)
    log(f"\n  Saved: {out_path}")
    log(f"  (Contains 'period' column: normal/anomaly — use for thesis plots)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[INFO] Loading trained models...")
    models = load_models()
    print("[INFO] Models loaded successfully.")

    all_report_lines = [
        "REAL EXPERIMENT EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55,
    ]

    if not EXPERIMENTS:
        print("[WARN] No experiments configured.")
        print("       Edit the EXPERIMENTS list at the top of this script.")
        return

    print(f"[INFO] Running {len(EXPERIMENTS)} experiment(s)...")

    for exp in EXPERIMENTS:
        evaluate_experiment(exp, models, all_report_lines)

    # ── Summary table ─────────────────────────────────────────────────────────
    all_report_lines.append(f"\n{'='*55}")
    all_report_lines.append("  SUMMARY TABLE FOR THESIS")
    all_report_lines.append(f"  {'Experiment':25s}  {'Detection (active)':>20}  {'False alarm':>12}")
    all_report_lines.append(f"  {'-'*25}  {'-'*20}  {'-'*12}")
    all_report_lines.append(f"  (fill from individual results above)")
    all_report_lines.append(f"  Retry storm (real)         : __ %                  __ %")
    all_report_lines.append(f"  Signal degradation (real)  : __ %                  __ %")
    all_report_lines.append(f"  Link failure (real)        : __ %                  __ %")
    all_report_lines.append(f"  Traffic burst (real)       : __ %                  __ %")
    all_report_lines.append(f"\n  Comparison (from synthetic testing):")
    all_report_lines.append(f"  Retry storm (synthetic)    : 16 %                 1.5 %")
    all_report_lines.append(f"  Signal degradation (synth) : 12 %                 1.5 %")
    all_report_lines.append(f"  Link failure (synthetic)   : 100%                 1.5 %")
    all_report_lines.append(f"  Traffic burst (synthetic)  : 100%                 1.5 %")

    report_path = f"{RESULTS_DIR}/real_experiment_summary.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_report_lines))

    print(f"\n[INFO] Full report saved: {report_path}")
    print(f"[INFO] Done.")


if __name__ == '__main__':
    main()
