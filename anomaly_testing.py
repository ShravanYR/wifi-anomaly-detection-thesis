#!/usr/bin/env python3
"""
anomaly_testing.py
==================
Testing phase for the Isolation Forest WiFi anomaly detector.

TWO TESTING APPROACHES:

A) SYNTHETIC INJECTION (run now — no new data collection needed)
   Programmatically corrupts normal rows to simulate 4 real anomaly types.
   Tests whether the trained model can detect each type.

B) REAL ANOMALY EVALUATION (run after controlled experiments)
   Loads separately collected anomaly data, runs it through the trained model,
   reports detection rates per experiment type.

ANOMALY TYPES TESTED:
   1. Signal degradation  — RSSI drops, MCS falls to low indices
   2. Link failure        — tx_success = 0, tx_failed = 1 (like e6:96 in your data)
   3. Traffic burst       — packet rate and throughput spike far above normal
   4. Retry storm         — retry_rate spikes while MCS stays normal

USAGE:
   python anomaly_testing.py

INPUTS:
   ml_ready_data.csv          — your normal training data
   models/if_idle_model.pkl   — trained idle model
   models/if_active_model.pkl — trained active model
   models/scaler_idle.pkl
   models/scaler_active.pkl

OUTPUTS:
   results/test_synthetic_results.csv   — all injected rows with detection flags
   results/test_detection_report.txt    — detection rate per anomaly type
"""

#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE   = "ml_ready_data.csv"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"

N_PER_TYPE = 50

EXCLUDE_MACS = ['a8:74:1d:be:20:48', 'a8:74:1d:47:e6:4e']

IDLE_FEATURES = ['rssi']

ACTIVE_FEATURES = [
    'tx_packet_rate', 'rx_packet_rate', 'tx_byte_rate', 'rx_byte_rate',
    'tx_success_rate', 'retry_rate', 'tx_failed_rate', 'rssi',
    'tx_throughput_mbps', 'rx_throughput_mbps',
    'low_mcs_ratio', 'mid_mcs_ratio', 'high_mcs_ratio',
    'mcs_weighted_score', 'mcs_entropy',
]


# ── Load models ───────────────────────────────────────────────────────────────

def load_models():
    def load(path):
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

    if idle_mask.sum() > 0:
        X = models['scaler_idle'].transform(df.loc[idle_mask, IDLE_FEATURES].values)
        df.loc[idle_mask, 'anomaly_score'] = models['idle_model'].score_samples(X)
        df.loc[idle_mask, 'anomaly_flag']  = (models['idle_model'].predict(X) == -1).astype(int)
        df.loc[idle_mask, 'model_used']    = 'idle'

    if active_mask.sum() > 0:
        X = models['scaler_active'].transform(df.loc[active_mask, ACTIVE_FEATURES].values)
        df.loc[active_mask, 'anomaly_score'] = models['active_model'].score_samples(X)
        df.loc[active_mask, 'anomaly_flag']  = (models['active_model'].predict(X) == -1).astype(int)
        df.loc[active_mask, 'model_used']    = 'active'

    return df


# ── Signal degradation ────────────────────────────────────────────────────────

def inject_signal_degradation(df_active, n):
    rows = df_active.sample(n, replace=True).copy()
    rows['anomaly_type'] = 'signal_degradation'

    rows['rssi'] = np.random.randint(8, 16, n)

    rows['low_mcs_ratio']  = np.random.uniform(0.5, 0.95, n)
    rows['high_mcs_ratio'] = np.random.uniform(0.0, 0.1, n)
    rows['mid_mcs_ratio']  = 1 - rows['low_mcs_ratio'] - rows['high_mcs_ratio']
    rows['mid_mcs_ratio']  = rows['mid_mcs_ratio'].clip(0, 1)

    rows['mcs_weighted_score'] = np.random.uniform(0.5, 2.0, n)
    rows['mcs_entropy'] = np.random.uniform(1.5, 2.8, n)

    rows['retry_rate'] = np.random.uniform(0.2, 0.6, n)

    rows['tx_throughput_mbps'] *= np.random.uniform(0.1, 0.4, n)
    rows['rx_throughput_mbps'] *= np.random.uniform(0.1, 0.4, n)

    return rows


# ── Link failure ──────────────────────────────────────────────────────────────

def inject_link_failure(df_active, n):
    rows = df_active.sample(n, replace=True).copy()
    rows['anomaly_type'] = 'link_failure'

    rows['tx_success_rate'] = 0.0
    rows['tx_failed_rate']  = 1.0

    # ✅ FIXED
    rows['rssi'] = np.random.randint(5, 15, n)

    rows['low_mcs_ratio']      = np.random.uniform(0.95, 1.0, n)
    rows['high_mcs_ratio']     = 0.0
    rows['mid_mcs_ratio']      = np.random.uniform(0.0, 0.05, n)
    rows['mcs_weighted_score'] = np.random.uniform(0.3, 0.5, n)
    rows['mcs_entropy']        = np.random.uniform(0.9, 1.2, n)

    rows['tx_throughput_mbps'] = np.random.uniform(0.5, 3.0, n)

    return rows


# ── Traffic burst ─────────────────────────────────────────────────────────────

def inject_traffic_burst(df_active, n):
    rows = df_active.sample(n, replace=True).copy()
    rows['anomaly_type'] = 'traffic_burst'

    normal_max_pkt = df_active['tx_packet_rate'].quantile(0.95)

    rows['tx_packet_rate'] = normal_max_pkt * np.random.uniform(3.0, 8.0, n)
    rows['rx_packet_rate'] = normal_max_pkt * np.random.uniform(3.0, 8.0, n)

    normal_max_byte = df_active['tx_byte_rate'].quantile(0.95)

    rows['tx_byte_rate'] = normal_max_byte * np.random.uniform(3.0, 8.0, n)
    rows['rx_byte_rate'] = normal_max_byte * np.random.uniform(3.0, 8.0, n)

    # ✅ FIXED
    rows['tx_throughput_mbps'] = np.random.uniform(200, 500, n)
    rows['rx_throughput_mbps'] = np.random.uniform(300, 700, n)

    rows['retry_rate'] = np.random.uniform(0.15, 0.5, n)

    return rows


# ── Retry storm ───────────────────────────────────────────────────────────────

def inject_retry_storm(df_active, n):
    rows = df_active.sample(n, replace=True).copy()
    rows['anomaly_type'] = 'retry_storm'

    # ✅ FIXED
    rows['retry_rate'] = np.random.uniform(0.6, 1.0, n)

    rows['tx_success_rate'] = np.random.uniform(0.55, 0.80, n)
    rows['tx_failed_rate']  = np.random.uniform(0.05, 0.25, n)

    rows['high_mcs_ratio'] = rows['high_mcs_ratio'] * np.random.uniform(0.6, 0.9, n)
    rows['mcs_entropy'] += np.random.uniform(0.3, 0.8, n)

    return rows


# ── Main synthetic testing ────────────────────────────────────────────────────

def run_synthetic_tests(df_normal, models):
    print("\n[TEST] Running synthetic anomaly injection...")

    df_active = df_normal[df_normal['is_active'] == 1].copy()

    n = min(N_PER_TYPE, len(df_active) // 4)

    injected_frames = []

    for fn, label in [
        (inject_signal_degradation, 'signal_degradation'),
        (inject_link_failure,       'link_failure'),
        (inject_traffic_burst,      'traffic_burst'),
        (inject_retry_storm,        'retry_storm'),
    ]:
        rows = fn(df_active, n)
        rows['is_active'] = 1
        rows['ground_truth'] = 1
        injected_frames.append(rows)
        print(f"Injected {n} rows of {label}")

    normal_sample = df_active.sample(min(n * 4, len(df_active))).copy()
    normal_sample['anomaly_type'] = 'normal'
    normal_sample['ground_truth'] = 0

    injected_frames.append(normal_sample)

    df_test = pd.concat(injected_frames, ignore_index=True)
    df_test = score_rows(df_test, models)

    return df_test


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(df_test):
    print("\n================ TEST RESULTS ================")

    for atype in sorted(df_test['anomaly_type'].unique()):
        subset = df_test[df_test['anomaly_type'] == atype]

        total = len(subset)
        flagged = subset['anomaly_flag'].sum()

        if subset['ground_truth'].iloc[0] == 1:
            rate = 100 * flagged / total
            print(f"{atype:25s} : {flagged}/{total} detected ({rate:.1f}%)")
        else:
            rate = 100 * flagged / total
            print(f"normal rows              : {flagged}/{total} false alarms ({rate:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    df = df[~df['client_mac'].isin(EXCLUDE_MACS)]

    print(f"[INFO] Loaded {len(df):,} rows")

    print("[INFO] Loading models...")
    models = load_models()

    print("[INFO] Running synthetic tests...")
    df_test = run_synthetic_tests(df, models)

    df_test.to_csv(f"{RESULTS_DIR}/test_synthetic_results.csv", index=False)

    evaluate(df_test)

    print("\n✅ Synthetic testing complete")
    print(f"Saved: {RESULTS_DIR}/test_synthetic_results.csv")


if __name__ == '__main__':
    main()