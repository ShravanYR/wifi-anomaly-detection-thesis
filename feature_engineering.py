#!/usr/bin/env python3
"""
WiFi Feature Engineering — Corrected Version (+ RX error rate added)
"""

import pandas as pd
import numpy as np
import re

INPUT_FILE  = "clean_output_ex_signalD.csv"
OUTPUT_FILE = "ml_ready_data_ex_signalD.csv"


def get_mcs_index(col):
    m = re.search(r'mcs_(\d+)', col)
    return int(m.group(1)) if m else -1


def safe_col(df, name, default=0):
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['client_mac', 'timestamp']).reset_index(drop=True)

    # ── Time delta
    df['time_delta'] = (
        df.groupby('client_mac')['timestamp']
        .diff()
        .dt.total_seconds()
        .replace(0, np.nan)
    )

    # ── Activity flag
    tx_delta = df.groupby('client_mac')['tx_data_packets'].diff().fillna(0)
    rx_delta = df.groupby('client_mac')['rx_data_packets'].diff().fillna(0)
    df['is_active'] = ((tx_delta > 0) | (rx_delta > 0)).astype(int)

    # ── Traffic rates
    print("[INFO] Building traffic rate features...")

    df['tx_packet_rate'] = tx_delta / df['time_delta']
    df['rx_packet_rate'] = rx_delta / df['time_delta']

    tx_bytes_delta = df.groupby('client_mac')['tx_data_bytes'].diff().fillna(0)
    rx_bytes_delta = df.groupby('client_mac')['rx_data_bytes'].diff().fillna(0)

    df['tx_byte_rate'] = tx_bytes_delta / df['time_delta']
    df['rx_byte_rate'] = rx_bytes_delta / df['time_delta']

    # ── TX success rate
    tx_success_delta = df.groupby('client_mac')['tx_success_data_packets'].diff().fillna(0)
    denom = tx_delta.replace(0, np.nan)

    df['tx_success_rate'] = (tx_success_delta / denom).clip(0, 1)

    # ── Retry rate
    retry_col = 'msdu_tx_retry_count' if 'msdu_tx_retry_count' in df.columns else \
                'msdu_tx_retry_count_'
    retry_delta = df.groupby('client_mac')[retry_col].diff().fillna(0)
    df['retry_rate'] = (retry_delta / denom).clip(0, 2)

    # ── TX failed rate
    failed_col = 'tx_failed' if 'tx_failed' in df.columns else 'tx_failures'
    failed_delta = df.groupby('client_mac')[failed_col].diff().fillna(0)
    df['tx_failed_rate'] = (failed_delta / denom).clip(0, 1)

    # ── NEW: RX error rate (ADDED)
    rx_packets_delta = df.groupby('client_mac')['rx_data_packets'].diff().fillna(0)
    rx_errors_delta = df.groupby('client_mac')['rx_errors'].diff().fillna(0)

    df['rx_error_rate'] = (rx_errors_delta / rx_packets_delta.replace(0, np.nan)).clip(0, 1)

    # ── RSSI
    df['rssi'] = safe_col(df, 'rx_rssi')

    # ── Security errors
    df['security_errors'] = (
        safe_col(df, 'rx_mic_errors') +
        safe_col(df, 'rx_pn_errors') +
        safe_col(df, 'rx_decryption_errors')
    )

    # ── Throughput
    df['tx_throughput_mbps'] = safe_col(df, 'avg_ppdu_tx_rate_kbps') / 1000
    df['rx_throughput_mbps'] = safe_col(df, 'avg_ppdu_rx_rate_kbps') / 1000

    if (df['tx_throughput_mbps'] == 0).all():
        df['tx_throughput_mbps'] = safe_col(df, 'average_tx_rate_kbps') / 1000
    if (df['rx_throughput_mbps'] == 0).all():
        df['rx_throughput_mbps'] = safe_col(df, 'average_rx_rate_kbps') / 1000

    # ── MCS features
    print("[INFO] Computing MCS features...")

    mcs_cols = [
        c for c in df.columns
        if c.startswith('tx_') and 'mcs' in c and 'ofdma' not in c
    ]

    if mcs_cols:
        low_cols  = [c for c in mcs_cols if 0 <= get_mcs_index(c) <= 3]
        mid_cols  = [c for c in mcs_cols if 4 <= get_mcs_index(c) <= 7]
        high_cols = [c for c in mcs_cols if get_mcs_index(c) >= 8]

        total_mcs = df[mcs_cols].sum(axis=1).replace(0, np.nan)

        df['low_mcs_ratio']  = df[low_cols].sum(axis=1)  / total_mcs
        df['mid_mcs_ratio']  = df[mid_cols].sum(axis=1)  / total_mcs
        df['high_mcs_ratio'] = df[high_cols].sum(axis=1) / total_mcs

        df['mcs_weighted_score'] = sum(
            df[c] * get_mcs_index(c) for c in mcs_cols if get_mcs_index(c) >= 0
        ) / total_mcs

        def mcs_entropy(row):
            vals = row[mcs_cols].values.astype(float)
            total = vals.sum()
            if total == 0:
                return np.nan
            p = vals / total
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        df['mcs_entropy'] = df.apply(mcs_entropy, axis=1)

        for col in ['low_mcs_ratio', 'mid_mcs_ratio', 'high_mcs_ratio',
                    'mcs_weighted_score', 'mcs_entropy']:
            df.loc[df['is_active'] == 0, col] = np.nan

    # ── Clean
    print("[INFO] Cleaning...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # ── Final features
    print("[INFO] Selecting final features...")

    final_columns = [
        'timestamp',
        'client_mac',
        'is_active',
        'tx_packet_rate',
        'rx_packet_rate',
        'tx_byte_rate',
        'rx_byte_rate',
        'tx_success_rate',
        'retry_rate',
        'tx_failed_rate',
        'rx_error_rate',   # ✅ ADDED HERE
        'rssi',
        'security_errors',
        'tx_throughput_mbps',
        'rx_throughput_mbps',
        'low_mcs_ratio',
        'mid_mcs_ratio',
        'high_mcs_ratio',
        'mcs_weighted_score',
        'mcs_entropy',
    ]

    final_columns = [c for c in final_columns if c in df.columns]
    df_final = df[final_columns]

    df_final.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Feature engineering complete")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(df_final)}")


if __name__ == "__main__":
    main()