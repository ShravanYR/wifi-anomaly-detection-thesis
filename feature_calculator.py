#!/usr/bin/env python3
"""
feature_calculator.py
======================
Converts two consecutive raw apstats readings for one client into the same
17 features used in feature_engineering.py and trained on by autoencoder_training.py.

WHY THIS NEEDS "PREVIOUS READING" STATE:
    A single apstats -a -R call gives you raw counters at one instant.
    Rates and ratios (packets/sec, retry fraction, MCS distribution this
    interval) need TWO readings — current minus previous, divided by the
    elapsed time between them. This is exactly what feature_engineering.py
    does with pandas .diff() across consecutive rows; here it's done one
    client at a time, in memory, as new readings arrive live.

WHICH FIELDS ARE DIFFERENCED vs USED DIRECTLY (matches feature_engineering.py exactly):
    DIFFERENCED (current - previous, then /time_delta for rates):
        tx_data_packets, rx_data_packets, tx_data_bytes, rx_data_bytes,
        tx_success_data_packets, msdu_tx_retry_count, tx_failed,
        all TX MCS histogram bins (he/vht/ht/ofdm/cck)
    USED DIRECTLY, no differencing:
        rx_rssi, avg_ppdu_tx_rate_kbps, avg_ppdu_rx_rate_kbps,
        rx_mic_errors, rx_pn_errors, rx_decryption_errors

A client's FIRST-EVER reading has no previous reading to diff against, so
compute_features() returns None for it — the live monitor treats this as
"establishing baseline" and does not attempt to score that cycle.
"""

import math


# All TX MCS histogram column names this AP's firmware can produce.
# (HE is what your WiFi 6 clients use; HT/VHT/OFDM/CCK covers older devices
# on the same AP, matching the same superset handled in parse_apstats.py.)
def is_tx_mcs_field(key):
    return key.startswith('tx_') and any(
        fam in key for fam in ('ofdm', 'cck', 'ht_mcs', 'vht_mcs', 'he_mcs')
    ) and 'ofdma' not in key


def mcs_index_from_key(key):
    """'tx_he_mcs_11_1024_qam_5_6' -> 11. Returns -1 if no index found."""
    import re
    m = re.search(r'mcs_(\d+)', key)
    return int(m.group(1)) if m else -1


def shannon_entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


class ClientFeatureState:
    """Holds the previous raw reading + timestamp for ONE client MAC."""
    def __init__(self):
        self.prev_raw = None
        self.prev_ts = None

    def compute_features(self, raw, ts):
        """
        raw: dict of field_name -> int, from apstats_parser.parse_snapshot()
        ts : float, time.time() at the moment this reading was taken

        Returns a dict of the 17 trained feature names -> float, or None if
        this is the first reading ever seen for this client (no previous
        reading to diff against yet).
        """
        if self.prev_raw is None:
            self.prev_raw = raw
            self.prev_ts = ts
            return None   # establishing baseline — caller should skip scoring

        time_delta = ts - self.prev_ts
        if time_delta <= 0:
            # Clock didn't advance (shouldn't happen, but never divide by zero)
            self.prev_raw, self.prev_ts = raw, ts
            return None

        def delta(field, default=0):
            return raw.get(field, default) - self.prev_raw.get(field, default)

        def safe_div(a, b):
            return a / b if b else 0.0

        tx_delta      = delta('tx_data_packets')
        rx_delta      = delta('rx_data_packets')
        tx_byte_delta = delta('tx_data_bytes')
        rx_byte_delta = delta('rx_data_bytes')
        tx_success_delta = delta('tx_success_data_packets')
        retry_delta   = delta('msdu_tx_retry_count')
        failed_delta  = delta('tx_failed')

        is_active = 1 if (tx_delta > 0 or rx_delta > 0) else 0

        features = {
            'tx_packet_rate'    : safe_div(tx_delta, time_delta),
            'rx_packet_rate'    : safe_div(rx_delta, time_delta),
            'tx_byte_rate'      : safe_div(tx_byte_delta, time_delta),
            'rx_byte_rate'      : safe_div(rx_byte_delta, time_delta),
            'tx_success_rate'   : max(0.0, min(1.0, safe_div(tx_success_delta, tx_delta))) if tx_delta else 0.0,
            'retry_rate'        : max(0.0, min(2.0, safe_div(retry_delta, tx_delta))) if tx_delta else 0.0,
            'tx_failed_rate'    : max(0.0, min(1.0, safe_div(failed_delta, tx_delta))) if tx_delta else 0.0,
            'rssi'              : float(raw.get('rx_rssi', 0)),                     # direct, no diff
            'tx_throughput_mbps': raw.get('avg_ppdu_tx_rate_kbps', 0) / 1000.0,      # direct, no diff
            'rx_throughput_mbps': raw.get('avg_ppdu_rx_rate_kbps', 0) / 1000.0,      # direct, no diff
        }

        # ── MCS features: differenced histogram bins, this interval only ──────
        mcs_deltas = {}
        for key in raw:
            if is_tx_mcs_field(key):
                mcs_deltas[key] = max(0, delta(key))   # clip negative (counter reset edge case)

        total_mcs = sum(mcs_deltas.values())
        low_sum  = sum(v for k, v in mcs_deltas.items() if 0 <= mcs_index_from_key(k) <= 3)
        mid_sum  = sum(v for k, v in mcs_deltas.items() if 4 <= mcs_index_from_key(k) <= 7)
        high_sum = sum(v for k, v in mcs_deltas.items() if mcs_index_from_key(k) >= 8)

        if total_mcs > 0:
            features['low_mcs_ratio']  = low_sum / total_mcs
            features['mid_mcs_ratio']  = mid_sum / total_mcs
            features['high_mcs_ratio'] = high_sum / total_mcs
            features['mcs_weighted_score'] = sum(
                v * mcs_index_from_key(k) for k, v in mcs_deltas.items() if mcs_index_from_key(k) >= 0
            ) / total_mcs
            features['mcs_entropy'] = shannon_entropy(list(mcs_deltas.values()))
        else:
            features['low_mcs_ratio']      = 0.0
            features['mid_mcs_ratio']      = 0.0
            features['high_mcs_ratio']     = 0.0
            features['mcs_weighted_score'] = 0.0
            features['mcs_entropy']        = 0.0

        # security_errors: summed directly from the latest reading, not differenced
        # (these are rare/sparse counters; -R semantics already make a single
        # reading's value represent "since last reset", matching feature_engineering.py)
        features['security_errors'] = (
            raw.get('rx_mic_errors', 0) + raw.get('rx_pn_errors', 0)
            + raw.get('rx_decryption_errors', 0)
        )

        # Update state for next cycle
        self.prev_raw, self.prev_ts = raw, ts

        return is_active, features


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Self-test: two synthetic consecutive readings, 10 seconds apart\n")

    reading_1 = {
        'tx_data_packets': 1000, 'rx_data_packets': 1000,
        'tx_data_bytes': 50000, 'rx_data_bytes': 50000,
        'tx_success_data_packets': 1000, 'msdu_tx_retry_count': 10, 'tx_failed': 0,
        'rx_rssi': 52, 'avg_ppdu_tx_rate_kbps': 200000, 'avg_ppdu_rx_rate_kbps': 250000,
        'tx_he_mcs_11_1024_qam_5_6': 500, 'tx_he_mcs_9_256_qam_5_6': 300,
        'rx_mic_errors': 0, 'rx_pn_errors': 0, 'rx_decryption_errors': 0,
    }
    reading_2 = {
        'tx_data_packets': 1020, 'rx_data_packets': 1020,        # +20 packets in 10s -> 2.0 pkt/s
        'tx_data_bytes': 51000, 'rx_data_bytes': 51000,           # +1000 bytes -> 100 B/s
        'tx_success_data_packets': 1019, 'msdu_tx_retry_count': 12, 'tx_failed': 1,
        'rx_rssi': 36, 'avg_ppdu_tx_rate_kbps': 80000, 'avg_ppdu_rx_rate_kbps': 90000,
        'tx_he_mcs_11_1024_qam_5_6': 502, 'tx_he_mcs_9_256_qam_5_6': 308,  # +2, +8 -> mostly high MCS
        'rx_mic_errors': 0, 'rx_pn_errors': 0, 'rx_decryption_errors': 0,
    }

    state = ClientFeatureState()
    result1 = state.compute_features(reading_1, ts=1000.0)
    print(f"First reading  -> {result1}  (expected: None, establishing baseline)")

    result2 = state.compute_features(reading_2, ts=1010.0)
    is_active, feats = result2
    print(f"\nSecond reading -> is_active={is_active}")
    for k, v in feats.items():
        print(f"   {k:22s} = {v}")

    print("\nHand-check:")
    print(f"   tx_packet_rate expected 2.0   -> got {feats['tx_packet_rate']}")
    print(f"   tx_byte_rate   expected 100.0 -> got {feats['tx_byte_rate']}")
    print(f"   tx_success_rate expected 19/20=0.95 -> got {feats['tx_success_rate']}")
    print(f"   retry_rate     expected 2/20=0.10   -> got {feats['retry_rate']}")
    print(f"   tx_failed_rate expected 1/20=0.05   -> got {feats['tx_failed_rate']}")
    print(f"   rssi           expected 36.0 (direct, no diff) -> got {feats['rssi']}")
    print(f"   tx_throughput_mbps expected 80.0 (direct) -> got {feats['tx_throughput_mbps']}")
    print(f"   high_mcs_ratio expected 1.0 (only high-index bins moved) -> got {feats['high_mcs_ratio']}")
