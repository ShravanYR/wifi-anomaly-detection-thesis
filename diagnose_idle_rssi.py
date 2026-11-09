#!/usr/bin/env python3
"""
diagnose_idle_rssi.py
=======================
Loads your REAL trained idle model (ae_weights_export.json) and computes
the actual reconstruction error (MSE) for a set of specific rssi values —
no guessing, no fabricated numbers, just your real model's real output.

WHY THIS EXISTS:
    We confirmed mathematically that the idle model's GLOBAL check needs
    zero history/warm-up — it scores every reading fresh. So when it stays
    silent at rssi=24, that is the trained model's genuine, considered
    answer: "not unusual." This script shows you the EXACT numbers behind
    that answer, using your real weights, so you have concrete figures to
    cite rather than a distributional argument alone.

WHAT IT COMPUTES, for each test rssi value (with tx_success_rate=0,
retry_rate=0, matching genuine idle behaviour):
    - mse              : this exact reading's reconstruction error
    - vs threshold      : whether the GLOBAL check alone would fire
    - vs training mean  : ratio against mse at rssi=27 (the training mean),
                           used as a stand-in "typical" baseline — this is
                           the same kind of ratio ap_monitor.py's RELATIVE
                           check computes against a client's own history,
                           shown here so you can see directly whether a
                           given drop would cross RELATIVE_MULTIPLIER (4.0x)

RSSI VALUES TESTED:
    51, 52   — the "strong signal" cluster actually seen in your PC test file
    27       — training mean / your AP's everyday baseline
    24, 23   — the "degraded" cluster actually seen in your PC test file
    18       — training 5th percentile (the real boundary of "rare")
    10, 5, 0 — genuinely rare values, well outside training experience

USAGE:
    python diagnose_idle_rssi.py ae_weights_export.json
"""

import sys
from ae_inference import AEModel

RELATIVE_MULTIPLIER = 4.0  # matches the constant used in ap_monitor.py

TEST_RSSI_VALUES = [52, 51, 29, 27, 26, 24, 23, 18, 10, 5, 0]
BASELINE_RSSI = 27.0   # training mean — used as the reference point for ratios


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_idle_rssi.py ae_weights_export.json")
        sys.exit(1)

    path = sys.argv[1]
    model = AEModel.load(path, 'idle')

    print(f"Idle model loaded from: {path}")
    print(f"Idle threshold (global check): {model.threshold:.6f}")
    print(f"Feature order: {model.feature_order}")
    print()

    # Compute the baseline reference MSE first
    baseline_features = {'rssi': BASELINE_RSSI, 'tx_success_rate': 0.0, 'retry_rate': 0.0}
    baseline_mse, _ = model.score_from_dict(baseline_features)
    print(f"Reference baseline: rssi={BASELINE_RSSI:.0f}  ->  mse={baseline_mse:.6f}")
    print(f"(this stands in for a client's own typical/calm reading)\n")

    print(f"{'rssi':>6s} {'mse':>12s} {'global_flag':>12s} {'ratio_vs_baseline':>18s} {'would_trip_4x?':>15s}")
    print("-" * 70)

    for rssi in TEST_RSSI_VALUES:
        features = {'rssi': float(rssi), 'tx_success_rate': 0.0, 'retry_rate': 0.0}
        mse, global_flag = model.score_from_dict(features)
        ratio = mse / baseline_mse if baseline_mse > 0 else float('inf')
        trips_relative = ratio > RELATIVE_MULTIPLIER

        print(f"{rssi:>6d} {mse:>12.6f} {str(global_flag):>12s} "
              f"{ratio:>17.2f}x {str(trips_relative):>15s}")

    print()
    print("Reading this table:")
    print("  global_flag        : would the TRAINED MODEL ALONE flag this")
    print("                       reading, with zero history, right now?")
    print("  ratio_vs_baseline  : how many times worse is this than a typical")
    print("                       (rssi=27) reading for this AP?")
    print("  would_trip_4x?     : would this cross ap_monitor.py's RELATIVE_MULTIPLIER")
    print("                       if it were compared against a baseline=27 reading?")
    print()
    print("Use this to find the rssi value where global_flag actually turns True —")
    print("that is the real, concrete threshold your live test needs to cross.")


if __name__ == '__main__':
    main()
