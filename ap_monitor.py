#!/usr/bin/env python3
"""
ap_monitor.py
==============
The live, continuously-running WiFi anomaly monitor for the AP.
Pure Python 3 standard library only — no pip, no internet, no third-party
packages, by design (this device has no network route to install anything).

WHAT IT DOES, every POLL_INTERVAL seconds, forever:
    1. Run `apstats -a -R` and capture its output
    2. Parse it into per-client raw field dicts (apstats_parser.py)
    3. For each client, diff this reading against its previous reading to
       get the 17 trained features (feature_calculator.py)
    4. Skip scoring for any client on its first-ever reading (no previous
       reading to diff against yet — this is "establishing baseline")
    5. Score the features through the idle or active Autoencoder, whichever
       matches this row's is_active flag (ae_inference.py)
    6. If the reconstruction error exceeds that model's threshold, append
       one line to the alert log
    7. Sleep, repeat — forever, until the process is stopped

FILES THIS SCRIPT NEEDS IN THE SAME FOLDER:
    apstats_parser.py
    feature_calculator.py
    ae_inference.py
    ae_weights_export.json

OUTPUT:
    ap_monitor.log   — one line per cycle: heartbeat or anomaly alert
    Log size is capped — see MAX_LOG_LINES — so disk usage never grows
    unbounded on a device with only ~422MB free.

USAGE:
    python3 ap_monitor.py
    (Ctrl+C to stop manually; in production this runs under systemd — see
     the accompanying .service file)
"""

import subprocess
import time
import json
import os
import sys

from apstats_parser import parse_snapshot
from feature_calculator import ClientFeatureState
from ae_inference import AEModel

# ── Configuration ─────────────────────────────────────────────────────────────

POLL_INTERVAL   = 10            # seconds between polls — matches training data interval
WEIGHTS_PATH    = 'ae_weights_export.json'
LOG_PATH        = 'ap_monitor.log'
MAX_LOG_LINES   = 5000           # caps log file size; oldest lines trimmed when exceeded
APSTATS_CMD     = ['apstats', '-a', '-R']
APSTATS_TIMEOUT = 5              # seconds — never let a hung subprocess stall the loop


def log_line(text):
    """Append one line to the log, trimming the file if it grows too large."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}\n"
    print(line, end='')   # also visible live if run interactively / via journalctl
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH) as f:
                lines = f.readlines()
            if len(lines) >= MAX_LOG_LINES:
                lines = lines[-(MAX_LOG_LINES // 2):]   # trim to half, avoid trimming every cycle
            lines.append(line)
        else:
            lines = [line]
        with open(LOG_PATH, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"[WARN] Could not write log: {e}")


def run_apstats():
    """Run apstats -a -R and return its stdout text, or None on failure."""
    try:
        result = subprocess.run(
            APSTATS_CMD, capture_output=True, text=True, timeout=APSTATS_TIMEOUT
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        log_line("[ERROR] apstats call timed out")
        return None
    except FileNotFoundError:
        log_line("[ERROR] apstats command not found — check PATH")
        return None
    except Exception as e:
        log_line(f"[ERROR] apstats call failed: {e}")
        return None


def main():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[FATAL] {WEIGHTS_PATH} not found in this folder. "
              f"Copy it here before starting.")
        sys.exit(1)

    log_line(f"=== AP Monitor starting (poll interval: {POLL_INTERVAL}s) ===")

    idle_model = AEModel.load(WEIGHTS_PATH, 'idle')
    active_model = AEModel.load(WEIGHTS_PATH, 'active')
    log_line(f"Models loaded. idle_threshold={idle_model.threshold:.6f}  "
             f"active_threshold={active_model.threshold:.6f}")

    client_states = {}   # mac -> ClientFeatureState
    cycle = 0

    while True:
        cycle += 1
        cycle_start = time.time()

        raw_text = run_apstats()
        if raw_text is None:
            time.sleep(POLL_INTERVAL)
            continue

        try:
            clients = parse_snapshot(raw_text)
        except Exception as e:
            log_line(f"[ERROR] Parse failure on cycle {cycle}: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        n_scored = 0
        n_anomalous = 0
        n_baseline = 0

        for mac, raw in clients.items():
            if mac not in client_states:
                client_states[mac] = ClientFeatureState()

            try:
                result = client_states[mac].compute_features(raw, cycle_start)
            except Exception as e:
                log_line(f"[ERROR] Feature calc failed for {mac}: {e}")
                continue

            if result is None:
                n_baseline += 1
                continue   # first-ever reading for this client — no score yet

            is_active, features = result
            model = active_model if is_active else idle_model

            try:
                mse, is_anomaly = model.score_from_dict(features)
            except KeyError as e:
                log_line(f"[ERROR] Missing feature {e} for {mac} — skipping")
                continue

            n_scored += 1
            if is_anomaly:
                n_anomalous += 1
                top_changed = sorted(
                    features.items(), key=lambda kv: abs(kv[1]), reverse=True
                )[:3]
                changed_str = ', '.join(f"{k}={v:.4f}" for k, v in top_changed)
                log_line(
                    f"[ANOMALY] {mac}  model={'active' if is_active else 'idle'}  "
                    f"mse={mse:.6f}  threshold={model.threshold:.6f}  "
                    f"top_features=({changed_str})"
                )

        # One heartbeat line per cycle, even when nothing anomalous happened —
        # this is how you confirm the service is alive without grepping logs.
        log_line(
            f"[heartbeat] cycle={cycle}  clients={len(clients)}  "
            f"scored={n_scored}  anomalous={n_anomalous}  "
            f"establishing_baseline={n_baseline}"
        )

        elapsed = time.time() - cycle_start
        sleep_for = max(0.0, POLL_INTERVAL - elapsed)
        time.sleep(sleep_for)


if __name__ == '__main__':
    main()
