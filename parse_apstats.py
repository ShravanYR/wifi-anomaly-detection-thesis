#!/usr/bin/env python3
"""
parse_apstats.py  —  WiFi Thesis Data Parser (Final Production Version)
========================================================================
Converts raw apstats collection log -> clean CSV for ML pipeline.
Tested against real AP output with 8 clients (Qualcomm/Atheros AP).

USAGE:
    python3 parse_apstats.py <input.txt> <output.csv>

EXAMPLE:
    python3 parse_apstats.py apstats_dataset.txt apstats_clean.csv

KEY DESIGN DECISIONS:
    - Only extracts Node Level Stats (per-client) — skips AP/Radio/VAP blocks
    - Skips ghost node 00:00:00:00:00:00 (driver phantom entry, always all-zeros)
    - Handles all MCS families: OFDM, CCK, HT, VHT, HE with TX/RX direction prefix
    - Handles TX OFDMA RU allocation stats
    - Handles stbc/chainmask lines (parsed separately)
    - Attaches timestamp + snapshot_id to every row
    - Output: one row per client per snapshot = time-series format for ML
"""

import re
import csv
import sys
from collections import OrderedDict

# ── Regex patterns ─────────────────────────────────

RE_SNAP_BEGIN = re.compile(r'###SNAP_BEGIN ts=(\S+) n=(\d+)###')
RE_SNAP_END   = re.compile(r'###SNAP_END###')

RE_NODE = re.compile(
    r'Node Level Stats:\s+([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})\s+\(under VAP\s+(\S+)\)',
    re.IGNORECASE
)

RE_LEAVE_NODE = re.compile(
    r'^(AP Level Stats|WLAN Stats|Radio Level Stats|VAP Level Stats)\s*[:\s]',
    re.IGNORECASE
)

RE_NODE_SUBSECTION = re.compile(
    r'^(Tx Data Packets per AC|Rx Data Packets per AC|Excessive retries per AC'
    r'|TX MCS Stats|RX MCS Stats|TX OFDMA Stats|lithium_cycle_counts'
    r'|Co-located RNR stats|6GHz stats)\s*[:\s]?$',
    re.IGNORECASE
)

RE_MCS = re.compile(
    r'^((?:OFDM|CCK|HT MCS|VHT MCS|HE MCS|RU_)[^:]+?)\s*:\s*(\d+)\s*$'
)

# 🔥 UPDATED — handles <DISABLED>
RE_KV = re.compile(r'^(.+?)\s*=\s*(\d+|<DISABLED>)\s*$')

RE_STBC = re.compile(r'^stbc\s+tx\((\d+)\)\s+rx\((\d+)\)')
RE_CHAIN = re.compile(r'^chainmask\s+\(NSS\)\s+tx\((\d+)\)\s+rx\((\d+)\)')

GHOST_MAC = '00:00:00:00:00:00'


def clean_key(raw):
    k = raw.strip()
    k = re.sub(r'[():/]', '_', k)
    k = re.sub(r'[^a-zA-Z0-9_]', '_', k)
    k = re.sub(r'_+', '_', k)
    return k.strip('_').lower()


def mcs_col(label, direction):
    lb = label.strip()
    is_ru = lb.upper().startswith('RU_')
    lb = re.sub(r'[^a-zA-Z0-9 ]', ' ', lb)
    lb = lb.lower().strip()
    lb = re.sub(r'\s+', '_', lb)
    if is_ru:
        return f"{direction}_ofdma_{lb}"
    return f"{direction}_{lb}"


def parse_file(filepath):
    records = []

    with open(filepath, 'r', errors='replace') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    snap_ts = None
    snap_id = None
    in_node = False
    current_mac = None
    current_vap = None
    current = OrderedDict()
    mcs_dir = 'tx'

    def flush():
        nonlocal current, current_mac, current_vap, in_node, mcs_dir
        if snap_ts and current_mac and current_mac != GHOST_MAC and len(current) > 3:
            rec = OrderedDict()
            rec['timestamp'] = snap_ts
            rec['snapshot_id'] = snap_id
            rec['client_mac'] = current_mac
            rec['vap'] = current_vap
            rec.update(current)
            records.append(rec)
        current = OrderedDict()
        current_mac = None
        current_vap = None
        in_node = False
        mcs_dir = 'tx'

    for line in lines:
        s = line.strip()

        # 🔥 Skip error lines
        if s.lower().startswith("error received"):
            continue

        # 🔥 Skip APSTATS markers safely
        if "APSTATS_BEGIN" in s:
            continue

        # SNAP BEGIN
        m = RE_SNAP_BEGIN.match(s)
        if m:
            flush()
            snap_ts = m.group(1)
            snap_id = int(m.group(2))
            continue

        # SNAP END
        if RE_SNAP_END.match(s):
            flush()
            continue

        if not s:
            continue

        # Leaving node
        if RE_LEAVE_NODE.match(s):
            flush()
            continue

        # Node start
        m = RE_NODE.match(s)
        if m:
            flush()
            current_mac = m.group(1).lower()
            current_vap = m.group(2)
            in_node = True
            mcs_dir = 'tx'
            continue

        if not in_node:
            continue

        # Subsection headers
        if RE_NODE_SUBSECTION.match(s):
            sl = s.lower()
            if sl.startswith('tx mcs'):
                mcs_dir = 'tx'
            elif sl.startswith('rx mcs'):
                mcs_dir = 'rx'
            elif sl.startswith('tx ofdma'):
                mcs_dir = 'tx'
            continue

        # 🔥 FIXED — only skip AC breakdown lines
        if line.startswith(' ') and any(x in s.lower() for x in ['best effort', 'background', 'video', 'voice']):
            continue

        # STBC
        m = RE_STBC.match(s)
        if m:
            current['stbc_tx'] = int(m.group(1))
            current['stbc_rx'] = int(m.group(2))
            continue

        # NSS
        m = RE_CHAIN.match(s)
        if m:
            current['nss_tx'] = int(m.group(1))
            current['nss_rx'] = int(m.group(2))
            continue

        # MCS
        m = RE_MCS.match(s)
        if m:
            col = mcs_col(m.group(1), mcs_dir)
            current[col] = int(m.group(2))
            continue

        # KV pairs
        m = RE_KV.match(s)
        if m:
            raw_key = m.group(1)

            if 'lithium' in raw_key.lower() or 'cycle_cnt' in raw_key.lower():
                continue

            col = clean_key(raw_key)

            val = m.group(2)
            if val == "<DISABLED>":
                val = 0
            else:
                val = int(val)

            current[col] = val
            continue

    flush()
    return records


def all_columns(records):
    meta = ['timestamp', 'snapshot_id', 'client_mac', 'vap']
    seen = set(meta)
    extra = []
    for rec in records:
        for k in rec:
            if k not in seen:
                seen.add(k)
                extra.append(k)
    return meta + extra


def write_csv(records, columns, outpath):
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for rec in records:
            writer.writerow({col: rec.get(col, 0) for col in columns})


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_apstats.py <input.txt> <output.csv>")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    print(f"[INFO] Reading : {infile}")

    records = parse_file(infile)

    if not records:
        print("[WARN] No records found. Check SNAP markers.")
        sys.exit(1)

    # 🔥 IMPORTANT — sort combined data
    records = sorted(records, key=lambda x: (x['client_mac'], x['timestamp']))

    cols = all_columns(records)
    write_csv(records, cols, outfile)

    from collections import Counter
    clients = Counter(r['client_mac'] for r in records)
    snaps = len(set(r['snapshot_id'] for r in records))

    print(f"[INFO] Snapshots   : {snaps}")
    print(f"[INFO] Clients     : {len(clients)}")
    print(f"[INFO] Total rows  : {len(records)}")
    print(f"[INFO] CSV columns : {len(cols)}")
    print(f"[INFO] Output      : {outfile}")

    print("\n[INFO] Rows per client:")
    for mac, n in clients.most_common():
        print(f"       {mac}  ->  {n} rows")

    print("\n[INFO] Done. Ready for feature engineering.")


if __name__ == '__main__':
    main()