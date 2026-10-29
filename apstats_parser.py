#!/usr/bin/env python3
"""
apstats_parser.py
==================
Pure Python 3 standard-library parser for ONE `apstats -a -R` text capture.

This reuses the exact regex logic validated in parse_apstats.py earlier in
this project, simplified for a single live snapshot rather than a file
containing many ###SNAP_BEGIN/###SNAP_END### blocks — on the AP there is
no file, just the fresh text output of one subprocess call each cycle.

OUTPUT:
    parse_snapshot(raw_text) -> dict[client_mac] = {field_name: int_value, ...}
    The ghost node 00:00:00:00:00:00 is automatically dropped.
    Only "Node Level Stats" blocks are kept (AP/Radio/VAP level lines skipped).
"""

import re

GHOST_MAC = '00:00:00:00:00:00'

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
RE_MCS = re.compile(r'^((?:OFDM|CCK|HT MCS|VHT MCS|HE MCS|RU_)[^:]+?)\s*:\s*(\d+)\s*$')
RE_KV = re.compile(r'^(.+?)\s*=\s*(\d+)\s*$')
RE_STBC = re.compile(r'^stbc\s+tx\((\d+)\)\s+rx\((\d+)\)')
RE_CHAIN = re.compile(r'^chainmask\s+\(NSS\)\s+tx\((\d+)\)\s+rx\((\d+)\)')


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


def parse_snapshot(raw_text):
    """
    Parse one full apstats -a -R text capture.
    Returns: dict[client_mac_lowercase] = dict[field_name] = int_value
    """
    clients = {}

    current_mac = None
    current = {}
    in_node = False
    mcs_dir = 'tx'

    def flush():
        nonlocal current_mac, current, in_node
        if current_mac and current_mac != GHOST_MAC and len(current) > 2:
            clients[current_mac] = current
        current_mac = None
        current = {}
        in_node = False

    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue

        if RE_LEAVE_NODE.match(s):
            flush()
            continue

        m = RE_NODE.match(s)
        if m:
            flush()
            current_mac = m.group(1).lower()
            current = {'vap': m.group(2)}
            in_node = True
            mcs_dir = 'tx'
            continue

        if not in_node:
            continue

        if RE_NODE_SUBSECTION.match(s):
            sl = s.lower()
            if sl.startswith('tx mcs'):
                mcs_dir = 'tx'
            elif sl.startswith('rx mcs'):
                mcs_dir = 'rx'
            elif sl.startswith('tx ofdma'):
                mcs_dir = 'tx'
            continue

        if line and line[0] == ' ' and s:
            continue   # indented AC sub-items, skip

        m = RE_STBC.match(s)
        if m:
            current['stbc_tx'] = int(m.group(1))
            current['stbc_rx'] = int(m.group(2))
            continue

        m = RE_CHAIN.match(s)
        if m:
            current['nss_tx'] = int(m.group(1))
            current['nss_rx'] = int(m.group(2))
            continue

        m = RE_MCS.match(s)
        if m:
            col = mcs_col(m.group(1), mcs_dir)
            current[col] = int(m.group(2))
            continue

        m = RE_KV.match(s)
        if m:
            raw_key = m.group(1)
            if 'lithium' in raw_key.lower() or 'cycle_cnt' in raw_key.lower():
                continue
            current[clean_key(raw_key)] = int(m.group(2))
            continue

    flush()
    return clients


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample_apstats_raw.txt'
    with open(path) as f:
        text = f.read()
    clients = parse_snapshot(text)
    print(f"Parsed {len(clients)} client(s) (ghost node auto-dropped if present):")
    for mac, fields in clients.items():
        print(f"\n  {mac}  ({len(fields)} fields)")
        for k in ('tx_data_packets', 'rx_data_packets', 'rx_rssi',
                  'avg_ppdu_tx_rate_kbps', 'avg_ppdu_rx_rate_kbps',
                  'tx_he_mcs_11_1024qam_5_6', 'rx_he_mcs_11_1024qam_5_6'):
            if k in fields:
                print(f"    {k:30s} = {fields[k]}")
