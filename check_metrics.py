import pandas as pd

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "ml_ready_data_ex_signalD.csv"

TARGET_MAC = "a8:74:1d:47:e6:3e"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE, parse_dates=['timestamp'])

print("\n===================================================")
print(" SIGNAL DEGRADATION INSPECTION")
print("===================================================")

print(f"\nTotal rows        : {len(df):,}")
print(f"Unique clients    : {df['client_mac'].nunique()}")

print("\nRows per client:")
print(df['client_mac'].value_counts())

# ============================================================
# FILTER TARGET DEVICE
# ============================================================

target = df[df['client_mac'] == TARGET_MAC].copy()

print("\n===================================================")
print(" TARGET DEVICE")
print("===================================================")

print(f"\nTarget MAC        : {TARGET_MAC}")
print(f"Rows for target   : {len(target)}")

if len(target) == 0:
    print("\n[ERROR] Target MAC not found.")
    exit()

print(f"\nTime range:")
print(f"  Start : {target['timestamp'].min()}")
print(f"  End   : {target['timestamp'].max()}")

# ============================================================
# IMPORTANT FEATURES
# ============================================================

features = [

    'timestamp',

    'rssi',

    'retry_rate',

    'tx_success_rate',

    'tx_packet_rate',
    'rx_packet_rate',

    'tx_throughput_mbps',
    'rx_throughput_mbps',

    'low_mcs_ratio',
    'mid_mcs_ratio',
    'high_mcs_ratio',

    'mcs_entropy'
]

available = [c for c in features if c in target.columns]

print("\n===================================================")
print(" SAMPLE ROWS")
print("===================================================")

print(target[available].head(40))

# ============================================================
# STATISTICS
# ============================================================

stats_cols = [

    'rssi',

    'retry_rate',

    'tx_success_rate',

    'tx_packet_rate',
    'rx_packet_rate',

    'tx_throughput_mbps',
    'rx_throughput_mbps',

    'low_mcs_ratio',
    'mid_mcs_ratio',
    'high_mcs_ratio',

    'mcs_entropy'
]

stats_cols = [c for c in stats_cols if c in target.columns]

print("\n===================================================")
print(" STATISTICS")
print("===================================================")

print(target[stats_cols].describe())

# ============================================================
# RSSI ANALYSIS
# ============================================================

print("\n===================================================")
print(" RSSI ANALYSIS")
print("===================================================")

print(f"\nRSSI Range:")
print(f"  Min RSSI : {target['rssi'].min():.2f}")
print(f"  Max RSSI : {target['rssi'].max():.2f}")
print(f"  Avg RSSI : {target['rssi'].mean():.2f}")

# Weak signal rows
weak_signal = target[target['rssi'] >= 55]

print(f"\nRows with degraded RSSI (>=55): {len(weak_signal)}")

# ============================================================
# MCS ANALYSIS
# ============================================================

print("\n===================================================")
print(" MCS ANALYSIS")
print("===================================================")

if 'low_mcs_ratio' in target.columns:

    print(f"\nAverage low_mcs_ratio : {target['low_mcs_ratio'].mean():.4f}")

if 'high_mcs_ratio' in target.columns:

    print(f"Average high_mcs_ratio: {target['high_mcs_ratio'].mean():.4f}")

if 'mcs_entropy' in target.columns:

    print(f"Average mcs_entropy   : {target['mcs_entropy'].mean():.4f}")

# ============================================================
# THROUGHPUT ANALYSIS
# ============================================================

print("\n===================================================")
print(" THROUGHPUT ANALYSIS")
print("===================================================")

if 'tx_throughput_mbps' in target.columns:

    print(f"\nTX throughput avg : {target['tx_throughput_mbps'].mean():.2f}")

if 'rx_throughput_mbps' in target.columns:

    print(f"RX throughput avg : {target['rx_throughput_mbps'].mean():.2f}")

# ============================================================
# TARGET VS OTHER CLIENTS
# ============================================================

print("\n===================================================")
print(" TARGET vs OTHER CLIENTS")
print("===================================================")

others = df[df['client_mac'] != TARGET_MAC]

compare_cols = [

    'rssi',

    'retry_rate',

    'low_mcs_ratio',

    'high_mcs_ratio',

    'tx_throughput_mbps',

    'rx_throughput_mbps'
]

for col in compare_cols:

    if col not in df.columns:
        continue

    target_mean = target[col].mean()
    others_mean = others[col].mean()

    print(f"\n{col}")
    print(f"  Target avg : {target_mean:.4f}")
    print(f"  Others avg : {others_mean:.4f}")

# ============================================================
# MOST DEGRADED ROWS
# ============================================================

print("\n===================================================")
print(" MOST DEGRADED ROWS")
print("===================================================")

severity = target.copy()

severity['degradation_score'] = (

    severity['rssi']

    + severity['low_mcs_ratio'] * 10

    + severity['retry_rate'] * 10

    - severity['high_mcs_ratio'] * 5
)

severity = severity.sort_values(
    'degradation_score',
    ascending=False
)

show_cols = [

    'timestamp',

    'rssi',

    'retry_rate',

    'low_mcs_ratio',

    'mid_mcs_ratio',

    'high_mcs_ratio',

    'mcs_entropy',

    'tx_throughput_mbps',

    'rx_throughput_mbps',

    'degradation_score'
]

show_cols = [c for c in show_cols if c in severity.columns]

print(severity[show_cols].head(20))

# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n===================================================")
print(" INTERPRETATION")
print("===================================================")

print("""
GOOD SIGNAL DEGRADATION EXPERIMENT:
  ✓ Target RSSI consistently weaker than other clients
  ✓ Device remains continuously connected
  ✓ Throughput reduction visible
  ✓ MCS distribution shifts toward lower MCS
  ✓ Retry behavior slightly elevated
  ✓ Active rows remain high

WEAK EXPERIMENT:
  ✗ RSSI similar to other devices
  ✗ Device disappears from telemetry
  ✗ Too many disconnects
  ✗ No MCS behavior change
  ✗ No throughput impact
""")

print("\n===================================================")
print(" INSPECTION COMPLETE")
print("===================================================")