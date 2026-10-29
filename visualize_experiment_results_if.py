import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "results/real_signal_degradation_evaluation.csv"

EXPERIMENT_NAME = "signal_degradation"

OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_FILE)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================================================
# BASIC INFO
# ============================================================

print("\n===================================================")
print(" EXPERIMENT VISUALIZATION")
print("===================================================")

print(f"\nCSV Loaded       : {CSV_FILE}")
print(f"Rows             : {len(df):,}")

# ============================================================
# DETECT BOUNDARY
# ============================================================

boundary = pd.to_datetime(df['detected_boundary'].iloc[0])

print(f"Boundary         : {boundary}")

# ============================================================
# FIGURE 1 — ANOMALY SCORE TIMELINE
# ============================================================

print("\nGenerating IF score timeline...")

fig, ax = plt.subplots(figsize=(14, 6))

# ------------------------------------------------------------
# NORMAL REGION
# ------------------------------------------------------------

normal_df = df[df['period'] == 'normal']

ax.plot(
    normal_df['timestamp'],
    normal_df['anomaly_score'],
    label='Normal Period',
    linewidth=2
)

# ------------------------------------------------------------
# ANOMALY REGION
# ------------------------------------------------------------

anomaly_df = df[df['period'] == 'anomaly']

ax.plot(
    anomaly_df['timestamp'],
    anomaly_df['anomaly_score'],
    label='Anomaly Period',
    linewidth=2
)

# ------------------------------------------------------------
# FLAGGED ANOMALIES
# ------------------------------------------------------------

flagged = df[df['anomaly_flag'] == 1]

ax.scatter(
    flagged['timestamp'],
    flagged['anomaly_score'],
    s=80,
    marker='x',
    label='Flagged Anomalies'
)

# ------------------------------------------------------------
# BOUNDARY LINE
# ------------------------------------------------------------

ax.axvline(
    boundary,
    linestyle='--',
    linewidth=2,
    label='Detected Boundary'
)

# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

ax.set_title(
    f"Isolation Forest Scores — {EXPERIMENT_NAME}",
    fontsize=16
)

ax.set_xlabel("Timestamp", fontsize=12)

ax.set_ylabel("Anomaly Score", fontsize=12)

ax.legend()

ax.grid(alpha=0.3)

plt.xticks(rotation=30)

plt.tight_layout()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

score_plot = os.path.join(
    OUTPUT_DIR,
    f"if_scores_{EXPERIMENT_NAME}.png"
)

plt.savefig(
    score_plot,
    dpi=300,
    bbox_inches='tight'
)

print(f"Saved: {score_plot}")

plt.show()

# ============================================================
# FIGURE 2 — FEATURE EVOLUTION
# ============================================================

print("\nGenerating feature evolution plots...")

# ------------------------------------------------------------
# FEATURES TO PLOT
# ------------------------------------------------------------

candidate_features = [

    'retry_rate',

    'tx_packet_rate',

    'rx_packet_rate',

    'tx_throughput_mbps',

    'rx_throughput_mbps',

    'rssi',

    'tx_success_rate',
]

features = [
    f for f in candidate_features
    if f in df.columns
]

# ------------------------------------------------------------
# CREATE FIGURE
# ------------------------------------------------------------

fig, axes = plt.subplots(
    len(features),
    1,
    figsize=(14, 3 * len(features)),
    sharex=True
)

# If only one feature exists
if len(features) == 1:
    axes = [axes]

# ------------------------------------------------------------
# PLOT EACH FEATURE
# ------------------------------------------------------------

for ax, feature in zip(axes, features):

    ax.plot(
        df['timestamp'],
        df[feature],
        linewidth=2
    )

    # Boundary line
    ax.axvline(
        boundary,
        linestyle='--',
        linewidth=2
    )

    ax.set_ylabel(feature)

    ax.grid(alpha=0.3)

# ------------------------------------------------------------
# FINAL LABELS
# ------------------------------------------------------------

axes[-1].set_xlabel("Timestamp")

fig.suptitle(
    f"Feature Evolution — {EXPERIMENT_NAME}",
    fontsize=16
)

plt.xticks(rotation=30)

plt.tight_layout()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

feature_plot = os.path.join(
    OUTPUT_DIR,
    f"features_{EXPERIMENT_NAME}.png"
)

plt.savefig(
    feature_plot,
    dpi=300,
    bbox_inches='tight'
)

print(f"Saved: {feature_plot}")

plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\n===================================================")
print(" VISUALIZATION COMPLETE")
print("===================================================")

print("\nGenerated plots:")

print(f"\n1. IF Score Timeline")
print(f"   {score_plot}")

print(f"\n2. Feature Evolution")
print(f"   {feature_plot}")