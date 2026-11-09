# # # plot_ae_timeline_auto.py

# # import pandas as pd
# # import matplotlib.pyplot as plt

# # REPORT_CSV = "results/ae_auto_ml_ready_data_ex_RStorm_report.csv"
# # SCORED_CSV = "results/ae_auto_ml_ready_data_ex_RStorm_scored.csv"

# # # --------------------------------------------------
# # # Load report
# # # --------------------------------------------------

# # report = pd.read_csv(REPORT_CSV)

# # report = report.sort_values(
# #     "anomaly_rate_active_pct",
# #     ascending=False
# # )

# # top_client = report.iloc[0]["client_mac"]

# # ae_threshold = report.iloc[0]["ae_threshold_active"]

# # print("\nMost anomalous client detected:")
# # print(top_client)

# # # --------------------------------------------------
# # # Load scored data
# # # --------------------------------------------------

# # df = pd.read_csv(SCORED_CSV)

# # df["timestamp"] = pd.to_datetime(df["timestamp"])

# # df = df[df["client_mac"] == top_client].copy()

# # boundary = pd.to_datetime(
# #     df["anomaly_start_detected"].iloc[0]
# # )

# # # --------------------------------------------------
# # # Plot
# # # --------------------------------------------------

# # plt.figure(figsize=(14, 6))

# # plt.plot(
# #     df["timestamp"],
# #     df["anomaly_score"],
# #     linewidth=2,
# #     label="Reconstruction Error (MSE)"
# # )

# # flagged = df[df["anomaly_flag"] == 1]

# # plt.scatter(
# #     flagged["timestamp"],
# #     flagged["anomaly_score"],
# #     marker="x",
# #     s=50,
# #     label="Flagged Anomaly"
# # )

# # plt.axhline(
# #     y=ae_threshold,
# #     linestyle="--",
# #     label=f"AE Threshold ({ae_threshold:.3f})"
# # )

# # plt.axvline(
# #     x=boundary,
# #     linestyle="--",
# #     label="Detected Boundary"
# # )

# # plt.xlabel("Timestamp")
# # plt.ylabel("Reconstruction Error (MSE)")

# # plt.title(
# #     f"Autoencoder Detection Timeline\nDetected Client: {top_client}"
# # )

# # plt.legend()

# # plt.tight_layout()

# # plt.savefig(
# #     "plots/ae_reconstruction_timeline_auto.png",
# #     dpi=300
# # )

# # plt.show()

# # print(
# #     "\nSaved: plots/ae_reconstruction_timeline_auto.png"
# # )

# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # ============================================================
# # CONFIGURATION
# # ============================================================

# SCORED_CSV = "results/ae_auto_ml_ready_data_ex_RStorm_scored.csv"
# REPORT_CSV = "results/ae_auto_ml_ready_data_ex_RStorm_report.csv"

# EXPERIMENT_NAME = "retry_storm"

# OUTPUT_DIR = "plots"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ============================================================
# # LOAD DATA
# # ============================================================

# report = pd.read_csv(REPORT_CSV)

# # Most anomalous client automatically
# report = report.sort_values(
#     "anomaly_rate_active_pct",
#     ascending=False
# )

# top_client = report.iloc[0]["client_mac"]

# ae_threshold = report.iloc[0]["ae_threshold_active"]

# print("\n===================================================")
# print(" AUTOENCODER VISUALIZATION")
# print("===================================================")

# print(f"\nDetected Client : {top_client}")

# df = pd.read_csv(SCORED_CSV)

# df["timestamp"] = pd.to_datetime(df["timestamp"])

# df = df[df["client_mac"] == top_client].copy()

# boundary = pd.to_datetime(
#     df["anomaly_start_detected"].iloc[0]
# )

# print(f"Boundary        : {boundary}")
# print(f"AE Threshold    : {ae_threshold:.6f}")

# # ============================================================
# # SPLIT NORMAL / ANOMALY PERIOD
# # ============================================================

# normal_df = df[df["timestamp"] < boundary]

# anomaly_df = df[df["timestamp"] >= boundary]

# # ============================================================
# # FIGURE 1
# # AE RECONSTRUCTION ERROR TIMELINE
# # ============================================================

# print("\nGenerating AE timeline...")

# fig, ax = plt.subplots(figsize=(14, 6))

# # ------------------------------------------------------------
# # NORMAL PERIOD
# # ------------------------------------------------------------

# ax.plot(
#     normal_df["timestamp"],
#     normal_df["anomaly_score"],
#     linewidth=2,
#     label="Normal Period"
# )

# # ------------------------------------------------------------
# # ANOMALY PERIOD
# # ------------------------------------------------------------

# ax.plot(
#     anomaly_df["timestamp"],
#     anomaly_df["anomaly_score"],
#     linewidth=2,
#     label="Anomaly Period"
# )

# # ------------------------------------------------------------
# # FLAGGED ROWS
# # ------------------------------------------------------------

# flagged = df[df["anomaly_flag"] == 1]

# ax.scatter(
#     flagged["timestamp"],
#     flagged["anomaly_score"],
#     s=80,
#     marker="x",
#     label="Flagged Anomalies"
# )

# # ------------------------------------------------------------
# # THRESHOLD
# # ------------------------------------------------------------

# ax.axhline(
#     ae_threshold,
#     linestyle="--",
#     linewidth=2,
#     label=f"AE Threshold ({ae_threshold:.3f})"
# )

# # ------------------------------------------------------------
# # BOUNDARY
# # ------------------------------------------------------------

# ax.axvline(
#     boundary,
#     linestyle="--",
#     linewidth=2,
#     label="Detected Boundary"
# )

# # ------------------------------------------------------------
# # IMPORTANT
# # LOG SCALE FIXES HUGE OUTLIER
# # ------------------------------------------------------------

# ax.set_yscale("log")

# # ------------------------------------------------------------
# # LABELS
# # ------------------------------------------------------------

# ax.set_title(
#     f"Autoencoder Reconstruction Error — {EXPERIMENT_NAME}\n"
#     f"Detected Client: {top_client}",
#     fontsize=16
# )

# ax.set_xlabel(
#     "Timestamp",
#     fontsize=12
# )

# ax.set_ylabel(
#     "Reconstruction Error (MSE)",
#     fontsize=12
# )

# ax.legend()

# ax.grid(alpha=0.3)

# plt.xticks(rotation=30)

# plt.tight_layout()

# # ------------------------------------------------------------
# # SAVE
# # ------------------------------------------------------------

# timeline_plot = os.path.join(
#     OUTPUT_DIR,
#     f"ae_timeline_{EXPERIMENT_NAME}.png"
# )

# plt.savefig(
#     timeline_plot,
#     dpi=300,
#     bbox_inches="tight"
# )

# print(f"Saved: {timeline_plot}")

# plt.show()

# # ============================================================
# # FIGURE 2
# # FEATURE EVOLUTION
# # ============================================================

# print("\nGenerating feature evolution plots...")

# candidate_features = [

#     "retry_rate",

#     "tx_packet_rate",

#     "rx_packet_rate",

#     "tx_throughput_mbps",

#     "rx_throughput_mbps",

#     "rssi",

#     "tx_success_rate",
# ]

# features = [
#     f for f in candidate_features
#     if f in df.columns
# ]

# fig, axes = plt.subplots(
#     len(features),
#     1,
#     figsize=(14, 3 * len(features)),
#     sharex=True
# )

# if len(features) == 1:
#     axes = [axes]

# for ax, feature in zip(axes, features):

#     ax.plot(
#         df["timestamp"],
#         df[feature],
#         linewidth=2
#     )

#     ax.axvline(
#         boundary,
#         linestyle="--",
#         linewidth=2
#     )

#     ax.set_ylabel(feature)

#     ax.grid(alpha=0.3)

# axes[-1].set_xlabel("Timestamp")

# fig.suptitle(
#     f"Feature Evolution — {EXPERIMENT_NAME}",
#     fontsize=16
# )

# plt.xticks(rotation=30)

# plt.tight_layout()

# feature_plot = os.path.join(
#     OUTPUT_DIR,
#     f"ae_features_{EXPERIMENT_NAME}.png"
# )

# plt.savefig(
#     feature_plot,
#     dpi=300,
#     bbox_inches="tight"
# )

# print(f"Saved: {feature_plot}")

# plt.show()

# # ============================================================
# # SUMMARY
# # ============================================================

# print("\n===================================================")
# print(" VISUALIZATION COMPLETE")
# print("===================================================")

# print(f"\nDetected Client : {top_client}")
# print(f"Boundary        : {boundary}")

# print("\nGenerated plots:")

# print(f"\n1. AE Timeline")
# print(f"   {timeline_plot}")

# print(f"\n2. Feature Evolution")
# print(f"   {feature_plot}")

"""
visualize_experiment.py
========================
Simple, single-chart visualization of one experiment's results.
Designed to be explainable to anyone — no jargon, no complex panels.
 
The chart shows one thing clearly:
  "Here is what the model saw, moment by moment.
   This is when normal behaviour ended and the anomaly began.
   These are the devices that looked suspicious."
 
USAGE:
    python visualize_experiment.py --scored results/ae_auto_<...>_scored.csv \
                                   --report results/ae_auto_<...>_report.csv \
                                   --name "Traffic Burst"
 
    # Or let it auto-detect the most recent files in results/:
    python visualize_experiment.py --name "Traffic Burst"
 
OUTPUT:
    plots/<experiment_name>_what_happened.png
"""
 
# import os
# import glob
# import argparse
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches

# warnings.filterwarnings('ignore')
# os.makedirs('plots', exist_ok=True)

# COLOR_ANOMALY   = '#e05c5c'   # red  — above threshold
# COLOR_NORMAL    = '#3ecf8e'   # green — below threshold
# COLOR_THRESHOLD = '#f6ad55'   # orange dashed line
# COLOR_OTHER     = '#cccccc'   # grey — undisturbed devices


# def find_latest(pattern):
#     files = glob.glob(os.path.join('results', pattern))
#     return max(files, key=os.path.getmtime) if files else None


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--scored', default=None,
#                         help='Path to *_scored.csv (auto-detects latest if omitted)')
#     parser.add_argument('--report', default=None,
#                         help='Path to *_report.csv (optional — used only for summary box)')
#     parser.add_argument('--name', default='Experiment',
#                         help='Title for the chart, e.g. "Traffic Burst"')
#     args = parser.parse_args()

#     # ── Load scored CSV ───────────────────────────────────────────────────────
#     scored_path = args.scored or find_latest('*_scored.csv')
#     if not scored_path or not os.path.exists(scored_path):
#         print("No scored CSV found. Pass --scored path/to/file.csv")
#         return
#     df = pd.read_csv(scored_path, parse_dates=['timestamp'])
#     df = df.sort_values('timestamp')
#     print(f"[viz] Loaded: {scored_path}  ({len(df):,} rows, "
#           f"{df['client_mac'].nunique()} clients)")

#     # ── Load report CSV (optional) ────────────────────────────────────────────
#     report_df = None
#     report_path = args.report or find_latest('*_report.csv')
#     if report_path and os.path.exists(report_path):
#         report_df = pd.read_csv(report_path)
#         print(f"[viz] Report : {report_path}")

#     # ── Threshold ─────────────────────────────────────────────────────────────
#     threshold = (float(df['ae_threshold_active'].dropna().iloc[0])
#                  if 'ae_threshold_active' in df.columns
#                     and df['ae_threshold_active'].notna().any()
#                  else 0.322690)

#     # ── Active rows only ──────────────────────────────────────────────────────
#     active = df[df['is_active'] == 1].copy() if 'is_active' in df.columns else df.copy()
#     active['flagged'] = active['anomaly_score'] > threshold

#     # ── Auto-detect stressed device: most anomalous readings in scored CSV ────
#     flag_counts = (active.groupby('client_mac')['flagged']
#                    .sum().sort_values(ascending=False))
#     stressed_mac = flag_counts.index[0] if len(flag_counts) else None
#     print(f"[viz] Auto-detected stressed device: {stressed_mac} "
#           f"({int(flag_counts.iloc[0])} anomalous readings)")

#     all_macs = active['client_mac'].unique()

#     # ── Build the chart ───────────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(14, 7))

#     # Log scale: shows 0.05, 0.323, and 234 all meaningfully on one axis
#     ax.set_yscale('log')

#     # ── Undisturbed devices: one thin grey line each ──────────────────────────
#     for mac in all_macs:
#         if mac == stressed_mac:
#             continue
#         dev = active[active['client_mac'] == mac].sort_values('timestamp')
#         if len(dev) < 2:
#             continue
#         ax.plot(dev['timestamp'], dev['anomaly_score'],
#                 color=COLOR_OTHER, linewidth=0.9, alpha=0.6, zorder=1)

#     # ── Stressed device: each reading colored by its own flag status ──────────
#     if stressed_mac:
#         dev = active[active['client_mac'] == stressed_mac].sort_values('timestamp')
#         short = stressed_mac[-5:]

#         # Plot green (normal) readings
#         normal_rows  = dev[~dev['flagged']]
#         anomaly_rows = dev[dev['flagged']]

#         if len(normal_rows):
#             ax.scatter(normal_rows['timestamp'], normal_rows['anomaly_score'],
#                        color=COLOR_NORMAL, s=25, zorder=4, linewidths=0)
#             # Trend line through normal readings
#             if len(normal_rows) >= 3:
#                 trend = normal_rows.set_index('timestamp')['anomaly_score']\
#                             .rolling(3, min_periods=1).median()
#                 ax.plot(trend.index, trend.values,
#                         color=COLOR_NORMAL, linewidth=2.0, zorder=3, alpha=0.8)

#         if len(anomaly_rows):
#             ax.scatter(anomaly_rows['timestamp'], anomaly_rows['anomaly_score'],
#                        color=COLOR_ANOMALY, s=30, zorder=5, linewidths=0)
#             # Trend line through anomaly readings
#             if len(anomaly_rows) >= 3:
#                 trend = anomaly_rows.set_index('timestamp')['anomaly_score']\
#                             .rolling(3, min_periods=1).median()
#                 ax.plot(trend.index, trend.values,
#                         color=COLOR_ANOMALY, linewidth=2.5, zorder=4)

#     # ── Threshold line ────────────────────────────────────────────────────────
#     ax.axhline(threshold, color=COLOR_THRESHOLD, linestyle='--',
#                linewidth=2.0, zorder=2)
#     # Label on the left side of the threshold line
#     xlim = ax.get_xlim()
#     ax.text(ax.get_xlim()[0], threshold * 1.15,
#             f'Alert threshold: {threshold:.3f}\n'
#             f'↑ above this = ANOMALOUS    ↓ below this = normal',
#             fontsize=9, color=COLOR_THRESHOLD, va='bottom',
#             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
#                       edgecolor=COLOR_THRESHOLD, alpha=0.9))

#     # ── Legend ────────────────────────────────────────────────────────────────
#     legend_handles = [
#         mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_NORMAL,
#                       markersize=9, label=f'Device {stressed_mac[-5:] if stressed_mac else "?"} — normal reading (below threshold)'),
#         mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_ANOMALY,
#                       markersize=9, label=f'Device {stressed_mac[-5:] if stressed_mac else "?"} — anomalous reading (above threshold)'),
#         mlines.Line2D([0],[0], color=COLOR_THRESHOLD, linestyle='--', linewidth=2,
#                       label=f'Alert threshold ({threshold:.3f})'),
#         mlines.Line2D([0],[0], color=COLOR_OTHER, linewidth=1.5,
#                       label='Other devices (all normal)'),
#     ]
#     ax.legend(handles=legend_handles, loc='upper left', fontsize=10,
#               framealpha=0.95, edgecolor='#cccccc')

#     # ── Labels ────────────────────────────────────────────────────────────────
#     ax.set_title(
#         f'{args.name} — What the Model Saw\n'
#         f'Green = model says normal. Red = model says anomalous. '
#         f'Log scale so all values are visible.',
#         fontsize=13, fontweight='bold', pad=14)
#     ax.set_xlabel('Time', fontsize=12)
#     ax.set_ylabel('Reconstruction error\n(how unusual this reading looked to the model)',
#                   fontsize=11)
#     ax.tick_params(axis='x', rotation=20)
#     ax.set_facecolor('#fafafa')
#     fig.patch.set_facecolor('white')
#     ax.grid(True, which='both', color='#eeeeee', linewidth=0.5)
#     ax.spines[['top','right']].set_visible(False)

#     # ── Summary box ───────────────────────────────────────────────────────────
#     if stressed_mac:
#         n_anomaly = int(flag_counts.iloc[0])
#         n_total   = int(len(active[active['client_mac'] == stressed_mac]))
#         pct       = 100 * n_anomaly / n_total if n_total else 0

#         summary = (f"Device {stressed_mac[-5:]}:\n"
#                    f"{n_anomaly}/{n_total} readings flagged ({pct:.0f}%)")

#         # Add error_ratio from report if available
#         if report_df is not None:
#             def clean(s):
#                 if not isinstance(s, str): return ''
#                 for e in ['🔴','🟡','🔵','✅','❓','⚠️']:
#                     s = s.replace(e,'')
#                 return s.strip()
#             report_df['_sc'] = report_df['status'].apply(clean)
#             row = report_df[report_df['client_mac'] == stressed_mac]
#             if len(row) and 'error_ratio' in row.columns:
#                 ratio = row['error_ratio'].iloc[0]
#                 if pd.notna(ratio):
#                     summary += f"\n{ratio:.0f}x worse than its own normal"

#         ax.text(0.99, 0.97, summary,
#                 transform=ax.transAxes, fontsize=10, va='top', ha='right',
#                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e7',
#                           edgecolor=COLOR_THRESHOLD, alpha=0.95))

#     plt.tight_layout()
#     safe = args.name.lower().replace(' ', '_')
#     out = f'plots/{safe}_what_happened.png'
#     fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
#     plt.close(fig)
#     print(f"[viz] Saved: {out}")


# if __name__ == '__main__':
#     main()

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

COLOR_ANOMALY   = '#e05c5c'   # red  — above threshold
COLOR_NORMAL    = '#3ecf8e'   # green — below threshold
COLOR_THRESHOLD = '#f6ad55'   # orange dashed line
COLOR_OTHER     = '#cccccc'   # grey — undisturbed devices


def find_latest(pattern):
    files = glob.glob(os.path.join('results', pattern))
    return max(files, key=os.path.getmtime) if files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scored', default=None,
                        help='Path to *_scored.csv (auto-detects latest if omitted)')
    parser.add_argument('--report', default=None,
                        help='Path to *_report.csv (optional — used only for summary box)')
    parser.add_argument('--name', default='Experiment',
                        help='Title for the chart, e.g. "Traffic Burst"')
    args = parser.parse_args()

    # ── Load scored CSV ───────────────────────────────────────────────────────
    scored_path = args.scored or find_latest('*_scored.csv')
    if not scored_path or not os.path.exists(scored_path):
        print("No scored CSV found. Pass --scored path/to/file.csv")
        return
    df = pd.read_csv(scored_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    print(f"[viz] Loaded: {scored_path}  ({len(df):,} rows, "
          f"{df['client_mac'].nunique()} clients)")

    # ── Load report CSV (optional) ────────────────────────────────────────────
    report_df = None
    report_path = args.report or find_latest('*_report.csv')
    if report_path and os.path.exists(report_path):
        report_df = pd.read_csv(report_path)
        print(f"[viz] Report : {report_path}")

    # ── Threshold ─────────────────────────────────────────────────────────────
    threshold = (float(df['ae_threshold_active'].dropna().iloc[0])
                 if 'ae_threshold_active' in df.columns
                    and df['ae_threshold_active'].notna().any()
                 else 0.322690)

    # ── Active rows only ──────────────────────────────────────────────────────
    active = df[df['is_active'] == 1].copy() if 'is_active' in df.columns else df.copy()
    active['flagged'] = active['anomaly_score'] > threshold
    flag_counts = (active.groupby('client_mac')['flagged']
                   .sum().sort_values(ascending=False))
    all_macs = active['client_mac'].unique()

    # ── Auto-detect stressed device(s) from scored CSV ───────────────────────
    # A device is considered "stressed" if it has a SIGNIFICANT fraction of
    # its own active readings flagged — not just one random blip.
    # Threshold: flagged_count >= max(3 rows, 20% of that device's active rows)
    # This catches all genuinely stressed devices while ignoring bystanders.
    active_counts = active.groupby('client_mac')['anomaly_score'].count()
    stressed_macs = [
        mac for mac in flag_counts.index
        if flag_counts[mac] >= max(3, 0.20 * active_counts.get(mac, 1))
    ]
    if not stressed_macs and len(flag_counts):
        # Fallback: at least highlight the most-flagged device
        stressed_macs = [flag_counts.index[0]]

    print(f"[viz] Stressed device(s) found: {len(stressed_macs)}")
    for mac in stressed_macs:
        print(f"       {mac}  ({int(flag_counts.get(mac, 0))} anomalous readings)")

    # One distinct color per stressed device — readable up to ~4 devices
    STRESSED_COLORS = ['#e05c5c', '#f6a623', '#7c5cbf', '#2196f3']

    # ── Build the chart ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    # Log scale: shows 0.05, 0.323, and 234 all meaningfully on one axis
    ax.set_yscale('log')

    # ── Undisturbed devices: thin grey lines ──────────────────────────────────
    for mac in all_macs:
        if mac in stressed_macs:
            continue
        dev = active[active['client_mac'] == mac].sort_values('timestamp')
        if len(dev) < 2:
            continue
        ax.plot(dev['timestamp'], dev['anomaly_score'],
                color=COLOR_OTHER, linewidth=0.9, alpha=0.6, zorder=1)

    # ── Stressed device(s): each reading colored green/red, distinct per device
    legend_stressed = []
    for idx, mac in enumerate(stressed_macs):
        color = STRESSED_COLORS[idx % len(STRESSED_COLORS)]
        short = mac[-5:]
        dev = active[active['client_mac'] == mac].sort_values('timestamp')

        normal_rows  = dev[~dev['flagged']]
        anomaly_rows = dev[dev['flagged']]

        if len(normal_rows):
            ax.scatter(normal_rows['timestamp'], normal_rows['anomaly_score'],
                       color=COLOR_NORMAL, s=25, zorder=4, linewidths=0)
            if len(normal_rows) >= 3:
                trend = (normal_rows.set_index('timestamp')['anomaly_score']
                         .rolling(3, min_periods=1).median())
                ax.plot(trend.index, trend.values,
                        color=COLOR_NORMAL, linewidth=2.0, zorder=3, alpha=0.8)

        if len(anomaly_rows):
            ax.scatter(anomaly_rows['timestamp'], anomaly_rows['anomaly_score'],
                       color=color, s=30, zorder=5, linewidths=0)
            if len(anomaly_rows) >= 3:
                trend = (anomaly_rows.set_index('timestamp')['anomaly_score']
                         .rolling(3, min_periods=1).median())
                ax.plot(trend.index, trend.values,
                        color=color, linewidth=2.5, zorder=4)

        legend_stressed.append((short, color,
                                 int(flag_counts.get(mac, 0)),
                                 int(len(dev))))

    # ── Threshold line ────────────────────────────────────────────────────────
    ax.axhline(threshold, color=COLOR_THRESHOLD, linestyle='--',
               linewidth=2.0, zorder=2)
    # Label on the left side of the threshold line
    xlim = ax.get_xlim()
    ax.text(ax.get_xlim()[0], threshold * 1.15,
            f'Alert threshold: {threshold:.3f}\n'
            f'↑ above this = ANOMALOUS    ↓ below this = normal',
            fontsize=9, color=COLOR_THRESHOLD, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=COLOR_THRESHOLD, alpha=0.9))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_NORMAL,
                      markersize=9, label='Normal reading (below threshold)'),
    ]
    for short, color, n_flag, n_total in legend_stressed:
        pct = 100*n_flag/n_total if n_total else 0
        legend_handles.append(
            mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor=color,
                          markersize=9,
                          label=f'Device {short} — anomalous ({n_flag}/{n_total} = {pct:.0f}% of readings)')
        )
    legend_handles += [
        mlines.Line2D([0],[0], color=COLOR_THRESHOLD, linestyle='--', linewidth=2,
                      label=f'Alert threshold ({threshold:.3f})'),
        mlines.Line2D([0],[0], color=COLOR_OTHER, linewidth=1.5,
                      label='Other devices (undisturbed)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9,
              framealpha=0.95, edgecolor='#cccccc')

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.set_title(
        f'{args.name} — What the Model Saw\n'
        f'Green = model says normal. Red = model says anomalous. '
        f'Log scale so all values are visible.',
        fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Reconstruction error\n(how unusual this reading looked to the model)',
                  fontsize=11)
    ax.tick_params(axis='x', rotation=20)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    ax.grid(True, which='both', color='#eeeeee', linewidth=0.5)
    ax.spines[['top','right']].set_visible(False)

    # ── Summary box ───────────────────────────────────────────────────────────
    if legend_stressed:
        lines = []
        for short, color, n_flag, n_total in legend_stressed:
            pct = 100*n_flag/n_total if n_total else 0
            line = f"Device {short}: {n_flag}/{n_total} flagged ({pct:.0f}%)"
            if report_df is not None:
                def clean(s):
                    if not isinstance(s, str): return ''
                    for e in ['🔴','🟡','🔵','✅','❓','⚠️']:
                        s = s.replace(e,'')
                    return s.strip()
                report_df['_sc'] = report_df['status'].apply(clean)
                full_mac = next((m for m in all_macs if m.endswith(short)), None)
                if full_mac is not None:
                    row = report_df[report_df['client_mac'] == full_mac]
                    if len(row) and 'error_ratio' in row.columns:
                        ratio = row['error_ratio'].iloc[0]
                        if pd.notna(ratio) and ratio > 1:
                            line += f"  ({ratio:.0f}x worse than normal)"
            lines.append(line)
        ax.text(0.99, 0.97, '\n'.join(lines),
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e7',
                          edgecolor=COLOR_THRESHOLD, alpha=0.95))

    plt.tight_layout()
    safe = args.name.lower().replace(' ', '_')
    out = f'plots/{safe}_what_happened.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[viz] Saved: {out}")


if __name__ == '__main__':
    main()
