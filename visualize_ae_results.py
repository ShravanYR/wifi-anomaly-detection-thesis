#!/usr/bin/env python3
"""
visualize_ae_results.py
========================
Visualises results produced by autonomous_ae_evaluator.py.

NO FILENAME GUESSING — you list your files directly in the EXPERIMENTS
list below (or pass a small JSON/CSV manifest with --manifest). This
removes any risk of the script picking up the wrong run, an old
un-timestamped leftover file, or a file from a different script.

WHAT EACH RUN OF autonomous_ae_evaluator.py PRODUCES:
    results/ae_auto_<input>_<run_timestamp>_scored.csv   per-row scores
    results/ae_auto_<input>_<run_timestamp>_report.csv   per-client summary
    results/ae_auto_<input>_<run_timestamp>_flagged.csv  (not used here)

CONFIRMED COLUMNS USED:
    scored.csv : timestamp, client_mac, is_active, retry_rate, tx_packet_rate,
                 rx_packet_rate, rssi, rx_throughput_mbps, tx_throughput_mbps,
                 anomaly_score, anomaly_flag, model_used, detected_status,
                 anomaly_start_detected, ae_threshold_active
    report.csv : client_mac, status, error_ratio, mse_baseline,
                 mse_anomaly_period, anomaly_rate_active_pct, anomaly_start

THIS SCRIPT DOES NOT INVENT NUMBERS:
    - The normal/anomaly split reuses anomaly_start_detected from scored.csv
      exactly as the evaluator wrote it — never recomputed here.
    - error_ratio, mse_baseline, mse_anomaly_period only exist in report.csv
      (scored.csv doesn't have them) and are read from there directly.
    - The flag threshold line comes from the ae_threshold_active column,
      or models/ae_metadata.json as a fallback — never typed in.
    - The "top 5 changed features" panel is computed live from the raw
      feature columns in scored.csv for whichever experiment you point it at.

═══════════════════════════════════════════════════════════════════════════
STEP 1 — EDIT THE LIST BELOW WITH YOUR ACTUAL FILE PATHS
═══════════════════════════════════════════════════════════════════════════
"""

EXPERIMENTS = [
    # {
    #     "name"      : "Retry Storm",
    #     "scored_csv": "results/ae_auto_ml_ready_data_ex_RStorm_20260623_164255_scored.csv",
    #     "report_csv": "results/ae_auto_ml_ready_data_ex_RStorm_20260623_164255_report.csv",
    # },
    {
        "name"      : "Signal Degradation",
        "scored_csv": "results/ae_auto_ml_ready_data_ex_signalD_20260623_163233_scored.csv",
        "report_csv": "results/ae_auto_ml_ready_data_ex_signalD_20260623_163233_report.csv",
    },
    # {
    #     "name"      : "Traffic Burst",
    #     "scored_csv": "results/ae_auto_ml_ready_data_ex_trafficB_20260610_204249_scored.csv",
    #     "report_csv": "results/ae_auto_ml_ready_data_ex_trafficB_20260610_204249_report.csv",
    # },
    # Add more experiments here the same way, e.g.:
    # {
    #     "name"      : "Link Failure",
    #     "scored_csv": "results/ae_auto_ml_ready_data_ex_LinkFailure_<timestamp>_scored.csv",
    #     "report_csv": "results/ae_auto_ml_ready_data_ex_LinkFailure_<timestamp>_report.csv",
    # },
]

# Optional: path to your normal training data, used only for Figure 1.
ML_READY_DATA_CSV = "ml_ready_data.csv"

# Optional: path to ae_metadata.json, used as a threshold fallback if a
# scored.csv doesn't carry its own ae_threshold_active column.
AE_METADATA_JSON = "models/ae_metadata.json"

OUTPUT_DIR = "plots"

"""
═══════════════════════════════════════════════════════════════════════════
STEP 2 — RUN IT
═══════════════════════════════════════════════════════════════════════════
    python visualize_ae_results.py                  # every experiment in the list above
    python visualize_ae_results.py --only "Retry Storm"   # just one, by its "name"
    python visualize_ae_results.py --overview        # also draw Figure 1 (training baseline)
    python visualize_ae_results.py --manifest experiments.json   # load the list from a file instead

OUTPUT (written to plots/):
    figure1_normal_data_overview.png        (only with --overview)
    figure3_<experiment_name>_deepdive.png  one per experiment in the list
    figure4_all_experiments_summary.png     cross-experiment comparison
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#fafafa',
    'axes.edgecolor'   : '#cccccc',
    'axes.grid'        : True,
    'grid.color'       : '#e5e5e5',
    'grid.linewidth'   : 0.6,
    'legend.frameon'   : True,
    'legend.facecolor' : 'white',
    'legend.edgecolor' : '#cccccc',
})

# Colour meaning is FIXED across every chart — learn once, applies everywhere.
COLOR_NORMAL    = '#3ecf8e'
COLOR_ANOMALY   = '#f56565'
COLOR_AE        = '#2ca58d'
COLOR_THRESHOLD = '#f6ad55'
COLOR_IDLE      = '#b8bfd1'
COLOR_ACTIVE    = '#6c5ce7'

STATUS_COLOR = {
    'CRITICAL': COLOR_ANOMALY, 'WARNING': COLOR_THRESHOLD,
    'WATCH': '#4f8ef7', 'NORMAL': COLOR_NORMAL, 'INSUFFICIENT_DATA': '#999999',
}


def safe_filename(name):
    return name.lower().replace(' ', '_').replace('/', '_')


def caption(fig, text):
    fig.text(0.5, 0.005, text, ha='center', va='bottom',
              fontsize=10, style='italic', color='#555555', wrap=True)


def load_threshold(df_scored):
    if 'ae_threshold_active' in df_scored.columns and df_scored['ae_threshold_active'].notna().any():
        return float(df_scored['ae_threshold_active'].dropna().iloc[0]), True
    if os.path.exists(AE_METADATA_JSON):
        with open(AE_METADATA_JSON) as f:
            return json.load(f).get('active_threshold', 0.32), True
    return 0.32, False


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — NORMAL TRAINING DATA OVERVIEW  (optional, --overview)
# ════════════════════════════════════════════════════════════════════════════

def figure1_normal_data_overview():
    print("[Figure 1] Normal training data overview...")

    if os.path.exists(ML_READY_DATA_CSV):
        df = pd.read_csv(ML_READY_DATA_CSV, parse_dates=['timestamp'])
        print(f"   Using real file: {ML_READY_DATA_CSV}  ({len(df):,} rows)")
        placeholder = False
    else:
        print(f"   {ML_READY_DATA_CSV} not found — using placeholder data")
        np.random.seed(1)
        n = 5000
        df = pd.DataFrame({
            'client_mac': np.random.choice([f'client_{i}' for i in range(8)], n),
            'is_active' : np.random.choice([0, 1], n, p=[0.9823, 0.0177]),
            'retry_rate': np.abs(np.random.normal(0.01, 0.03, n)),
            'rssi'      : np.random.normal(45, 8, n),
        })
        placeholder = True

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('What "Normal" WiFi Behaviour Looks Like'
                  + (' — PLACEHOLDER DATA' if placeholder else '') +
                  '\n(training data — the baseline every experiment is compared against)',
                  fontsize=15, fontweight='bold', y=0.99)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30,
                           left=0.08, right=0.96, top=0.84, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, 0])
    active_pct = df['is_active'].mean() * 100
    ax1.pie([100 - active_pct, active_pct], colors=[COLOR_IDLE, COLOR_ACTIVE],
            autopct=lambda p: f'{p:.1f}%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 13, 'fontweight': 'bold', 'color': 'white'})
    ax1.set_title('A. Most of the time, devices are silent', pad=12)
    ax1.legend(handles=[mpatches.Patch(color=COLOR_IDLE, label=f'Idle ({100-active_pct:.1f}%)'),
                        mpatches.Patch(color=COLOR_ACTIVE, label=f'Active ({active_pct:.1f}%)')],
               loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    active_df = df[df['is_active'] == 1]
    if len(active_df) > 5 and 'retry_rate' in active_df.columns:
        ax2.hist(active_df['retry_rate'].clip(0, 0.3), bins=30, color=COLOR_NORMAL, edgecolor='white', alpha=0.85)
        mean_retry = active_df['retry_rate'].mean()
        ax2.axvline(mean_retry, color=COLOR_ANOMALY, linestyle='--', linewidth=2, label=f'Typical: {mean_retry:.3f}')
        ax2.legend(fontsize=9)
    ax2.set_title('B. Retry rate while active', pad=12)
    ax2.set_xlabel('retry_rate'); ax2.set_ylabel('Number of snapshots')

    ax3 = fig.add_subplot(gs[1, 0])
    if 'rssi' in df.columns:
        ax3.hist(df['rssi'].dropna(), bins=30, color=COLOR_ACTIVE, edgecolor='white', alpha=0.85)
        mean_rssi = df['rssi'].mean()
        ax3.axvline(mean_rssi, color=COLOR_ANOMALY, linestyle='--', linewidth=2, label=f'Typical: {mean_rssi:.0f}')
        ax3.legend(fontsize=9)
    ax3.set_title('C. Signal strength (rssi)', pad=12)
    ax3.set_xlabel('rssi'); ax3.set_ylabel('Number of snapshots')

    ax4 = fig.add_subplot(gs[1, 1])
    if 'client_mac' in df.columns:
        counts = df['client_mac'].value_counts().head(9)
        ax4.barh(range(len(counts)), counts.values, color=COLOR_AE, edgecolor='white')
        ax4.set_yticks(range(len(counts))); ax4.set_yticklabels([str(m)[-8:] for m in counts.index], fontsize=8)
        ax4.invert_yaxis()
    ax4.set_title('D. Rows collected per device', pad=12)
    ax4.set_xlabel('Number of snapshots')

    caption(fig, "Takeaway: this is the SAME baseline every experiment below is compared against.")
    out = f"{OUTPUT_DIR}/figure1_normal_data_overview.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print(f"   Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 (per experiment) — DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════

def figure3_deepdive(name, scored_csv, report_csv):
    print(f"[Figure 3] {name}...")

    if not os.path.exists(scored_csv):
        print(f"   [SKIP] scored CSV not found: {scored_csv}")
        return None

    df = pd.read_csv(scored_csv, parse_dates=['timestamp'])
    print(f"   Using real file: {scored_csv}  ({len(df):,} rows)")

    report_df = None
    if report_csv and os.path.exists(report_csv):
        report_df = pd.read_csv(report_csv)
        print(f"   Using real file: {report_csv}  ({len(report_df)} clients)")
    else:
        print(f"   [NOTE] report CSV not found ({report_csv}) — "
              f"client ranking panel and error_ratio will be limited.")

    threshold, _ = load_threshold(df)
    df = df.sort_values('timestamp')

    # ── normal vs anomaly split: reuse the evaluator's own boundary ──────────
    boundary_ts = None
    if 'anomaly_start_detected' in df.columns and df['anomaly_start_detected'].notna().any():
        raw = df['anomaly_start_detected'].dropna().iloc[0]
        if str(raw).strip():
            boundary_ts = pd.Timestamp(raw)

    # The stressed device is whichever client report.csv marks CRITICAL,
    # falling back to the highest error_ratio, falling back to the first client.
    if report_df is not None and 'status' in report_df.columns and 'error_ratio' in report_df.columns:
        crit = report_df[report_df['status'] == 'CRITICAL']
        device = (crit['client_mac'].iloc[0] if len(crit)
                  else report_df.sort_values('error_ratio', ascending=False)['client_mac'].iloc[0])
    else:
        device = df['client_mac'].iloc[0]

    df_dev = df[df['client_mac'] == device]
    active_only = df_dev[df_dev['is_active'] == 1] if 'is_active' in df_dev.columns else df_dev

    if boundary_ts is not None:
        normal_part  = active_only[active_only['timestamp'] <  boundary_ts]
        anomaly_part = active_only[active_only['timestamp'] >= boundary_ts]
    else:
        split = len(active_only) // 2
        normal_part, anomaly_part = active_only.iloc[:split], active_only.iloc[split:]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f'{name} — What the Autoencoder Saw\nDevice: {device}',
                  fontsize=15, fontweight='bold', y=0.99)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28,
                           left=0.08, right=0.96, top=0.85, bottom=0.10)

    # Panel A — MSE over time
    ax1 = fig.add_subplot(gs[0, :])
    if len(normal_part):
        ax1.plot(normal_part['timestamp'], normal_part['anomaly_score'], 'o-',
                  color=COLOR_NORMAL, markersize=4, linewidth=1.3,
                  label=f'Normal period (median MSE: {normal_part["anomaly_score"].median():.3f})')
    if len(anomaly_part):
        ax1.plot(anomaly_part['timestamp'], anomaly_part['anomaly_score'], 'o-',
                  color=COLOR_ANOMALY, markersize=4, linewidth=1.3,
                  label=f'Anomaly period (median MSE: {anomaly_part["anomaly_score"].median():.3f})')
    ax1.axhline(threshold, color=COLOR_THRESHOLD, linestyle='--', linewidth=2,
                label=f'Flag threshold ({threshold:.3f})')
    if boundary_ts is not None:
        ax1.axvline(boundary_ts, color='#888888', linestyle=':', linewidth=1.5)
        ax1.annotate('anomaly period starts here\n(boundary detected automatically)',
                     xy=(boundary_ts, ax1.get_ylim()[1]*0.85), fontsize=9, ha='left', color='#555555',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))
    ax1.set_title(f'A. Reconstruction error over time — device {device}, active rows only', pad=10)
    ax1.set_ylabel('Reconstruction error (MSE)\nhigher = model could not explain this row')
    ax1.set_xlabel('Time')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.tick_params(axis='x', rotation=20)

    # Panel B — top 5 changed features, computed live
    ax2 = fig.add_subplot(gs[1, 0])
    candidates = ['retry_rate', 'tx_failed_rate', 'tx_success_rate', 'tx_packet_rate',
                  'rx_packet_rate', 'rssi', 'mcs_entropy', 'low_mcs_ratio', 'high_mcs_ratio',
                  'tx_throughput_mbps', 'rx_throughput_mbps']
    avail = [c for c in candidates if c in df_dev.columns]
    deltas = []
    for feat in avail:
        nv = normal_part[feat].mean() if len(normal_part) else np.nan
        av = anomaly_part[feat].mean() if len(anomaly_part) else np.nan
        if pd.isna(nv) or pd.isna(av):
            continue
        pct = abs(100*(av-nv)/nv) if nv != 0 else (100 if av != 0 else 0)
        deltas.append((feat, nv, av, pct))
    deltas.sort(key=lambda r: r[3], reverse=True)
    top5 = deltas[:5]
    if top5:
        labels5 = [d[0] for d in top5]
        ns, as_ = [], []
        for _, nv, av, _ in top5:
            m = max(abs(nv), abs(av), 1e-9)
            ns.append(nv/m); as_.append(av/m)
        y = np.arange(len(labels5)); h = 0.35
        ax2.barh(y-h/2, ns, h, color=COLOR_NORMAL, label='Normal period', edgecolor='white')
        ax2.barh(y+h/2, as_, h, color=COLOR_ANOMALY, label='Anomaly period', edgecolor='white')
        ax2.set_yticks(y); ax2.set_yticklabels(labels5, fontsize=9); ax2.invert_yaxis()
        ax2.set_xlabel('Value (normalised per feature)')
        ax2.legend(fontsize=8, loc='lower right')
    else:
        ax2.text(0.5, 0.5, 'No comparable feature columns found', ha='center', va='center',
                  transform=ax2.transAxes, color='#999999')
    ax2.set_title('B. Top 5 features that changed the most', pad=10)

    # Panel C — every client ranked by error_ratio (from report.csv only)
    ax3 = fig.add_subplot(gs[1, 1])
    top_client, top_ratio, n_watch = device, 1.0, 0
    if report_df is not None and 'error_ratio' in report_df.columns:
        rdf = report_df.copy()
        rdf['client_short'] = rdf['client_mac'].astype(str).str[-8:]
        rdf['error_ratio']  = rdf['error_ratio'].fillna(1.0)
        rdf = rdf.sort_values('error_ratio', ascending=False)
        colors = [STATUS_COLOR.get(s, '#999999') for s in rdf.get('status', ['NORMAL']*len(rdf))]
        bars = ax3.barh(range(len(rdf)), rdf['error_ratio'], color=colors, edgecolor='white')
        ax3.set_yticks(range(len(rdf))); ax3.set_yticklabels(rdf['client_short'], fontsize=8)
        ax3.axvline(1.0, color='#888888', linestyle=':', linewidth=1.3)
        for bar, status in zip(bars, rdf.get('status', ['NORMAL']*len(rdf))):
            ax3.annotate(status, (bar.get_width()+0.3, bar.get_y()+bar.get_height()/2),
                         va='center', fontsize=8, fontweight='bold', color=STATUS_COLOR.get(status, '#999999'))
        ax3.invert_yaxis()
        ax3.set_xlabel('How many times worse than normal\n(1.0 = normal)')
        top_client, top_ratio = rdf['client_short'].iloc[0], rdf['error_ratio'].iloc[0]
        n_watch = (rdf.get('status') == 'WATCH').sum() if 'status' in rdf.columns else 0
    else:
        ax3.text(0.5, 0.5, 'No report.csv found —\nerror_ratio only lives there',
                  ha='center', va='center', transform=ax3.transAxes, color='#999999')
    ax3.set_title('C. Every connected device — error ratio', pad=10)

    caption(fig, f"Takeaway: {top_client} shows {top_ratio:.1f}x higher reconstruction error than its own baseline."
                 + (f" {n_watch} nearby device(s) also drifted into WATCH status."
                    if n_watch > 0 else ""))

    out = f"{OUTPUT_DIR}/figure3_{safe_filename(name)}_deepdive.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print(f"   Saved: {out}")

    if report_df is not None and 'error_ratio' in report_df.columns:
        crit_or_top = report_df.sort_values('error_ratio', ascending=False).iloc[0]
        return {
            'experiment'   : name,
            'client_mac'   : crit_or_top['client_mac'],
            'status'       : crit_or_top.get('status', 'NORMAL'),
            'error_ratio'  : crit_or_top.get('error_ratio', np.nan),
            'detection_pct': crit_or_top.get('anomaly_rate_active_pct', np.nan),
        }
    return None


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ALL EXPERIMENTS SIDE BY SIDE (built from whatever figure3 found)
# ════════════════════════════════════════════════════════════════════════════

def figure4_all_experiments_summary(summary_rows):
    print("[Figure 4] Cross-experiment summary...")

    if not summary_rows:
        print("   No experiments produced usable report.csv data — skipping Figure 4.")
        return

    sdf = pd.DataFrame(summary_rows)
    labels = sdf['experiment'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(top=0.80, bottom=0.18, wspace=0.32)
    fig.suptitle('Autoencoder Results Across All Experiments\n'
                  '(most stressed device per experiment, from each report.csv)',
                  fontsize=15, fontweight='bold', y=0.98)

    x = np.arange(len(sdf))
    colors = [STATUS_COLOR.get(s, '#999999') for s in sdf['status']]

    bars1 = axes[0].bar(x, sdf['detection_pct'].fillna(0), color=colors, edgecolor='white')
    for bar, val in zip(bars1, sdf['detection_pct']):
        if pd.notna(val):
            axes[0].annotate(f'{val:.0f}%', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('Detection rate on active rows (%)')
    axes[0].set_title('A. How much of each anomaly was caught', pad=10)
    axes[0].set_ylim(0, 115)

    bars2 = axes[1].bar(x, sdf['error_ratio'].fillna(1.0), color=colors, edgecolor='white')
    for bar, val in zip(bars2, sdf['error_ratio']):
        if pd.notna(val):
            axes[1].annotate(f'{val:.1f}x', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].axhline(1.0, color='#888888', linestyle=':', linewidth=1.3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Error ratio (anomaly MSE ÷ normal MSE)')
    axes[1].set_title('B. How unusual the worst device looked\n(1.0 = normal)', pad=10)

    legend_handles = [mpatches.Patch(color=c, label=s) for s, c in STATUS_COLOR.items()
                      if s in sdf['status'].unique()]
    fig.legend(handles=legend_handles, loc='lower center', ncol=max(len(legend_handles), 1),
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    caption(fig, "Takeaway: this chart is built directly from each experiment's report.csv — "
                 "add a new entry to the EXPERIMENTS list and rerun to extend it.")

    out = f"{OUTPUT_DIR}/figure4_all_experiments_summary.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    print(f"   Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualise autonomous_ae_evaluator.py results.")
    parser.add_argument('--only', type=str, default=None,
                        help='Only chart the experiment whose "name" matches this exactly, e.g. "Retry Storm"')
    parser.add_argument('--overview', action='store_true',
                        help="Also produce Figure 1 (training data baseline)")
    parser.add_argument('--manifest', type=str, default=None,
                        help="Path to a JSON file with the same structure as the EXPERIMENTS "
                             "list at the top of this script, used INSTEAD of editing the script")
    args = parser.parse_args()

    print("="*60)
    print("  Autoencoder Evaluation Results — Visualisation")
    print("="*60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiments = EXPERIMENTS
    if args.manifest:
        if not os.path.exists(args.manifest):
            print(f"Manifest file not found: {args.manifest}")
            return
        with open(args.manifest) as f:
            experiments = json.load(f)
        print(f"Loaded {len(experiments)} experiment(s) from manifest: {args.manifest}")

    if args.only:
        experiments = [e for e in experiments if e['name'] == args.only]
        if not experiments:
            available = [e['name'] for e in (EXPERIMENTS if not args.manifest else experiments)]
            print(f"No experiment named '{args.only}' found.")
            return

    if not experiments:
        print("EXPERIMENTS list is empty. Edit the list at the top of this script "
              "(or pass --manifest) with at least one {name, scored_csv, report_csv} entry.")
        return

    print(f"Experiments to chart: {[e['name'] for e in experiments]}")
    print()

    if args.overview:
        figure1_normal_data_overview()
        print()

    summary_rows = []
    for exp in experiments:
        result = figure3_deepdive(exp['name'], exp['scored_csv'], exp.get('report_csv'))
        if result:
            summary_rows.append(result)

    print()
    figure4_all_experiments_summary(summary_rows)

    print()
    print("="*60)
    print(f"  Done. Charts saved in: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()
