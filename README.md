# Unsupervised Anomaly Detection for WiFi-6 Clients

Master's thesis project: detecting anomalous behaviour of WiFi-6 (IEEE 802.11ax) client devices from access-point telemetry, without labelled anomaly data.

**Author:** Shravan Yeliyur Ravi
**Programme:** M.Sc. Computer Science, Bauhaus-Universität Weimar
**Host company:** Phoenix Contact

---

## Overview

The access point (Qualcomm/Atheros chipset) exposes per-client counters through the `apstats -a -R` command. This project:

1. Collects `apstats` snapshots every 10 seconds.
2. Parses them into per-client rows and engineers rate, ratio and MCS features.
3. Trains two unsupervised detectors on normal traffic:
   - **Isolation Forest** (scikit-learn), the baseline
   - **Autoencoder** (PyTorch), the final model, scored by reconstruction error
4. Evaluates both against synthetic anomalies and four controlled real-world experiments.
5. Deploys the autoencoder **on the AP itself** as a pure-Python monitor with no third-party dependencies.

### Why two models per detector

Client rows are either *idle* or *active* (`is_active` flag). One model trained on everything learns "idle = normal" and flags any traffic as anomalous. Each detector therefore trains a separate **idle model** and **active model**.

| Model | Features | Autoencoder architecture |
|---|---|---|
| Idle | `rssi`, `tx_success_rate`, `retry_rate` | 3 → 8 → 4 → 8 → 3 |
| Active | 15 traffic, throughput and MCS features | 15 → 32 → 16 → 8 → 16 → 32 → 15 |

The full feature lists and thresholds are in `models/ae_metadata.json` and `models_active/ae_metadata.json`.

---

## Repository structure

```
.
├── parse_apstats.py                 # raw apstats log  -> clean CSV
├── combine_files.py                 # concatenate normal + anomaly logs of one experiment
├── feature_engineering.py           # clean CSV -> ML-ready CSV (rates, ratios, MCS features)
│
├── isolation_forest_training.py     # train IF idle/active models        -> models/
├── anomaly_testing.py               # synthetic anomaly injection tests (IF)
├── real_experiment_evaluator.py     # evaluate IF on real experiments (manual split time)
├── real_experiment_evaluator_pelt.py# same, automatic change-point split (ruptures/PELT)
├── autonomous_if_detector.py        # IF: find anomalous client + onset automatically
├── check_metrics.py                 # print IF thresholds
│
├── autoencoder_training.py          # train AE idle/active models        -> models/
├── autoencoder_training_v2.py       # retrain active AE on larger dataset -> models_active/
├── autonomous_ae_evaluator.py       # AE: find anomalous client + onset  (uses models/)
├── autonomous_ae_evaluator_v2.py    # same, using models_active/         -> results_active/
│
├── export_model_weights.py          # .pt + .pkl -> ae_weights_export.json (models/)
├── export_model_weights_v2.py       # same, from models_active/
├── ae_weights_export.json           # exported weights deployed to the AP
│
├── ap_monitor.py                    # ON-AP live monitor (pure Python stdlib)
├── apstats_parser.py                #   parses one live apstats snapshot
├── feature_calculator.py            #   computes the features from two consecutive readings
├── ae_inference.py                  #   pure-Python autoencoder forward pass
├── diagnose_idle_rssi.py            # inspect idle-model reconstruction error for given RSSI
│
├── visualize_experiment_results_if.py   # IF result plots
├── visualize_ae_results.py              # AE result plots (thesis figures)
├── visualize_experiments_results_ae.py  # additional AE timeline plots
│
├── data_raw*/                       # raw apstats captures (training + experiments)
├── clean_output*.csv                # parsed data
├── ml_ready_data*.csv               # feature-engineered data
├── models/                          # IF models + original AE models
├── models_active/                   # retrained active AE (78,265 training rows)
├── results/  results_active/        # evaluation outputs and training reports
├── plots/                           # figures used in the thesis
└── WiFi_Feature_Engineering_*.pdf   # feature-engineering documentation
```

### File naming for experiments

| Suffix | Experiment |
|---|---|
| `_RStorm` | Retry storm |
| `_signalD` | Signal degradation |
| `_trafficB` | Traffic burst |
| `_LinkF` | Link failure |

---

## Setup

Python 3.10+ recommended.

```bash
git clone https://github.com/ShravanYR/wifi-anomaly-detection-thesis.git
cd wifi-anomaly-detection-thesis
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Several scripts read their input and output file names from constants at the top of the file (`INPUT_FILE`, `OUTPUT_FILE`, `MODELS_DIR`, `RESULTS_DIR`). Check these before running a step.

---

## Pipeline

### 1. Parse raw telemetry

```bash
python parse_apstats.py <raw_apstats.txt> <clean_output.csv>
```

For an experiment, first merge the normal and anomaly captures in order (edit `input_folder` and `file_order` in `combine_files.py`):

```bash
python combine_files.py              # -> combined_apstats.txt
python parse_apstats.py combined_apstats.txt clean_output_ex_RStorm.csv
```

### 2. Feature engineering

Set `INPUT_FILE` / `OUTPUT_FILE` in `feature_engineering.py`, then:

```bash
python feature_engineering.py        # e.g. -> ml_ready_data_ex_RStorm.csv
```

### 3a. Isolation Forest (baseline)

```bash
python isolation_forest_training.py          # trains on ml_ready_data.csv -> models/
python anomaly_testing.py                    # synthetic anomaly tests
python real_experiment_evaluator_pelt.py     # real experiments, automatic split
python autonomous_if_detector.py ml_ready_data_ex_signalD.csv
```

### 3b. Autoencoder (final model)

```bash
python autoencoder_training.py               # -> models/
python autoencoder_training_v2.py            # retrain active model -> models_active/
python autonomous_ae_evaluator_v2.py ml_ready_data_ex_RStorm.csv
python visualize_ae_results.py --overview    # thesis figures -> plots/
```

`autoencoder_training_v2.py` skips the idle model when there are fewer than 100 idle rows. In that case, the idle model from `models/` is copied into `models_active/`.

---

## On-device deployment (AP)

The AP has no internet access and no pip, so inference runs in **pure Python 3 standard library**: no numpy, torch or scikit-learn.

1. On the workstation, export the trained weights and scalers to plain JSON:
   ```bash
   python export_model_weights_v2.py      # -> ae_weights_export.json
   ```
2. Copy these files to one folder on the AP:
   ```
   ap_monitor.py  apstats_parser.py  feature_calculator.py  ae_inference.py  ae_weights_export.json
   ```
3. Run the monitor:
   ```bash
   python3 ap_monitor.py
   ```

Every 10 s the monitor polls `apstats -a -R`, computes features per client from consecutive readings, and scores them with the idle or active autoencoder. Results go to `ap_monitor.log` as heartbeat or alert lines. The log is size-capped for the AP's limited storage.

---

## Experiments

| Experiment | How it was induced | Expected signature |
|---|---|---|
| Retry storm | Network stress (e.g. `hrping`) | Retry and failure rates rise, success rate drops |
| Signal degradation | Client moved away from the AP | RSSI drops, MCS shifts to lower indices |
| Traffic burst | Sustained high-throughput download | Packet, byte and throughput rates spike |
| Link failure | WiFi disconnect / toggle | Client disappears, counters reset |

Each experiment capture contains a normal period followed by an anomaly period. Evaluation outputs are in `results/` (Isolation Forest, original AE) and `results_active/` (retrained AE). Figures are in `plots/`.

---

## License

Academic work. All rights reserved unless otherwise agreed with the author and Phoenix Contact.
