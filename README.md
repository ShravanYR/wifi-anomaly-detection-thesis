# WiFi Anomaly Detection — Master Thesis

Anomaly detection in WiFi client devices using AP telemetry (`apstats`) and Isolation Forest models.

## Pipeline

1. **Collect** — Periodic `apstats` snapshots from the AP (10s interval, ~10 clients)
2. **Parse** — `parse_apstats.py` converts raw logs to per-client CSV rows
3. **Feature engineering** — `feature_engineering.py` builds rates, MCS features, activity flags
4. **Train** — `isolation_forest_training.py` trains dual idle/active Isolation Forest models
5. **Evaluate** — Synthetic tests (`anomaly_testing.py`) and four real experiments (retry storm, link failure, traffic burst, signal degradation)

## Quick start

```bash
pip install pandas numpy scikit-learn matplotlib ruptures

python parse_apstats.py data_raw/apstats_*.txt clean_output.csv
python feature_engineering.py
python isolation_forest_training.py
python anomaly_testing.py
```

## Experiment types

| Experiment | Description |
|------------|-------------|
| Retry storm | High retransmissions under stress (e.g. hrping) |
| Link failure | WiFi disconnect / toggle |
| Traffic burst | Sustained high throughput download |
| Signal degradation | Device moved away from AP |

## Author

Shravan Yeliyur Ravi — Master Thesis, CS for Digital Media
