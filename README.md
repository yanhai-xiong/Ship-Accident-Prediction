# Ship-Accident-Prediction

Consolidated training pipeline (see `src/ship_accident/`, `configs/`).

## Data: `all_data_0710.xlsx`

The file **`all_data_0710.xlsx`** is available on the upstream GitHub repo  
[yanhai-xiong/Ship-Accident-Prediction](https://github.com/yanhai-xiong/Ship-Accident-Prediction) (`main` branch). Fetch locally with:

```bash
curl -fL -o all_data_0710.xlsx \
  "https://raw.githubusercontent.com/yanhai-xiong/Ship-Accident-Prediction/main/all_data_0710.xlsx"
```

## Run all models + report

Uses legacy column names from the original notebooks (Chinese field names + English where applicable):

```bash
PYTHONPATH=src python3 scripts/run_all_reports.py --data all_data_0710.xlsx --out-dir reports
```

Outputs:

- `reports/REPORT.md` — metrics tables + sklearn classification reports  
- `reports/model1a_tabular.json`, `model1a_tfidf.json`, `model3a_accident_level.json`

Single run:

```bash
PYTHONPATH=src python3 -m ship_accident.cli -c configs/legacy_excel_model1a.yaml -d all_data_0710.xlsx
```

## Install

```bash
python3 -m pip install -e .
```

Legacy notebooks are under `legacy/`.
