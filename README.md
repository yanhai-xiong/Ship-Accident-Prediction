# Ship-Accident-Prediction

Consolidated training pipeline (see `src/ship_accident/`, `configs/`).

## Data: Mandarin report folder (preferred for accident-type experiments)

Raw investigation reports (PDF / DOC / DOCX / WPS) can be used instead of the legacy spreadsheet. Build a training table:

```bash
python3 -m pip install -e ".[reports]"
PYTHONPATH=src python3 scripts/build_report_dataset.py \
  --reports-dir "/path/to/ship accident report" \
  -o data/report_corpus.csv
```

Then train (tabular engineered fields + TF-IDF on `full_text`; labels inferred from report titles):

```bash
PYTHONPATH=src python3 -m ship_accident.cli \
  -c configs/report_accident_type_tfidf.yaml \
  -d data/report_corpus.csv \
  --output-json reports/report_tfidf.json
```

Feature selection (`SelectFromModel` + random forest, inside cross-validation):

```bash
PYTHONPATH=src python3 -m ship_accident.cli \
  -c configs/report_accident_type_tfidf_fs.yaml \
  -d data/report_corpus.csv \
  --output-json reports/report_tfidf_fs.json
```

Optional config keys: `min_class_counts` (drop rare labels before split), `feature_selection` (see `configs/report_accident_type_tfidf_fs.yaml`).

**Parsed “hint” features** (vessel / environment / ops — only when the report text matches regexes; otherwise 0): ship dimensions and ratios (`num_beam_m`, `num_dwt_approx`, `num_year_built`, `num_vessel_age_years`, `ratio_length_to_beam`, `ratio_kw_per_gross_ton`, …), accident calendar (`num_accident_year`, `num_accident_month`), position (`num_latitude`, `num_longitude` when coordinates appear as `dd°mm′ N/E`), weather scalars (`num_wind_speed_ms`, `num_wave_height_m_env`, `num_air_temp_c`, …), and flags (`flag_mentions_solas`, `class_society_keyword_hits`, `num_passengers_reported`). See `engineer_report_features` / `_engineer_hint_based_features` in `src/ship_accident/report_parse.py`.

**Tuning:** `configs/report_accident_type_tuned.yaml` uses `class_weight: balanced`, `tfidf_max_features: 2500`, `HistGradientBoostingClassifier`, and a 75/25 split. On the rebuilt report corpus this reached about **0.84** test accuracy (multiclass accident type — dominated by 碰撞); results vary slightly with data and grid. This is **not** a guarantee of 90% accuracy on other samples.

### Accident type vs accident severity (two targets)

The built CSV includes:

- **`accident type`** — parsed from the report title (same as before).
- **`accident severity`** — training label: **prefer** a tier from **outcome fields** (`severity_tier_from_outcomes`: casualties and direct loss in 万元), else **fallback** to phrases in the report (`infer_accident_severity`). See `resolve_accident_severity_training_label` in `report_parse.py`.
- **`severity_label_source`** — `outcomes` or `text` (for analysis; dropped during training).
- **Post-outcome columns** (`num_direct_loss_wan`, casualties, etc.) — stored for **labeling and audit**; they are **not** used as model inputs for severity (see `drop_always` in `configs/report_severity_tfidf.yaml`).

Train **type** without leaking outcomes: `configs/report_accident_type_tfidf.yaml` drops severity columns and all post-outcome columns from inputs.

Train **severity** using only pre-outcome tabular fields + TF-IDF + **`accident type`** (one-hot); **post-outcome columns are excluded** from inputs:

```bash
PYTHONPATH=src python3 -m ship_accident.cli \
  -c configs/report_severity_tfidf.yaml \
  -d data/report_corpus.csv \
  --output-json reports/report_severity.json
```

---

## Data: `all_data_0710.xlsx` (legacy)

The file **`all_data_0710.xlsx`** is available on the upstream GitHub repo  
[yanhai-xiong/Ship-Accident-Prediction](https://github.com/yanhai-xiong/Ship-Accident-Prediction) (`main` branch). It is convenient for reproducing older notebooks but may not match raw reports one-to-one. Fetch locally with:

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
