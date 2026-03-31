# Training report

This document summarizes **features**, **configs**, **models**, and **held-out test metrics** (accuracy, macro-F1, weighted-F1, best CV score). Metrics come from `sklearn` `GridSearchCV` unless noted. They are **not** guarantees on new data.

### About this document

- **Pipeline:** Report corpus → `data/report_corpus.csv` (`scripts/build_report_dataset.py`); training → `ship_accident.cli` + YAML under `configs/`; metrics → `reports/*.json`.
- **Best test accuracy (report corpus):** **83.7%** accident type — `hist_gradient_boosting`, `reports/report_type_tuned.json`. **81.4%** severity — `gradient_boosting`, `reports/report_severity.json`.

---

## 1. Feature inventory

### 1.1 Report corpus (`data/report_corpus.csv`)

Built by `scripts/build_report_dataset.py`. Each row is one investigation report file.

**Identifier / label / text columns (not all used as model inputs):**

| Column | Role |
|--------|------|
| `file_id` | File stem; dropped from X (`drop_always`). |
| `path` | Full path; dropped from X. |
| `accident type` | Target for type models; kept as categorical input for **severity** models (one-hot after encoding). |
| `title_type_raw` | Parsed title fragment; dropped from X. |
| `accident severity` | Target for severity models; dropped from X when predicting **type**. |
| `severity_label_source` | `outcomes` or `text`; dropped from X. |
| `severity_match_raw` | Matched phrase; dropped from X. |
| `full_text` | Report body; **not** in tabular matrix — fed to **TF-IDF** when `mode: tabular_tfidf`. |

**Post-outcome / audit columns** (`engineer_post_accident_features` — stored in CSV for labeling and audit; **excluded from X** via YAML `drop_always` for both type and severity predictors):

| Column | Description |
|--------|-------------|
| `num_direct_loss_wan` | Direct economic loss (万元) from text. |
| `num_deaths_reported` | Death count parsed from text. |
| `num_missing_reported` | Missing-person count parsed from text. |
| `flag_injury_mentioned` | 1 if injury wording present. |
| `post_outcome_keyword_hits` | Count of outcome-related keywords. |

**Engineered tabular features** (`engineer_report_features` + `_engineer_hint_based_features` in `src/ship_accident/report_parse.py`):

| Group | Feature name | Notes |
|-------|--------------|--------|
| Keyword counts | `env_keyword_hits` | Hits from `_ENV_KEYS` (e.g. 大风, 浓雾, 台风, …). |
| | `ops_keyword_hits` | `_OPS_KEYS` (靠泊, 狭水道, …). |
| | `vessel_type_hits` | `_VESSEL_KEYS` (干货船, 集装箱, …). |
| | `human_factor_hits` | `_HUMAN_KEYS` (瞭望, 配员, …). |
| | `consequence_hits` | `_CONSEQ_KEYS` (人员伤亡, 溢油, …). |
| | `area_keyword_hits` | `_AREA_KEYS` (近海, 内河, …). |
| | `equipment_keyword_hits` | `_EQUIPMENT_KEYS` (AIS, VDR, 雷达, …). |
| | `cargo_keyword_hits` | `_CARGO_KEYS` (煤炭, 原油, …). |
| | `regulatory_keyword_hits` | `_REGULATORY_KEYS` (安全管理, 适航证书, …). |
| | `pilot_tug_keyword_hits` | `_PILOT_TUG_KEYS` (引航, 拖轮, …). |
| Parsed numbers (regex) | `num_gross_ton` | 总吨 |
| | `num_net_ton` | 净吨 |
| | `num_main_engine_kw` | 主机功率 |
| | `num_length_m` | 船长 (m) |
| | `num_draft_m` | 吃水 (m) |
| | `num_speed_kn` | 航速 (节) |
| | `num_visibility_m` | 能见度 (m) |
| | `num_crew_onboard` | 在船人员 / 配员 |
| Time / weather (parsed) | `num_wind_level_max` | Max Beaufort-style level from ranges or single match. |
| | `num_accident_hour` | Hour of accident; `-1` if unknown. |
| | `flag_night_accident` | 1 if hour in night window. |
| Vessel / calendar hints | `num_year_built` | From 建造完工日期 / 建成日期, etc. |
| | `num_accident_year`, `num_accident_month` | From accident date line. |
| | `num_vessel_age_years` | `accident_year − year_built` (clamped). |
| | `num_beam_m` | 船宽 |
| | `num_dwt_approx` | 载重吨 / 载货量 |
| Ratios | `ratio_length_to_beam` | `num_length_m / num_beam_m` when both > 0. |
| | `ratio_kw_per_gross_ton` | `num_main_engine_kw / num_gross_ton`. |
| | `ratio_draft_to_depth_hint` | `num_draft_m / num_length_m`. |
| Position | `num_latitude`, `num_longitude` | First `dd°mm′ N/E` pair in header region. |
| Environment scalars | `num_wind_speed_ms` | 风速 (m/s). |
| | `num_wave_height_m_env` | 浪高 (m). |
| | `num_wave_period_s` | 浪周期 (s). |
| | `num_air_temp_c` | 气温 (℃). |
| | `num_sea_temp_c` | 水温. |
| | `num_pressure_hpa` | 气压 (hPa). |
| Flags / counts | `flag_mentions_solas` | SOLAS / 索拉斯. |
| | `class_society_keyword_hits` | CCS, DNV, 船级社, … |
| | `num_passengers_reported` | 旅客 / 载客 / 乘客. |
| | `num_crew_reported_alt` | Alternate 船员 patterns. |

**Encoding:** `encode_tabular_train_test` uses `pd.get_dummies` on remaining columns (string columns become multiple binary columns). **Total tabular width** depends on which categorical levels appear in the training split.

**Text features:** `TfidfVectorizer(max_features=tfidf_max_features)` on `full_text` (default `analyzer='word'`, unigrams). The pipeline **concatenates** `[tabular_dense, tfidf_dense]`; the combined column count is `n_features` in each JSON file.

---

### 1.2 Legacy Excel (`all_data_0710.xlsx`)

Raw columns (before YAML drops and one-hot encoding):

`Unnamed: 0`, `accident type`, `发生地种类`, `season`, `wind speed level`, `dayNight`, `fatality`, `作业情况`, `损伤种类`, `损伤位置`, `wind direction`, `能见度等级`, `是否超速`, `路况种类`, `accident level`, `ship type`, `船体材料`, `主机类型`, `船长`, `船宽`, `Cause of Accident`, `设备故障`, `环境恶劣`, `事发水域路况`.

Configs drop overlapping / leakage columns (e.g. `fatality`, `损伤位置`, `损伤种类`; plus target-specific drops in `legacy_excel_model1a.yaml` / `model3a`). Remaining fields are one-hot encoded to the `n_features` shown below.

---

## 2. Report corpus — training metrics (by JSON artifact)

### report_type.json

- **Config:** `configs/report_accident_type_tfidf.yaml`
- **Target:** `accident type`
- **Mode:** `tabular_tfidf`
- **Train / test rows:** 463 / 199
- **Feature dim:** 821
- **Feature selection:** off

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7437 | 0.3746 | 0.6901 | 0.7452 |
| gradient_boosting | 0.7688 | 0.4429 | 0.7441 | 0.7711 |

---

### report_tfidf.json

- **Config:** `configs/report_accident_type_tfidf.yaml` (same as above; alternate output name from README workflow)
- **Target:** `accident type`
- **Mode:** `tabular_tfidf`
- **Train / test rows:** 463 / 199
- **Feature dim:** 815
- **Feature selection:** off

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7487 | 0.3905 | 0.6956 | 0.7429 |
| gradient_boosting | 0.7688 | 0.4398 | 0.7458 | 0.7668 |

---

### report_tfidf_fs.json

- **Config:** `configs/report_accident_type_tfidf_fs.yaml` (`SelectFromModel`, threshold grid `median` / `mean`)
- **Target:** `accident type`
- **Mode:** `tabular_tfidf`
- **Train / test rows:** 463 / 199
- **Feature dim (input to pipeline):** 815
- **Feature selection:** on

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7588 | 0.3967 | 0.7079 | 0.7580 |
| gradient_boosting | 0.7739 | 0.4362 | 0.7491 | 0.7690 |

---

### report_tfidf_no_docmeta.json · report_tfidf_fs_no_docmeta.json

Experimental ablations (additional columns dropped before training; exact YAML may not be checked in). Metrics:

**report_tfidf_no_docmeta.json** — `n_features`: 810, feature selection off.

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7487 | 0.3895 | 0.6957 | 0.7472 |
| gradient_boosting | 0.7538 | 0.4232 | 0.7333 | 0.7711 |

**report_tfidf_fs_no_docmeta.json** — `n_features`: 810, feature selection on.

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7789 | 0.4087 | 0.7308 | 0.7581 |
| gradient_boosting | 0.7487 | 0.4175 | 0.7185 | 0.7690 |

---

### report_type_tuned.json

- **Config:** `configs/report_accident_type_tuned.yaml`
- **Target:** `accident type`
- **Mode:** `tabular_tfidf` (`tfidf_max_features: 2500`, `class_weight: balanced`)
- **Train / test rows:** 496 / 166
- **Feature dim:** 2542
- **Feature selection:** off

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7590 | 0.4240 | 0.7087 | 0.7077 |
| hist_gradient_boosting | 0.8373 | 0.6018 | 0.8378 | 0.7924 |

---

### report_type_tuned_v2.json

- **Config:** `configs/report_accident_type_tuned_v2.yaml` (HistGradientBoosting only, `tfidf_max_features: 4000`)
- **Target:** `accident type`
- **Mode:** `tabular_tfidf`
- **Train / test rows:** 496 / 166
- **Feature dim:** 4042
- **Feature selection:** off

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| hist_gradient_boosting | 0.8012 | 0.5543 | 0.8016 | 0.7945 |

---

### report_severity.json

- **Config:** `configs/report_severity_tfidf.yaml`
- **Target:** `accident severity`
- **Mode:** `tabular_tfidf` (includes **one-hot `accident type`**; post-outcome columns dropped from X)
- **Train / test rows:** 463 / 199
- **Feature dim:** 833
- **Feature selection:** off

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| random_forest | 0.7889 | 0.3721 | 0.7331 | 0.8122 |
| gradient_boosting | 0.8141 | 0.4138 | 0.7886 | 0.8402 |

---

## 3. Legacy Excel — training metrics

Data file: `all_data_0710.xlsx` (path may differ on your machine).

### model1a_tabular.json

- **Config:** `configs/legacy_excel_model1a.yaml`
- **Target:** `accident type`
- **Mode:** `tabular_only`
- **Train / test rows:** 592 / 254
- **Feature dim:** 67

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| gradient_boosting | 0.7087 | 0.4411 | 0.6914 | 0.6673 |
| random_forest | 0.7047 | 0.3561 | 0.6493 | 0.6909 |
| svm | 0.6732 | 0.4639 | 0.6620 | 0.6790 |
| knn | 0.6457 | 0.3206 | 0.6136 | 0.6471 |

---

### model1a_tfidf.json

- **Config:** `configs/legacy_excel_model1a_tfidf.yaml`
- **Target:** `accident type`
- **Mode:** `tabular_tfidf`
- **Train / test rows:** 558 / 240
- **Feature dim:** 2068

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| gradient_boosting | 0.6958 | 0.4048 | 0.6725 | 0.7116 |
| random_forest | 0.6917 | 0.3909 | 0.6603 | 0.7133 |
| svm | 0.7042 | 0.4339 | 0.6984 | 0.7151 |
| knn | 0.6583 | 0.3478 | 0.6360 | 0.6703 |

---

### model3a_accident_level.json

- **Config:** `configs/legacy_excel_model3a.yaml`
- **Target:** `accident level`
- **Mode:** `tabular_only`
- **Train / test rows:** 592 / 254
- **Feature dim:** 77

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|-------------|---------------|
| gradient_boosting | 0.7480 | 0.4089 | 0.7105 | 0.7906 |
| random_forest | 0.7953 | 0.3332 | 0.7194 | 0.8075 |
| svm | 0.7953 | 0.3217 | 0.7119 | 0.8024 |
| knn | 0.7559 | 0.3590 | 0.7169 | 0.7787 |

---

## 4. Classification reports (test set) — legacy Excel

Per-class sklearn `classification_report` strings from the same runs as Section 3.

### model1a_tabular.json

#### gradient_boosting

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         1
                    触碰       0.43      0.17      0.24        18
                    触礁       0.42      0.56      0.48         9
                others       0.71      0.71      0.71        41
self-sinking (sinking)       0.68      0.53      0.60        43
             Grounding       0.20      0.20      0.20         5
        Fire/explosion       0.80      0.44      0.57         9
         Wind Accident       0.78      0.78      0.78         9
             collision       0.78      0.92      0.84       118
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.71       254
             macro avg       0.48      0.43      0.44       254
          weighted avg       0.69      0.71      0.69       254

```

#### random_forest

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         1
                    触碰       0.00      0.00      0.00        18
                    触礁       0.33      0.11      0.17         9
                others       0.69      0.76      0.72        41
self-sinking (sinking)       0.70      0.53      0.61        43
             Grounding       0.00      0.00      0.00         5
        Fire/explosion       1.00      0.33      0.50         9
         Wind Accident       0.86      0.67      0.75         9
             collision       0.71      0.97      0.82       118
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.70       254
             macro avg       0.43      0.34      0.36       254
          weighted avg       0.63      0.70      0.65       254

```

#### svm

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         1
                    触碰       0.33      0.33      0.33        18
                    触礁       0.31      0.44      0.36         9
                others       0.60      0.73      0.66        41
self-sinking (sinking)       0.61      0.51      0.56        43
             Grounding       0.00      0.00      0.00         5
        Fire/explosion       0.50      0.11      0.18         9
         Wind Accident       0.75      0.67      0.71         9
             collision       0.82      0.86      0.84       118
         wind Accident       1.00      1.00      1.00         1

              accuracy                           0.67       254
             macro avg       0.49      0.47      0.46       254
          weighted avg       0.66      0.67      0.66       254

```

#### knn

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         1
                    触碰       0.30      0.17      0.21        18
                    触礁       0.50      0.44      0.47         9
                others       0.53      0.76      0.62        41
self-sinking (sinking)       0.55      0.51      0.53        43
             Grounding       0.00      0.00      0.00         5
        Fire/explosion       0.00      0.00      0.00         9
         Wind Accident       0.80      0.44      0.57         9
             collision       0.76      0.85      0.80       118
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.65       254
             macro avg       0.34      0.32      0.32       254
          weighted avg       0.60      0.65      0.61       254

```

### model1a_tfidf.json

#### gradient_boosting

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         0
                    触碰       0.25      0.06      0.10        17
                    触礁       0.38      0.56      0.45         9
                others       0.56      0.54      0.55        37
self-sinking (sinking)       0.64      0.74      0.69        43
             Grounding       0.40      0.40      0.40         5
        Fire/explosion       0.57      0.44      0.50         9
         Wind Accident       1.00      0.33      0.50         9
             collision       0.82      0.91      0.86       110
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.70       240
             macro avg       0.46      0.40      0.40       240
          weighted avg       0.68      0.70      0.67       240

```

#### random_forest

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         0
                    触碰       0.14      0.06      0.08        17
                    触礁       0.80      0.44      0.57         9
                others       0.52      0.59      0.56        37
self-sinking (sinking)       0.62      0.67      0.64        43
             Grounding       1.00      0.20      0.33         5
        Fire/explosion       1.00      0.22      0.36         9
         Wind Accident       1.00      0.33      0.50         9
             collision       0.78      0.95      0.86       110
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.69       240
             macro avg       0.59      0.35      0.39       240
          weighted avg       0.69      0.69      0.66       240

```

#### svm

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         0
                    触碰       0.29      0.24      0.26        17
                    触礁       0.50      0.56      0.53         9
                others       0.56      0.54      0.55        37
self-sinking (sinking)       0.60      0.67      0.64        43
             Grounding       0.40      0.40      0.40         5
        Fire/explosion       0.38      0.33      0.35         9
         Wind Accident       1.00      0.56      0.71         9
             collision       0.89      0.92      0.90       110
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.70       240
             macro avg       0.46      0.42      0.43       240
          weighted avg       0.70      0.70      0.70       240

```

#### knn

```
                        precision    recall  f1-score   support

        fire/explosion       0.00      0.00      0.00         0
                    触碰       0.40      0.24      0.30        17
                    触礁       0.40      0.44      0.42         9
                others       0.44      0.65      0.53        37
self-sinking (sinking)       0.60      0.60      0.60        43
             Grounding       0.00      0.00      0.00         5
        Fire/explosion       0.00      0.00      0.00         9
         Wind Accident       1.00      0.67      0.80         9
             collision       0.80      0.85      0.83       110
         wind Accident       0.00      0.00      0.00         1

              accuracy                           0.66       240
             macro avg       0.37      0.35      0.35       240
          weighted avg       0.63      0.66      0.64       240

```

### model3a_accident_level.json

#### gradient_boosting

```
              precision    recall  f1-score   support

          一般       0.80      0.92      0.86       200
          较大       0.27      0.12      0.17        48
          重大       0.25      0.17      0.20         6

    accuracy                           0.75       254
   macro avg       0.44      0.40      0.41       254
weighted avg       0.69      0.75      0.71       254

```

#### random_forest

```
              precision    recall  f1-score   support

          一般       0.80      0.99      0.89       200
          较大       0.60      0.06      0.11        48
          重大       0.00      0.00      0.00         6

    accuracy                           0.80       254
   macro avg       0.47      0.35      0.33       254
weighted avg       0.74      0.80      0.72       254

```

#### svm

```
              precision    recall  f1-score   support

          一般       0.79      1.00      0.88       200
          较大       1.00      0.04      0.08        48
          重大       0.00      0.00      0.00         6

    accuracy                           0.80       254
   macro avg       0.60      0.35      0.32       254
weighted avg       0.81      0.80      0.71       254

```

#### knn

```
              precision    recall  f1-score   support

          一般       0.80      0.92      0.86       200
          较大       0.32      0.17      0.22        48
          重大       0.00      0.00      0.00         6

    accuracy                           0.76       254
   macro avg       0.37      0.36      0.36       254
weighted avg       0.69      0.76      0.72       254

```

---

## 5. Report corpus — per-class reports

For test-set per-class precision / recall / F1 for **report** runs, open the matching file under `reports/` and read the `classification_report` field inside each model block (e.g. `reports/report_type_tuned.json` → `models.hist_gradient_boosting.classification_report`).

---

*Figures rounded to four decimals where applicable; values match `reports/*.json` at time of last update.*
