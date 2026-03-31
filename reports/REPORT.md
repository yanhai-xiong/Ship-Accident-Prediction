# Training report

Data file: `/Users/junnuo.yu1@iqvia.com/Ship-Accident-Prediction/all_data_0710.xlsx`

## model1a_tabular.json
- **Config**: `configs/legacy_excel_model1a.yaml`
- **Target**: `accident type`
- **Mode**: `tabular_only`
- **Train / test rows**: 592 / 254
- **Feature dim**: 67

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|---------------|---------------|
| gradient_boosting | 0.7087 | 0.4411 | 0.6914 | 0.6673 |
| random_forest | 0.7047 | 0.3561 | 0.6493 | 0.6909 |
| svm | 0.6732 | 0.4639 | 0.6620 | 0.6790 |
| knn | 0.6457 | 0.3206 | 0.6136 | 0.6471 |

## model1a_tfidf.json
- **Config**: `configs/legacy_excel_model1a_tfidf.yaml`
- **Target**: `accident type`
- **Mode**: `tabular_tfidf`
- **Train / test rows**: 558 / 240
- **Feature dim**: 2068

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|---------------|---------------|
| gradient_boosting | 0.6958 | 0.4048 | 0.6725 | 0.7116 |
| random_forest | 0.6917 | 0.3909 | 0.6603 | 0.7133 |
| svm | 0.7042 | 0.4339 | 0.6984 | 0.7151 |
| knn | 0.6583 | 0.3478 | 0.6360 | 0.6703 |

## model3a_accident_level.json
- **Config**: `configs/legacy_excel_model3a.yaml`
- **Target**: `accident level`
- **Mode**: `tabular_only`
- **Train / test rows**: 592 / 254
- **Feature dim**: 77

| Model | Test accuracy | Macro-F1 | Weighted-F1 | Best CV score |
|-------|---------------|----------|---------------|---------------|
| gradient_boosting | 0.7480 | 0.4089 | 0.7105 | 0.7906 |
| random_forest | 0.7953 | 0.3332 | 0.7194 | 0.8075 |
| svm | 0.7953 | 0.3217 | 0.7119 | 0.8024 |
| knn | 0.7559 | 0.3590 | 0.7169 | 0.7787 |

## Classification reports (test set)

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

