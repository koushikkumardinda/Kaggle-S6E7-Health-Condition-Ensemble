# 🩺 Kaggle Playground Series S6E7: Breaking the 0.950 Balanced Accuracy Barrier

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-brightgreen.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-yellow.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-GBDT-red.svg)
![Optuna](https://img.shields.io/badge/Optuna-Optimization-blueviolet.svg)

An end-to-end, competition-grade machine learning pipeline built for the **Kaggle Playground Series S6E7** (`health_condition` multi-class classification). Designed specifically to overcome severe class imbalance and break through the 0.949 public leaderboard ceiling on **Balanced Accuracy**.

---

## 📌 Problem Statement

The target variable `health_condition` exhibits extreme class imbalance:
* **`at-risk`**: ~86%
* **`unhealthy`**: ~8%
* **`fit`**: ~6%

Because the competition is evaluated on **Balanced Accuracy**, standard out-of-the-box pipelines and naive imputation strategies heavily favor the majority class and fail to generalize on minority classes.

---

## 🚀 Pipeline Architecture & Key Methodologies

### 1. Zero Imputation (Native Missing Value Routing)
* **Numerical Features** (`sleep_duration`, `heart_rate`, `bmi`, `calorie_expenditure`, `step_count`, `exercise_duration`, `water_intake`): Left completely untouched (`NaN` preserved). Modern GBDTs natively route missing values to the optimal split branch, preserving the predictive signal embedded in missingness patterns.
* **Categorical Features** (`diet_type`, `stress_level`, `sleep_quality`, `physical_activity_level`, `smoking_alcohol`, `gender`): Missing entries are isolated into an explicit `"Missing"` category and encoded using `OrdinalEncoder`.

### 2. Cross-Fitted Exact-Value Target Encoding
* Because synthetic wearable-sensor features contain discrete, repeating numerical values, standard binning loses granular risk spikes.
* Multi-class `TargetEncoder` (`smooth='auto'`, `cv=7`) is applied directly to exact numerical values, transforming raw measurements into leak-free class probability representations.

### 3. 7-Fold Stratified Cross-Validation & GBDT Ensemble
* Uses `StratifiedKFold(n_splits=7)` to strictly preserve the ~15:1 class ratio across every validation fold.
* Trains three diverse gradient-boosted architectures (`n_estimators=1200`, `learning_rate=0.04`, `early_stopping=50`):
  * **LightGBM** (`class_weight='balanced'`)
  * **CatBoost** (`auto_class_weights='Balanced'`)
  * **XGBoost** (`tree_method='hist'`)

### 4. Mathematical Prior-Correction & Optuna Blending
* **Prior-Correction**: Even with balanced class weights, tree models over-predict the majority class. Predicted probabilities are scaled against dataset class priors:
  $$P_{\text{corrected}} = \frac{P}{\text{priors}^{\beta}}$$
  The optimal exponent $\beta \in [0.0, 2.5]$ is grid-searched exclusively on Out-Of-Fold (OOF) predictions for each model to maximize Balanced Accuracy.
* **Optuna Probability Blending**: Runs 500 Optuna trials on the prior-corrected OOF probabilities to mathematically derive the optimal normalized ensemble weights (`w_lgb`, `w_cat`, `w_xgb`) before generating final test predictions.

---

## 🛠️ Tech Stack

* **Language**: Python 3.12
* **Data Processing**: `numpy`, `pandas`, `scikit-learn`
* **Modeling**: `lightgbm`, `catboost`, `xgboost`
* **Optimization**: `optuna`

---

## 📂 Repository Structure

```text
├── notebooks/
│   └── s6e7_complete_pipeline.ipynb   # Full end-to-end training & inference notebook
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
