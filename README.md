# Customer Churn Prediction

End‑to‑end ML pipeline to predict whether a customer will churn using tabular behavioral & account data. This repo contains a single notebook and the raw dataset to reproduce results and iterate quickly.

> **Files:**
>
> * `Customer_churn_prediction.ipynb` — full workflow (EDA → preprocessing → modeling → evaluation)
> * `Data_Science_Challenge.csv` — source dataset (tabular)

---

## 1) Problem & Objective

Businesses lose revenue when customers stop using their product or service (churn). The goal is to build a supervised classifier that predicts churn probability per customer so retention teams can target at‑risk users.

**Target variable:** `churn` (binary: 1 = churned, 0 = active).
**Prediction:** Probability of churn within the defined horizon (per dataset).

---

## 2) High‑Level Pipeline

1. **Data ingestion**: Load CSV → inspect schema, dtypes, missingness, target distribution.
2. **EDA**: Univariate & bivariate analysis, class imbalance check, correlations/associations, leakage checks.
3. **Preprocessing** (scikit‑learn `ColumnTransformer`):

   * Missing values: median for numeric, most‑frequent for categoricals.
   * Encoding: one‑hot for categoricals (`handle_unknown='ignore'`).
   * Scaling: standardize numeric features (inside the pipeline; avoids leakage).
   * Optional: outlier winsorization / log transforms for skewed features.
4. **Imbalance handling** (if needed): `class_weight='balanced'` and/or `SMOTE` (via `imblearn.Pipeline`).
5. **Modeling** (wrapped in sklearn `Pipeline`):

   * Baselines: Logistic Regression.
   * Tree‑based: Random Forest, Gradient Boosting, LightGBM, XGBoost.
   * Cross‑validation: Stratified K‑Fold.
6. **Hyperparameter tuning**: `RandomizedSearchCV` / `GridSearchCV` optimized primarily on ROC‑AUC; also track F1/Recall.
7. **Evaluation**: Confusion matrix, ROC‑AUC, PR curve, classification report, calibration check.
8. **Interpretability**: Feature importances (trees), permutation importances; optional SHAP summary for final model.
9. **Threshold tuning**: Pick operating threshold to maximize business objective (e.g., Recall @ fixed Precision or F1).
10. **Export** (optional): Persist fitted `model.pkl` and `preprocessor.pkl` with `joblib` for downstream use.

---

## 3) What’s implemented in this repo

* A single Jupyter notebook driving the **entire workflow** (EDA → preprocessing → baseline & advanced models → evaluation).
* Clean separation of **preprocessing inside the model pipeline** to avoid data leakage.
* **Stratified train/validation split** for fair comparison.
* **Multiple algorithms compared** (logistic baseline + tree‑based), with **hyperparameter tuning** for LightGBM.
* **Comprehensive evaluation**: ROC‑AUC, F1, precision/recall, confusion matrix; plots for ROC/PR and feature importance.
* **Business‑aware thresholding** notes to align predictions with retention capacity.

> Tip: You can export the final model & preprocessor using `joblib.dump` for API/Streamlit use. See code snippet below.

---

## 4) Results

Best parameters found (LightGBM): `{'num_leaves': 40, 'n_estimators': 500, 'learning_rate': 0.05}`

| Metric    | Validation |
| --------- | ---------- |
| Accuracy  | **0.939**  |
| Precision | **0.878**  |
| Recall    | **0.670**  |
| F1‑score  | **0.760**  |
| ROC‑AUC   | **0.901**  |

**Confusion Matrix (at chosen threshold):**

```
TN = 561   FP = 9
FN = 32    TP = 65
```

**Top drivers** (from importances/coefficients): contract type, tenure, monthly charges, and payment method.

---

## 8) Notes on Modeling Choices

* **Why Logistic Regression?** Strong baseline, interpretable coefficients, fast to train.
* **Why Tree‑based models?** Capture non‑linearities & interactions; deliver feature importances.
* **Why LightGBM?** Efficient gradient boosting, strong performance on tabular data.
* **Why put preprocessing in the pipeline?** Guarantees identical transforms in CV & at inference.
* **Why ROC‑AUC + PR?** ROC‑AUC is threshold‑independent; PR is more informative under imbalance.
* **Threshold tuning** helps match precision/recall trade‑offs to retention capacity & cost of outreach.

---

## 9) Limitations & Next Steps

* Try **calibration** (Platt/Isotonic) if probabilities are over/under‑confident.
* Evaluate **cost‑sensitive metrics** (Expected Profit / Uplift) aligned with business goals.
* Add **model monitoring**: data drift (PSI), performance drift, alerting.
* Ship a lightweight **Streamlit** app or **FastAPI** for scoring.
* Log experiments & artifacts with **MLflow**.

---

### Author

**Saransh Dhage**
