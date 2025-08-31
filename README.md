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
   * Tree‑based: Random Forest, Gradient Boosting / XGBoost (optional).
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
* **Multiple algorithms compared** (logistic baseline + tree‑based), with basic **hyperparameter tuning**.
* **Comprehensive evaluation**: ROC‑AUC, F1, precision/recall, confusion matrix; plots for ROC/PR and feature importance.
* **Business‑aware thresholding** notes to align predictions with retention capacity.

> Tip: You can export the final model & preprocessor using `joblib.dump` for API/Streamlit use. See code snippet below.

---

## 4) Results (replace with your latest numbers)

| Metric    | Validation |
| --------- | ---------- |
| ROC‑AUC   | `0.XX`     |
| F1‑score  | `0.XX`     |
| Precision | `0.XX`     |
| Recall    | `0.XX`     |
| Accuracy  | `0.XX`     |

**Confusion Matrix (at chosen threshold):**

```
TN = ____   FP = ____
FN = ____   TP = ____
```

**Top drivers** (from importances/coefficients): `feature_1`, `feature_2`, `feature_3` …

> Replace with your actual insights from the notebook (e.g., contract type, tenure, charges).

---

## 5) Reproduce Locally

### Prerequisites

* Python ≥ 3.10
* Recommended packages:

```
pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost shap joblib jupyter
```

### Steps

```bash
# 1) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -U pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost shap joblib jupyter

# 3) Launch Jupyter and run the notebook cells in order
jupyter notebook
```

---

## 6) Inference Example (after training)

```python
import joblib
import pandas as pd

# Load artifacts
preprocessor = joblib.load("artifacts/preprocessor.pkl")  # if you saved it
model = joblib.load("artifacts/churn_model.pkl")          # if you saved it

# Example customer (replace with real feature names)
row = {
    "age": 34,
    "tenure_months": 5,
    "monthly_charges": 89.5,
    "contract_type": "Month-to-month",
    "is_paperless_billing": 1,
    # ... all required columns
}
X = pd.DataFrame([row])
Xp = preprocessor.transform(X)
proba = model.predict_proba(Xp)[:, 1][0]
label = int(proba >= 0.40)  # example threshold tuned to business needs
print({"churn_probability": round(proba, 3), "predicted_label": label})
```

---

## 7) Project Structure (suggested)

```
Customer-Churn-Prediction/
├─ Customer_churn_prediction.ipynb
├─ Data_Science_Challenge.csv
├─ artifacts/                 # (optional) saved model/preprocessor
├─ notebooks/                 # (optional) EDA or experiments
├─ src/                       # (optional) reusable pipeline code
│  ├─ data.py                 # loaders, splitters
│  ├─ preprocess.py           # ColumnTransformer builders
│  ├─ train.py                # train/tune/evaluate
│  └─ explain.py              # permutation/SHAP helpers
├─ requirements.txt           # (optional) pinned deps
└─ README.md
```

---

## 8) Notes on Modeling Choices

* **Why Logistic Regression?** Strong baseline, interpretable coefficients, fast to train.
* **Why Tree‑based models?** Capture non‑linearities & interactions; deliver feature importances.
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

## 10) License

This project is open‑sourced under the MIT License. See `LICENSE` (add if missing).

---

## 11) Acknowledgements

* scikit‑learn, imbalanced‑learn, XGBoost, SHAP communities.
* Dataset provided via `Data_Science_Challenge.csv` in this repository.

---

### Maintainer

**Saransh Dhage** — feel free to open an issue or reach out for collaboration.
