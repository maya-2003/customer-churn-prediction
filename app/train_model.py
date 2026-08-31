"""
Trains the churn model exactly as selected in customer_churn_project.ipynb
(Histogram Gradient Boosting, uncalibrated) and saves it as a portable
artifact the Streamlit app can load without re-running the notebook.

Run once from this folder:
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
DATA_FILE = HERE / "credit-card_customers.xlsx"
SHEET_NAME = "List"
RANDOM_STATE = 42

MODEL_NUMERIC_FEATURES = [
    "Age_clean", "Dependent_count",
    "Total_Relationship_Count", "Months_Inactive_12_mon",
    "Contacts_Count_12_mon", "Credit_Limit_clean",
    "Total_Revolving_Bal", "Avg_Utilization_Ratio",
    "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
    "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1",
]
MODEL_CATEGORICAL_FEATURES = [
    "Education_Level_clean", "Marital_Status_clean",
    "Income_Category_clean", "Card_Category_clean",
]
MODEL_FEATURES = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES


def load_and_clean():
    raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME, dtype={"Customer_Number": "string"}, na_values=["N/A"])
    base = raw.drop_duplicates().copy()

    age_out_of_range = base["Age"].notna() & ~base["Age"].between(18, 100)
    base["Age_clean"] = base["Age"].mask(age_out_of_range)

    nonpositive_credit_limit = base["Credit_Limit"].notna() & base["Credit_Limit"].le(0)
    base["Credit_Limit_clean"] = base["Credit_Limit"].copy()
    base.loc[nonpositive_credit_limit, "Credit_Limit_clean"] = (
        base["Avg_Open_To_Buy"] + base["Total_Revolving_Bal"]
    )

    for column in ["Education_Level", "Marital_Status", "Income_Category", "Card_Category"]:
        base[f"{column}_clean"] = base[column].fillna("Missing").astype("category")

    valid_target = base["Attrition_Flag"].isin(["Existing Customer", "Attrited Customer"])
    base["Churned"] = base["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1,
    }).astype("Int64")

    return base.loc[valid_target].copy()


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, MODEL_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=20, random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def build_metadata(modeling_base, scores):
    metadata = {"numeric_features": {}, "categorical_features": {}}
    for column in MODEL_NUMERIC_FEATURES:
        series = modeling_base[column].dropna()
        metadata["numeric_features"][column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "step": 1.0 if pd.api.types.is_integer_dtype(modeling_base[column].dtype) else round(
                float((series.max() - series.min()) / 100) or 0.01, 4
            ),
        }
    for column in MODEL_CATEGORICAL_FEATURES:
        counts = modeling_base[column].value_counts()
        metadata["categorical_features"][column] = {
            "options": counts.index.tolist(),
            "default": counts.index[0],
        }
    metadata["risk_thresholds"] = {
        "high": float(np.quantile(scores, 0.90)),
        "medium": float(np.quantile(scores, 0.70)),
    }
    metadata["training_rows"] = int(len(modeling_base))
    metadata["churn_rate"] = float(modeling_base["Churned"].mean())
    return metadata


def main():
    modeling_base = load_and_clean()
    X = modeling_base[MODEL_FEATURES]
    y = modeling_base["Churned"].astype(int)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    in_sample_scores = pipeline.predict_proba(X)[:, 1]
    metadata = build_metadata(modeling_base, in_sample_scores)

    joblib.dump(pipeline, HERE / "model.joblib")
    (HERE / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Trained on {len(X):,} labelled customers, churn rate {metadata['churn_rate']:.2%}")
    print(f"Saved model.joblib and model_metadata.json to {HERE}")


if __name__ == "__main__":
    main()
