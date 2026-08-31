import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="Customer Churn Predictor", page_icon="\U0001F4B3", layout="wide")


@st.cache_resource
def load_model():
    pipeline = joblib.load(HERE / "model.joblib")
    metadata = json.loads((HERE / "model_metadata.json").read_text())
    return pipeline, metadata


def risk_band(probability, thresholds):
    if probability >= thresholds["high"]:
        return "High", "\U0001F534"
    if probability >= thresholds["medium"]:
        return "Medium", "\U0001F7E0"
    return "Low", "\U0001F7E2"


def render_inputs(metadata):
    values = {}
    numeric_labels = {
        "Age_clean": "Age",
        "Dependent_count": "Dependents",
        "Total_Relationship_Count": "Products held",
        "Months_Inactive_12_mon": "Months inactive (last 12)",
        "Contacts_Count_12_mon": "Service contacts (last 12)",
        "Credit_Limit_clean": "Credit limit",
        "Total_Revolving_Bal": "Revolving balance",
        "Avg_Utilization_Ratio": "Utilization ratio",
        "Total_Amt_Chng_Q4_Q1": "Transaction amount change (Q4 vs Q1)",
        "Total_Trans_Amt": "Total transaction amount",
        "Total_Trans_Ct": "Total transaction count",
        "Total_Ct_Chng_Q4_Q1": "Transaction count change (Q4 vs Q1)",
    }
    categorical_labels = {
        "Education_Level_clean": "Education level",
        "Marital_Status_clean": "Marital status",
        "Income_Category_clean": "Income category",
        "Card_Category_clean": "Card category",
    }

    st.subheader("Demographic")
    col1, col2 = st.columns(2)
    with col1:
        values["Age_clean"] = number_input("Age_clean", numeric_labels, metadata)
        values["Dependent_count"] = number_input("Dependent_count", numeric_labels, metadata)
    with col2:
        values["Education_Level_clean"] = select_input("Education_Level_clean", categorical_labels, metadata)
        values["Marital_Status_clean"] = select_input("Marital_Status_clean", categorical_labels, metadata)

    st.subheader("Financial")
    col1, col2 = st.columns(2)
    with col1:
        values["Income_Category_clean"] = select_input("Income_Category_clean", categorical_labels, metadata)
        values["Card_Category_clean"] = select_input("Card_Category_clean", categorical_labels, metadata)
        values["Credit_Limit_clean"] = number_input("Credit_Limit_clean", numeric_labels, metadata)
        values["Total_Revolving_Bal"] = number_input("Total_Revolving_Bal", numeric_labels, metadata)
    with col2:
        values["Avg_Utilization_Ratio"] = number_input("Avg_Utilization_Ratio", numeric_labels, metadata)
        values["Total_Trans_Amt"] = number_input("Total_Trans_Amt", numeric_labels, metadata)
        values["Total_Trans_Ct"] = number_input("Total_Trans_Ct", numeric_labels, metadata)
        values["Total_Amt_Chng_Q4_Q1"] = number_input("Total_Amt_Chng_Q4_Q1", numeric_labels, metadata)

    st.subheader("Behavioral")
    col1, col2 = st.columns(2)
    with col1:
        values["Total_Relationship_Count"] = number_input("Total_Relationship_Count", numeric_labels, metadata)
        values["Months_Inactive_12_mon"] = number_input("Months_Inactive_12_mon", numeric_labels, metadata)
    with col2:
        values["Contacts_Count_12_mon"] = number_input("Contacts_Count_12_mon", numeric_labels, metadata)
        values["Total_Ct_Chng_Q4_Q1"] = number_input("Total_Ct_Chng_Q4_Q1", numeric_labels, metadata)

    return values


def number_input(feature, labels, metadata):
    spec = metadata["numeric_features"][feature]
    is_integer = spec["step"] == 1.0
    return st.number_input(
        labels[feature],
        min_value=(int(spec["min"]) if is_integer else float(spec["min"])),
        max_value=(int(spec["max"]) if is_integer else float(spec["max"])),
        value=(int(spec["median"]) if is_integer else float(spec["median"])),
        step=(1 if is_integer else float(spec["step"])),
        key=feature,
    )


def select_input(feature, labels, metadata):
    spec = metadata["categorical_features"][feature]
    return st.selectbox(
        labels[feature],
        options=spec["options"],
        index=spec["options"].index(spec["default"]),
        key=feature,
    )


def main():
    pipeline, metadata = load_model()

    st.title("Customer Churn Predictor")

    tab_predict, tab_about = st.tabs(["Predict", "Model info"])

    with tab_predict:
        left, right = st.columns([3, 2])
        with left:
            with st.form("predict_form"):
                values = render_inputs(metadata)
                predict_clicked = st.form_submit_button("Predict")
        with right:
            if predict_clicked:
                input_row = pd.DataFrame([values])
                st.session_state["probability"] = float(pipeline.predict_proba(input_row)[0, 1])

            if "probability" in st.session_state:
                probability = st.session_state["probability"]
                band, emoji = risk_band(probability, metadata["risk_thresholds"])
                st.metric("Churn probability", f"{probability:.1%}")
                st.markdown(f"### {emoji} {band} risk")
                st.progress(min(probability, 1.0))
                st.caption("High = top 10% of scores, Medium = next 20%.")
            else:
                st.caption("Fill in the fields and click Predict.")

    with tab_about:
        col1, col2, col3 = st.columns(3)
        col1.metric("ROC-AUC", "0.996")
        col2.metric("PR-AUC", "0.982")
        col3.metric("Training churn rate", f"{metadata['churn_rate']:.1%}")

        st.write(
            "Trained on {:,} labelled customers using a Histogram Gradient "
            "Boosting classifier.".format(metadata["training_rows"])
        )
        st.write(
            "A lot of the signal here comes from transaction and balance "
            "activity, which can already reflect someone who's checked out "
            "rather than an early warning sign — so treat this as a demo, "
            "not a production score."
        )


if __name__ == "__main__":
    main()
