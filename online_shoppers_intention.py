import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Page configuration
st.set_page_config(
    page_title="Real-Time E-Commerce Purchase Intent Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Real-Time E-Commerce Purchase Intent Prediction")
st.markdown(
    "Predict user purchase intent in real time based on live clickstream behavior, "
    "session metrics, and visitor attributes."
)

# Load dataset or train a fallback model if saved model is unavailable
@st.cache_resource
def load_or_train_model():
    model_path = "purchase_intent_model.joblib"
    data_path = "online_shoppers_intention.csv"

    if os.path.exists(model_path):
        return joblib.load(model_path), None

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df.dropna()

        le_month = LabelEncoder()
        le_visitor = LabelEncoder()

        df["Month_enc"] = le_month.fit_transform(df["Month"].astype(str))
        df["VisitorType_enc"] = le_visitor.fit_transform(df["VisitorType"].astype(str))
        df["Weekend_enc"] = df["Weekend"].astype(int)

        feature_cols = [
            "Administrative", "Administrative_Duration",
            "Informational", "Informational_Duration",
            "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues",
            "SpecialDay", "Month_enc", "OperatingSystems",
            "Browser", "Region", "TrafficType",
            "VisitorType_enc", "Weekend_enc"
        ]

        X = df[feature_cols]
        y = df["Revenue"].astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(X, y)

        encoders = {"Month": le_month, "VisitorType": le_visitor}
        joblib.dump((model, encoders), model_path)
        return (model, encoders), df

    return None, None


loaded, sample_df = load_or_train_model()

# Sidebar inputs for live session simulation
st.sidebar.header("📊 Live Session Parameters")

admin_pages = st.sidebar.number_input("Administrative Pages Visited", min_value=0, max_value=50, value=2)
admin_dur = st.sidebar.number_input("Administrative Duration (sec)", min_value=0.0, max_value=5000.0, value=45.0)

info_pages = st.sidebar.number_input("Informational Pages Visited", min_value=0, max_value=50, value=0)
info_dur = st.sidebar.number_input("Informational Duration (sec)", min_value=0.0, max_value=5000.0, value=0.0)

prod_pages = st.sidebar.number_input("Product-Related Pages Visited", min_value=0, max_value=500, value=15)
prod_dur = st.sidebar.number_input("Product-Related Duration (sec)", min_value=0.0, max_value=20000.0, value=450.0)

bounce_rate = st.sidebar.slider("Bounce Rate", min_value=0.0, max_value=0.2, value=0.02, step=0.005)
exit_rate = st.sidebar.slider("Exit Rate", min_value=0.0, max_value=0.2, value=0.03, step=0.005)
page_values = st.sidebar.number_input("Page Values ($)", min_value=0.0, max_value=500.0, value=12.5)
special_day = st.sidebar.slider("Special Day Closeness", min_value=0.0, max_value=1.0, value=0.0, step=0.2)

month = st.sidebar.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=8)
visitor_type = st.sidebar.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
weekend = st.sidebar.checkbox("Weekend Session", value=False)

os_type = st.sidebar.number_input("Operating System ID", min_value=1, max_value=8, value=2)
browser = st.sidebar.number_input("Browser ID", min_value=1, max_value=13, value=2)
region = st.sidebar.number_input("Region ID", min_value=1, max_value=9, value=1)
traffic_type = st.sidebar.number_input("Traffic Type ID", min_value=1, max_value=20, value=2)

# Main Dashboard Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🔍 Current Session Overview")
    session_summary = pd.DataFrame({
        "Metric": [
            "Product Pages Viewed",
            "Product Time Spent (s)",
            "Page Values ($)",
            "Bounce Rate",
            "Exit Rate",
            "Visitor Type",
            "Month"
        ],
        "Value": [
            prod_pages,
            f"{prod_dur:.1f}s",
            f"${page_values:.2f}",
            f"{bounce_rate:.3f}",
            f"{exit_rate:.3f}",
            visitor_type,
            month
        ]
    })
    st.table(session_summary)

with col2:
    st.subheader("🎯 Real-Time Intent Prediction")

    if loaded is not None:
        model, encoders = loaded
        try:
            month_enc = encoders["Month"].transform([month])[0]
        except Exception:
            month_enc = 0
        try:
            visitor_enc = encoders["VisitorType"].transform([visitor_type])[0]
        except Exception:
            visitor_enc = 0

        input_features = np.array([[
            admin_pages, admin_dur,
            info_pages, info_dur,
            prod_pages, prod_dur,
            bounce_rate, exit_rate, page_values,
            special_day, month_enc, os_type,
            browser, region, traffic_type,
            visitor_enc, int(weekend)
        ]])

        prob = model.predict_proba(input_features)[0][1]
        pred = int(prob >= 0.5)

        st.metric(label="Purchase Probability", value=f"{prob * 100:.1f}%")
        st.progress(float(prob))

        if pred == 1:
            st.success("✅ **High Purchase Intent Detected!** Recommend live checkout assist or loyalty perk.")
        else:
            st.warning("⚠️ **Low Purchase Intent.** Consider triggering a targeted discount or product recommendation.")
    else:
        # Heuristic fallback if CSV/model is not present in directory
        heuristic_score = min(1.0, max(0.0, (page_values * 0.03) + (prod_pages * 0.01) - (bounce_rate * 2.0)))
        st.metric(label="Estimated Intent Score (Heuristic Mode)", value=f"{heuristic_score * 100:.1f}%")
        st.progress(float(heuristic_score))
        st.info("ℹ️ Place `online_shoppers_intention.csv` in the repository root to activate full Random Forest model training.")

if sample_df is not None:
    st.markdown("---")
    st.subheader("📈 Dataset Insights (`online_shoppers_intention.csv`)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sessions", f"{len(sample_df):,}")
    c2.metric("Overall Conversion Rate", f"{sample_df['Revenue'].mean() * 100:.2f}%")
    c3.metric("Avg Page Value (Buyers)", f"${sample_df[sample_df['Revenue'] == True]['PageValues'].mean():.2f}")
