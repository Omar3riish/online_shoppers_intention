import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Page Configuration & Constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time E-Commerce Purchase Intent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
DATA_PATH = Path("online_shoppers_intention.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

CONTINUOUS_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Nominal integer codes + categoricals identified in your Data Quality Audit
CATEGORICAL_COLS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# 2. Data Loading & Deduplication
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        raw_df = pd.read_csv(DATA_PATH)
    else:
        return None, None

    # Ensure nominal codes are treated consistently as strings/categories
    df_clean = raw_df.drop_duplicates().reset_index(drop=True)
    return raw_df, df_clean


# ---------------------------------------------------------------------------
# 3. Model Training Pipeline (with PageValues Ablation Toggle)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_models(df_clean: pd.DataFrame, include_page_values: bool = True):
    num_features = [c for c in CONTINUOUS_COLS if include_page_values or c != "PageValues"]
    cat_features = CATEGORICAL_COLS.copy()
    all_features = num_features + cat_features

    X = df_clean[all_features].copy()
    for col in cat_features:
        X[col] = X[col].astype(str)
    y = df_clean["Revenue"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
        ]
    )

    candidates = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results[name] = {
            "pipeline": pipe,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

    # Save best artifact to disk matching your notebook structure
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    suffix = "full" if include_page_values else "no_pagevalues"
    joblib.dump(results[best_name]["pipeline"], ARTIFACT_DIR / f"model_{suffix}.joblib")

    return results, X_test, y_test


# ---------------------------------------------------------------------------
# 4. Sidebar Navigation & Dataset Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛒 Purchase Intent AI")
    st.caption("UCI Online Shoppers Intention Engine")

    uploaded_csv = st.file_uploader(
        "Upload `online_shoppers_intention.csv`",
        type=["csv"],
        help="Uses local `online_shoppers_intention.csv` automatically if present.",
    )

    raw_df, df_clean = load_and_clean_data(uploaded_csv)

    st.divider()
    page = st.radio(
        "Navigation",
        [
            "🎯 Real-Time Intent Predictor",
            "📊 EDA & Quality Audit",
            "⚙️ Model Benchmark & Ablation",
        ],
    )

    st.divider()
    include_pv = st.toggle(
        "Include `PageValues` Feature",
        value=True,
        help="Toggle off to run the domain leakage sensitivity ablation from Section 2.",
    )

if df_clean is None:
    st.error(
        "Dataset `online_shoppers_intention.csv` not found in the working directory. "
        "Please upload it via the sidebar to launch the app."
    )
    st.stop()

models_dict, X_test, y_test = train_models(df_clean, include_page_values=include_pv)


# ===========================================================================
# PAGE 1: REAL-TIME INTENT PREDICTOR
# ===========================================================================
if page == "🎯 Real-Time Intent Predictor":
    st.title("🎯 Will This Visitor Buy? Real-Time Intent Simulator")
    st.markdown(
        "Simulate an active browsing session to score purchase likelihood and trigger real-time retention actions."
    )

    col_cfg1, col_cfg2 = st.columns([1, 1])
    with col_cfg1:
        selected_model_name = st.selectbox(
            "Active Inference Model",
            list(models_dict.keys()),
            index=0,
        )
    with col_cfg2:
        decision_threshold = st.slider(
            "Conversion Action Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Lower threshold prioritizes Recall (catching more potential buyers); higher threshold prioritizes Precision.",
        )

    st.divider()

    # Session Input Form
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1. Browsing Funnel Depth")
        product_related = st.number_input("Product Pages Visited", min_value=0, max_value=750, value=18)
        product_duration = st.number_input(
            "Product Browsing Duration (sec)", min_value=0.0, max_value=70000.0, value=610.0, step=10.0
        )
        admin_pages = st.number_input("Administrative Pages (Account/Cart)", min_value=0, max_value=30, value=2)
        admin_duration = st.number_input(
            "Administrative Duration (sec)", min_value=0.0, max_value=4000.0, value=45.0, step=5.0
        )
        info_pages = st.number_input("Informational Pages", min_value=0, max_value=30, value=0)
        info_duration = st.number_input(
            "Informational Duration (sec)", min_value=0.0, max_value=3000.0, value=0.0, step=5.0
        )

    with c2:
        st.subheader("2. Friction & Value Signals")
        page_values = st.number_input(
            "PageValues ($)",
            min_value=0.0,
            max_value=400.0,
            value=12.5,
            step=1.0,
            disabled=not include_pv,
            help="Disabled when PageValues ablation mode is active in the sidebar.",
        )
        bounce_rates = st.slider("Average Bounce Rate", min_value=0.0, max_value=0.20, value=0.005, step=0.005, format="%.3f")
        exit_rates = st.slider("Average Exit Rate", min_value=0.0, max_value=0.20, value=0.025, step=0.005, format="%.3f")
        special_day = st.select_slider(
            "Special Day Proximity",
            options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            value=0.0,
            help="Closeness to major shopping holidays (e.g., Mother's Day, Valentine's Day).",
        )

    with c3:
        st.subheader("3. Visitor Context & Tech")
        visitor_type = st.selectbox(
            "Visitor Type",
            sorted(df_clean["VisitorType"].unique()),
            index=0,
        )
        month = st.selectbox(
            "Month",
            [m for m in MONTH_ORDER if m in df_clean["Month"].unique()],
            index=7,  # Nov
        )
        weekend = st.checkbox("Weekend Session", value=False)
        traffic_type = st.selectbox("Traffic Source ID", sorted(df_clean["TrafficType"].unique()), index=1)
        region = st.selectbox("Region ID", sorted(df_clean["Region"].unique()), index=0)
        os_id = st.selectbox("Operating System ID", sorted(df_clean["OperatingSystems"].unique()), index=1)
        browser_id = st.selectbox("Browser ID", sorted(df_clean["Browser"].unique()), index=1)

    # Assemble single-row dataframe matching training schema
    input_row = {
        "Administrative": admin_pages,
        "Administrative_Duration": admin_duration,
        "Informational": info_pages,
        "Informational_Duration": info_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": str(month),
        "OperatingSystems": str(os_id),
        "Browser": str(browser_id),
        "Region": str(region),
        "TrafficType": str(traffic_type),
        "VisitorType": str(visitor_type),
        "Weekend": str(weekend),
    }

    active_pipe = models_dict[selected_model_name]["pipeline"]
    expected_cols = active_pipe.feature_names_in_
    input_df = pd.DataFrame([input_row])[expected_cols]

    prob = float(active_pipe.predict_proba(input_df)[0, 1])
    will_buy = prob >= decision_threshold

    st.divider()
    res_col1, res_col2 = st.columns([1.1, 1.4])

    with res_col1:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 42}},
                delta={"reference": decision_threshold * 100, "suffix": "% vs threshold"},
                title={"text": "Predicted Purchase Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2E7D32" if will_buy else "#C62828"},
                    "steps": [
                        {"range": [0, decision_threshold * 100], "color": "#FFEBEE"},
                        {"range": [decision_threshold * 100, 100], "color": "#E8F5E9"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": decision_threshold * 100,
                    },
                },
            )
        )
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(gauge, use_container_width=True)

    with res_col2:
        st.subheader("Real-Time Decision & Action Playbook")
        if will_buy:
            st.success(f"**HIGH PURCHASE INTENT ({prob:.1%})** — Predicted Outcome: **Revenue = True**")
            st.markdown(
                """
                **Recommended Action:**
                - **Avoid Margin-Eroding Discounts:** Visitor already shows strong conversion momentum; suppress pop-up coupon codes.
                - **Frictionless Checkout:** Pre-fill express shipping options and surface stock availability badges.
                - **Cross-Sell:** Display high-margin complementary accessories in the mini-cart.
                """
            )
        else:
            st.warning(f"**LOW / AT-RISK INTENT ({prob:.1%})** — Predicted Outcome: **Revenue = False**")
            st.markdown(
                """
                **Recommended Intervention:**
                - **Exit-Intent Trigger:** Deploy a time-limited free-shipping or first-order incentive if `ExitRates` > 0.04.
                - **Social Proof:** Surface top-rated reviews and low-stock alerts on viewed product pages.
                - **Retargeting Pool:** Tag session for dynamic product retargeting if visitor exits without cart addition.
                """
            )


# ===========================================================================
# PAGE 2: EDA & QUALITY AUDIT
# ===========================================================================
elif page == "📊 EDA & Quality Audit":
    st.title("📊 Data Quality Audit & Exploratory Dashboard")

    dup_count = int(raw_df.duplicated().sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw Sessions", f"{len(raw_df):,}")
    m2.metric("Exact Duplicates Removed", f"{dup_count:,}", f"-{dup_count / len(raw_df):.2%}")
    m3.metric("Cleaned Sessions", f"{len(df_clean):,}")
    m4.metric("Purchase Rate (Revenue=True)", f"{df_clean['Revenue'].mean():.2%}")

    with st.expander("📋 View Data Quality & Leakage Audit Notes", expanded=False):
        st.markdown(
            """
            1. **Zero Missing Values:** All 18 columns are 100% populated.
            2. **Exact Duplicates (125 rows / 1.01%):** Nearly all duplicate rows represent identical single-page bounce sessions (`ProductRelated=1, Duration=0, BounceRates=0.20`). Removing exact duplicates prevents artificial inflation of majority-class patterns.
            3. **Nominal Integer Codes:** `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are integer-coded categorical identifiers—not ordinal quantities—and are one-hot encoded in the modeling pipeline.
            4. **Domain Leakage Sensitivity (`PageValues`):** Reflects average monetary value of pages visited prior to completing a goal. Use the sidebar toggle to inspect model behavior with and without `PageValues`.
            """
        )

    # Recreating your 2x2 Session & Conversion Analysis Dashboard
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Target Class Distribution (Severe Imbalance)",
            "PageValues by Purchase Outcome (Outliers Hidden)",
            "Conversion Rate (%) by Month",
            "Conversion Rate (%) by Visitor Type & Weekend",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )

    counts = df_clean["Revenue"].value_counts()
    x_labels = ["No Purchase (False)", "Purchase (True)"]
    y_vals = [counts.get(False, 0), counts.get(True, 0)]
    total_len = len(df_clean)

    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=y_vals,
            marker_color=["#4C72B0", "#DD8452"],
            width=0.55,
            text=[f"{c:,}<br>({c/total_len:.1%})" for c in y_vals],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for rev_val, color, name in zip([False, True], ["#4C72B0", "#DD8452"], ["False", "True"]):
        subset = df_clean[df_clean["Revenue"] == rev_val]
        fig.add_trace(
            go.Box(
                y=subset["PageValues"],
                name=name,
                marker_color=color,
                boxpoints=False,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    ordered_months = [m for m in MONTH_ORDER if m in df_clean["Month"].unique()]
    monthly = df_clean.groupby("Month", observed=False)["Revenue"].agg(["count", "mean"]).reindex(ordered_months)
    annual_avg = df_clean["Revenue"].mean() * 100

    fig.add_trace(
        go.Bar(
            x=monthly.index,
            y=monthly["mean"] * 100,
            marker_color="#55A868",
            opacity=0.85,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=annual_avg,
        line_dash="dash",
        line_color="#C44E52",
        line_width=1.3,
        annotation_text=f"Annual Avg ({annual_avg:.1f}%)",
        annotation_position="top right",
        row=2,
        col=1,
    )

    vt_wk = df_clean.groupby(["VisitorType", "Weekend"])["Revenue"].mean().unstack() * 100
    colors_vt = ["#4C72B0", "#8172B3"]
    labels_vt = ["Weekday", "Weekend"]

    for i, col_name in enumerate(vt_wk.columns):
        fig.add_trace(
            go.Bar(
                x=vt_wk.index,
                y=vt_wk[col_name],
                name=labels_vt[i],
                marker_color=colors_vt[i],
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=780,
        title_text="<b>Session and Conversion Analysis Dashboard</b>",
        title_x=0.5,
        template="plotly_white",
        barmode="group",
        legend=dict(title="Session Timing", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Number of Sessions", row=1, col=1)
    fig.update_yaxes(title_text="PageValues ($)", row=1, col=2)
    fig.update_yaxes(title_text="Purchase Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Purchase Rate (%)", row=2, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Interactive Continuous Feature Boxplot Explorer
    st.subheader("Behavioral Separation Explorer")
    selected_num_col = st.selectbox("Inspect Continuous Feature by Purchase Outcome", CONTINUOUS_COLS, index=6)
    box_fig = px.box(
        df_clean,
        x="Revenue",
        y=selected_num_col,
        color="Revenue",
        points=False,
        color_discrete_map={False: "#4C72B0", True: "#DD8452"},
        title=f"{selected_num_col} Distribution by Revenue (Outliers Hidden)",
    )
    box_fig.update_layout(template="plotly_white", height=400, showlegend=False)
    st.plotly_chart(box_fig, use_container_width=True)


# ===========================================================================
# PAGE 3: MODEL BENCHMARK & ABLATION
# ===========================================================================
elif page == "⚙️ Model Benchmark & Ablation":
    st.title("⚙️ Holdout Evaluation & `PageValues` Sensitivity Analysis")
    st.caption(
        f"Evaluating on a stratified 20% holdout set ({len(y_test):,} sessions) | "
        f"Mode: **{'Full Feature Set' if include_pv else 'Ablation (Without PageValues)'}**"
    )

    summary_rows = []
    for name, res in models_dict.items():
        summary_rows.append(
            {
                "Model": name,
                "PR-AUC (Avg Precision)": round(res["pr_auc"], 4),
                "ROC-AUC": round(res["roc_auc"], 4),
                "F1-Score (t=0.50)": round(res["f1"], 4),
                "Precision (t=0.50)": round(res["precision"], 4),
                "Recall (t=0.50)": round(res["recall"], 4),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    c_roc, c_pr = st.columns(2)

    with c_roc:
        roc_fig = go.Figure()
        for name, res in models_dict.items():
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['roc_auc']:.3f})"))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
        roc_fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with c_pr:
        pr_fig = go.Figure()
        baseline_rate = y_test.mean()
        for name, res in models_dict.items():
            prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
            pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (PR-AUC={res['pr_auc']:.3f})"))
        pr_fig.add_hline(
            y=baseline_rate,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline ({baseline_rate:.1%})",
        )
        pr_fig.update_layout(
            title="Precision-Recall Curve (Primary Imbalance Metric)",
            xaxis_title="Recall",
            yaxis_title="Precision",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(pr_fig, use_container_width=True)

    # Confusion Matrix for chosen model
    st.subheader("Holdout Confusion Matrix")
    cm_model = st.selectbox("Select Model for Confusion Matrix", list(models_dict.keys()))
    cm = confusion_matrix(y_test, models_dict[cm_model]["y_pred"])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted: No Purchase", "Predicted: Purchase"],
        y=["Actual: No Purchase", "Actual: Purchase"],
        color_continuous_scale="Blues",
        aspect="auto",
    )
    cm_fig.update_layout(height=360, coloraxis_showscale=False)
    st.plotly_chart(cm_fig, use_container_width=True)