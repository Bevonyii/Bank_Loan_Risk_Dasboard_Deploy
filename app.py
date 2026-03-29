import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Loan Risk & Segmentation Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main {
        padding-top: 1.2rem;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        padding: 1.5rem 1.75rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 16px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }
    .small-note {
        color: #475569;
        font-size: 0.93rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_data():
    df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
    return df

@st.cache_resource
def train_assets(df: pd.DataFrame):
    data = df.copy()

    feature_cols = [
        "Age", "Experience", "Income", "ZIP Code", "Family", "CCAvg",
        "Education", "Mortgage", "Securities Account", "CD Account",
        "Online", "CreditCard"
    ]

    X = data[feature_cols]
    y = data["Personal Loan"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cluster_cols = [
        "Age", "Income", "Family", "CCAvg", "Mortgage", "Education",
        "Securities Account", "CD Account", "Online", "CreditCard"
    ]

    cluster_scaler = StandardScaler()
    X_cluster_scaled = cluster_scaler.fit_transform(data[cluster_cols])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_cluster_scaled)

    out = data.copy()
    out["Cluster"] = clusters
    out["PCA1"] = coords[:, 0]
    out["PCA2"] = coords[:, 1]

    cluster_summary = out.groupby("Cluster")[["Age", "Income", "Family", "CCAvg", "Mortgage", "Personal Loan"]].mean().round(2)

    cluster_names = {}
    for c, row in cluster_summary.iterrows():
        if row["Income"] >= cluster_summary["Income"].quantile(0.67):
            cluster_names[c] = "High-Value Professionals"
        elif row["Family"] >= cluster_summary["Family"].quantile(0.67):
            cluster_names[c] = "Family-Focused Customers"
        else:
            cluster_names[c] = "Everyday Banking Customers"

    out["Cluster Label"] = out["Cluster"].map(cluster_names)
    cluster_summary["Cluster Label"] = cluster_summary.index.map(cluster_names)

    return {
        "clf": clf,
        "acc": acc,
        "feature_cols": feature_cols,
        "cluster_cols": cluster_cols,
        "cluster_scaler": cluster_scaler,
        "kmeans": kmeans,
        "df": out,
        "cluster_summary": cluster_summary,
    }


def predict_cluster(input_df: pd.DataFrame, assets):
    scaled = assets["cluster_scaler"].transform(input_df[assets["cluster_cols"]])
    cluster = assets["kmeans"].predict(scaled)[0]
    label = assets["cluster_summary"].loc[cluster, "Cluster Label"]
    return cluster, label


def recommendation(probability: float, cluster_label: str):
    if probability >= 0.7:
        rec = "Prioritize this customer for a targeted personal-loan campaign with premium messaging."
    elif probability >= 0.4:
        rec = "Use a softer cross-sell approach with educational messaging and a pre-qualified offer."
    else:
        rec = "Focus on relationship-building products first before pushing a personal loan."

    if cluster_label == "High-Value Professionals":
        rec += " This segment responds well to premium financial offers and convenience-focused messaging."
    elif cluster_label == "Family-Focused Customers":
        rec += " Family-oriented benefits and flexible repayment options may be effective here."
    else:
        rec += " Simpler offers and digital-first communication are likely a better fit."
    return rec


df = load_data()
assets = train_assets(df)
model_df = assets["df"]

st.markdown(
    """
    <div class='hero'>
        <h1 style='margin:0;'>Bank Loan Risk & Customer Segmentation Dashboard</h1>
        <p style='margin:0.4rem 0 0 0;'>An interactive Streamlit app for loan prediction, customer clustering, and business recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Customer Inputs")
    age = st.slider("Age", 20, 70, 35)
    experience = st.slider("Experience", 0, 45, 10)
    income = st.slider("Income ($000)", 5, 250, 80)
    zip_code = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=94112)
    family = st.selectbox("Family Size", [1, 2, 3, 4], index=1)
    ccavg = st.slider("CCAvg", 0.0, 10.0, 2.0, 0.1)
    education = st.selectbox("Education", [1, 2, 3], format_func=lambda x: {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}[x])
    mortgage = st.slider("Mortgage", 0, 700, 50)
    securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cd = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    credit = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    run_pred = st.button("Predict Customer Outcome", use_container_width=True)


input_df = pd.DataFrame([{
    "Age": age,
    "Experience": experience,
    "Income": income,
    "ZIP Code": zip_code,
    "Family": family,
    "CCAvg": ccavg,
    "Education": education,
    "Mortgage": mortgage,
    "Securities Account": securities,
    "CD Account": cd,
    "Online": online,
    "CreditCard": credit,
}])

col1, col2, col3, col4 = st.columns(4, gap="large")
col1.metric("Customers", f"{len(model_df):,}")
col2.metric("Loan Acceptance Rate", f"{model_df['Personal Loan'].mean()*100:.1f}%")
col3.metric("Avg Income", f"${model_df['Income'].mean():.1f}K")
col4.metric("Model Accuracy", f"{assets['acc']*100:.1f}%")

st.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #e6e6e6; margin-top: 20px; margin-bottom: 30px;">
    """, 
    unsafe_allow_html=True)

st.subheader("Project Overview")
st.write("""
This dashboard combines supervised learning for personal loan prediction with unsupervised learning for customer segmentation. 

Key metrics and cluster summaries provide insights across the full dataset, while the interactive section allows users to simulate individual customers and observe how their financial profile affects loan acceptance probability and segment classification.

This approach demonstrates how predictive modeling and segmentation can be used together to support targeted marketing and data-driven financial decisions.
""")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")
with left:

    fig, ax = plt.subplots(figsize=(7, 4))
    loan_rate = model_df.groupby("Education")["Personal Loan"].mean().sort_index()
    ax.bar(["Undergrad", "Graduate", "Advanced"], loan_rate.values)
    ax.set_title("Loan Acceptance Rate by Education")
    ax.set_ylabel("Acceptance Rate")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #e6e6e6; margin-top: 20px; margin-bottom: 30px;">
    """, 
    unsafe_allow_html=True)
    
    st.subheader("Interactive Prediction")

    if run_pred:
        pred = assets["clf"].predict(input_df[assets["feature_cols"]])[0]
        prob = assets["clf"].predict_proba(input_df[assets["feature_cols"]])[0][1]
        cluster_num, cluster_label = predict_cluster(input_df, assets)

        label_map = {
            "High-Value Professionals": "High-Value",
            "Family-Focused Customers": "Family-Focused",
            "Everyday Banking Customers": "Everyday Banking"
        }
        cluster_label = label_map.get(cluster_label, cluster_label)

        outcome = "Likely" if pred == 1 else "Unlikely"
        
        r1, r2, r3 = st.columns([1, 1, 1], gap="large")
        with r1:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:16px; color:#cbd5e1; margin-bottom:8px;">Loan Outcome</div>
                <div style="font-size:30px; font-weight:700; color:white;">{outcome}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:16px; color:#cbd5e1; margin-bottom:8px;">Loan Probability</div>
                <div style="font-size:30px; font-weight:700; color:white;">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="font-size:16px; color:#cbd5e1; margin-bottom:8px;">Customer Segment</div>
                <div style="font-size:22px; font-weight:700; color:white;">{cluster_label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(max(float(prob), 0.0), 1.0))
        st.caption(f"Cluster {cluster_num}: {cluster_label}")

        st.info(recommendation(prob, cluster_label))
    else:
        st.write("Use the sidebar to enter customer details, then click **Predict Customer Outcome**.")

with right:
    st.subheader("Cluster Summary")
    show_summary = assets["cluster_summary"].copy()
    st.dataframe(show_summary, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #e6e6e6; margin-top: 20px; margin-bottom: 30px;">
    """, 
    unsafe_allow_html=True)
    
    st.subheader("Cluster View")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for c in sorted(model_df["Cluster"].unique()):
        temp = model_df[model_df["Cluster"] == c]
        ax2.scatter(temp["PCA1"], temp["PCA2"], alpha=0.6, label=assets["cluster_summary"].loc[c, "Cluster Label"])
    ax2.set_title("Customer Segments (PCA View)")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p class='small-note'>Built with Streamlit, scikit-learn, pandas, and matplotlib.</p>", unsafe_allow_html=True)
