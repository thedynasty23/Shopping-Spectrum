# app.py
# -------------------------------------------------------------------
# Shopper Spectrum Dashboard  –  Minimal white layout (Streamlit)
# -------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# -------------------------------------------------------------------
# 1.   Caching helpers
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_customer_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_resource(show_spinner=False)
def build_item_similarity(df: pd.DataFrame):
    rec_cols = [c for c in df.columns if c.startswith("Rec") and c.endswith("StockCode")]
    long_df  = pd.melt(df[["CustomerID"] + rec_cols],
                       id_vars="CustomerID",
                       value_vars=rec_cols,
                       value_name="StockCode").dropna()
    events   = long_df.drop_duplicates()

    cust_ids = events["CustomerID"].astype("category").cat.codes
    item_ids = events["StockCode" ].astype("category").cat.codes
    matrix   = csr_matrix((np.ones(len(events), dtype=np.int8),
                           (cust_ids, item_ids)))
    sim      = cosine_similarity(matrix.T, dense_output=False)
    labels   = events["StockCode"].astype("category").cat.categories
    return pd.DataFrame(sim.toarray(), index=labels, columns=labels)

@st.cache_resource(show_spinner=False)
def load_bundle(pkl_path: str):
    return joblib.load(pkl_path)      # returns {"scaler": ..., "model": ...}

# -------------------------------------------------------------------
# 2.   Global page setup  (→ white background)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# Make Streamlit’s built-in theme pure white
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: white;
        }
        [data-testid="stSidebar"] {
            background: #f8f9fa;          /* subtle grey for sidebar only */
            padding-top: 2rem;
        }
        h1, h2, h3 { color: #111 !important; }
        .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# 3.   Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.header("🛒 Shopper Spectrum")
choice = st.sidebar.radio("Go to", ("Product Recommendation", "Customer Segmentation"))

# -------------------------------------------------------------------
# 4A.  Product Recommendation
# -------------------------------------------------------------------
if choice == "Product Recommendation":
    st.header("🔍 Product Recommendation")

    data_path = Path("customer_data_with_recommendations.csv")   # ⚠ adjust if needed
    if not data_path.exists():
        st.error("CSV file not found.")
        st.stop()

    df      = load_customer_csv(data_path)
    sim_df  = build_item_similarity(df)

    stock   = st.selectbox("Enter / pick a StockCode", sim_df.index)
    k       = st.slider("How many products to suggest?", 1, 10, 5)

    if st.button("Recommend"):
        recs = (
            sim_df.loc[stock]
                  .drop(stock)
                  .sort_values(ascending=False)
                  .head(k)
                  .index
                  .tolist()
        )
        st.subheader("Recommended Products:")
        for item in recs:
            st.write(f"- **{item}**")

# -------------------------------------------------------------------
# 4B.  Customer Segmentation
# -------------------------------------------------------------------
else:
    st.header("👥 Customer Segmentation")

    pkl_path = Path("kmeans_rfm_model.pkl")
    if not pkl_path.exists():
        st.error("Trained model not found.")
        st.stop()

    bundle  = load_bundle(pkl_path)
    scaler  = bundle["scaler"]
    kmeans  = bundle["model"]

    col1, col2, col3 = st.columns(3)
    with col1:
        recency   = st.number_input("Recency (days)",      0, value=30)
    with col2:
        frequency = st.number_input("Frequency (purchases)",0, value=5)
    with col3:
        monetary  = st.number_input("Monetary (total spend)",0.0, value=500.0)

    if st.button("Predict Segment"):
        X_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster  = int(kmeans.predict(X_scaled)[0])
        labels   = {0: "High-Value", 1: "Regular", 2: "Occasional", 3: "At-Risk"}
        st.success(f"Predicted segment → **{labels.get(cluster, f'Cluster {cluster}')}**")

# -------------------------------------------------------------------
# 5.   Footer
# -------------------------------------------------------------------
st.caption("© 2025 Shopper Spectrum • Powered by Streamlit")
