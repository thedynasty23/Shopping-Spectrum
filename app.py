# app.py
# -------------------------------------------------------------------
# Shopper Spectrum Dashboard  –  v2  (Streamlit)
# -------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# -------------------------------------------------------------------
# 1.   Helpers & caching
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_customer_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_resource(show_spinner=False)
def build_similarity_and_lookup(df: pd.DataFrame):
    # inside build_similarity_and_lookup(df) – add just after the function starts
    if 'Description' not in df.columns or 'StockCode' not in df.columns:
        df = df.copy()
        df['Description'] = df['Rec1_Description']
        df['StockCode']   = df['Rec1_StockCode']

    # Build description↔code lookup  (uses first occurrence only)
    lookup = df[['Description', 'StockCode']].drop_duplicates().set_index('Description')['StockCode']
    reverse_lookup = lookup.reset_index().set_index('StockCode')['Description']

    # Build long customer-item table (all Rec columns)
    rec_cols = [c for c in df.columns if c.startswith("Rec") and c.endswith("StockCode")]
    long_df  = pd.melt(df[["CustomerID"] + rec_cols],
                       id_vars="CustomerID",
                       value_vars=rec_cols,
                       value_name="StockCode").dropna().drop_duplicates()

    # Sparse matrix
    cust_ids = long_df["CustomerID"].astype("category").cat.codes
    item_ids = long_df["StockCode" ].astype("category").cat.codes
    mat      = csr_matrix((np.ones(len(long_df), dtype=np.int8),
                           (cust_ids, item_ids)))
    sim      = cosine_similarity(mat.T, dense_output=False)
    labels   = long_df["StockCode"].astype("category").cat.categories
    sim_df   = pd.DataFrame(sim.toarray(), index=labels, columns=labels)

    return sim_df, lookup, reverse_lookup

@st.cache_resource(show_spinner=False)
def load_bundle(pkl_path: str):
    return joblib.load(pkl_path)      # {"scaler": ..., "model": ...}

# -------------------------------------------------------------------
# 2.   Global page setup  (white theme + accent)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:white; }
[data-testid="stSidebar"]{ background:#f8f9fa;padding-top:2rem; }
h1,h2,h3{color:#111!important;}
.stButton>button{width:100%;background:#4f8bf9;color:white;}
.stButton>button:hover{background:#3b6ec9;color:white;}
div[data-testid="stSpinner"] > div > div { color:#4f8bf9; }  /* spinner */
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 3.   Sidebar navigation
# -------------------------------------------------------------------
st.sidebar.header("🛒  Shopper Spectrum")
choice = st.sidebar.radio("Go to", ("Product Recommendation", "Customer Segmentation"),
                          format_func=lambda x: "🔍 " + x if x.startswith("Product") else "👥 " + x)

# ===================================================================
# 4A.  PRODUCT RECOMMENDATION  (by Description)
# ===================================================================
if choice.endswith("Recommendation"):
    st.header("🔍 Product Recommendation")
    data_path = Path("customer_data_with_recommendations.csv")

    if not data_path.exists():
        st.error("CSV file not found.")
        st.stop()

    with st.spinner("Loading data & similarity model …"):
        df = load_customer_csv(data_path)
        sim_df, desc2code, code2desc = build_similarity_and_lookup(df)

    # --- User input ---
    description = st.selectbox("Select a product (description)",
                               sorted(desc2code.index), index=0,
                               help="Dropdown shows unique product descriptions.")
    k = st.slider("How many similar products?", 1, 10, 5,
                  help="Top-N items ranked by cosine similarity.")

    # --- Action ---
    if st.button("Recommend"):
        code = desc2code[description]
        scores = sim_df.loc[code].drop(code).sort_values(ascending=False)
        top_codes = scores.head(k).index
        st.subheader("Products customers also buy:")
        for c in top_codes:
            st.write(f"• **{code2desc.get(c, c)}**")

# ===================================================================
# 4B.  CUSTOMER SEGMENTATION
# ===================================================================
else:
    st.header("👥 Customer Segmentation")
    pkl_path = Path("kmeans_rfm_model.pkl")
    if not pkl_path.exists():
        st.error("Trained model not found.")
        st.stop()

    bundle = load_bundle(pkl_path)
    scaler = bundle["scaler"]; kmeans = bundle["model"]

    col1, col2, col3 = st.columns(3)
    with col1:  recency   = st.number_input("Recency (days)",       0, value=30)
    with col2:  frequency = st.number_input("Frequency (purchases)",0, value=5)
    with col3:  monetary  = st.number_input("Monetary (total spend)",0.0, value=500.0)

    with st.expander("ℹ️  What do these mean?"):
        st.markdown("""
        • **Recency** – Days since last purchase  
        • **Frequency** – Number of orders placed  
        • **Monetary** – Total spend amount
        """)
    if st.button("Predict segment"):
        X_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster  = int(kmeans.predict(X_scaled)[0])
        labels   = {0:"High-Value",1:"Regular",2:"Occasional",3:"At-Risk"}
        st.success(f"Predicted segment → **{labels.get(cluster, f'Cluster {cluster}')}**")

# -------------------------------------------------------------------
# 5.   Footer
# -------------------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Shopper Spectrum • Powered by Streamlit")
