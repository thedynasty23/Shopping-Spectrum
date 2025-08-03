# app.py
# ---------------------------------------------
# Shopper Spectrum Dashboard   (Streamlit)
# ---------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

###############################################################################
# ---------- 1.  Load data & models (cached) ----------------------------------
###############################################################################
@st.cache_data(show_spinner=False)
def load_customer_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource(show_spinner=False)
def build_item_similarity(df: pd.DataFrame):
    """
    Build a sparse Customer × Product matrix and
    compute item-to-item cosine similarity.
    Returns: (similarity_df, product_lookup)
    """
    pivot = (df[['CustomerID', 'Rec1_StockCode']]
             .rename(columns={'Rec1_StockCode': 'StockCode'}))

    # Stack the 10 rec columns into long form  (faster than loop)
    rec_cols = [c for c in df.columns if c.startswith('Rec') and c.endswith('StockCode')]
    long_df = pd.melt(df[['CustomerID'] + rec_cols],
                      id_vars='CustomerID',
                      value_vars=rec_cols,
                      value_name='StockCode').dropna()

    # Concatenate, drop duplicates to mark purchase/exposure
    events = pd.concat([pivot, long_df])[['CustomerID', 'StockCode']].dropna().drop_duplicates()

    # Numeric index mapping
    cust_ids = events['CustomerID'].astype(int).astype('category').cat.codes
    item_ids = events['StockCode'].astype(str).astype('category').cat.codes

    matrix = csr_matrix(
        (np.ones(len(events), dtype=np.int8),
         (cust_ids, item_ids)),
        shape=(cust_ids.max()+1, item_ids.max()+1)
    )

    sim = cosine_similarity(matrix.T, dense_output=False)
    # similarity to DataFrame with StockCode labels
    idx_to_code = events['StockCode'].astype('category').cat.categories
    sim_df = pd.DataFrame(sim.toarray(), index=idx_to_code, columns=idx_to_code)

    return sim_df

@st.cache_resource(show_spinner=False)
def load_kmeans_model(pkl_path: str):
    return joblib.load(pkl_path)

###############################################################################
# ---------------------- 2.   Page layout -------------------------------------
###############################################################################
st.set_page_config(page_title="Shopper Spectrum Dashboard",
                   page_icon="🛒",
                   layout="wide")

st.title("🛒 Shopper Spectrum Dashboard")

# Sidebar – navigation
section = st.sidebar.radio("Choose module", ["Product Recommendation",
                                             "Customer Segmentation"])

###############################################################################
# ------------ 3A.   Product Recommendation -----------------------------------
###############################################################################
if section == "Product Recommendation":
    st.header("🔍 Product Recommendation")

    # Load CSV & build similarity
    data_file = Path("customer_data_with_recommendations.csv")
    if not data_file.exists():
        st.error("CSV file not found in working directory.")
        st.stop()

    df = load_customer_csv(str(data_file))
    sim_df = build_item_similarity(df)

    # User input
    prod_codes = sim_df.index.tolist()
    default_code = prod_codes[0] if prod_codes else ""
    product = st.selectbox("Select a StockCode", prod_codes, index=0)

    n_recs = st.slider("Number of recommendations", 1, 10, 5, 1)

    if st.button("Get Recommendations", use_container_width=True):
        if product not in sim_df.index:
            st.warning("Unknown StockCode.")
        else:
            scores = sim_df.loc[product].drop(product).sort_values(ascending=False)
            top_items = scores.head(n_recs).index
            st.subheader("📋 Recommended Products")
            for code in top_items:
                st.write(f"- **{code}**")

###############################################################################
# --------------- 3B.   Customer Segmentation ---------------------------------
###############################################################################
if section == "Customer Segmentation":
    st.header("👥 Predict Customer Segment")

    kmeans_path = Path("kmeans_rfm_model.pkl")
    if not kmeans_path.exists():
        st.error("Trained K-Means model (kmeans_rfm_model.pkl) not found.")
        st.stop()

    model = load_kmeans_model(str(kmeans_path))

    # Numeric inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (# purchases)", min_value=0, value=5)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Cluster", use_container_width=True):
        sample = np.array([[recency, frequency, monetary]])
        cluster = int(model.predict(sample)[0])

        seg_map = {0: "High-Value",
                   1: "Regular",
                   2: "Occasional",
                   3: "At-Risk"}
        seg_label = seg_map.get(cluster, f"Cluster {cluster}")
        st.success(f"💡 Predicted Segment: **{seg_label}**  (cluster {cluster})")

###############################################################################
# ----------------- 4.   Footer ------------------------------------------------
###############################################################################
st.caption("© 2025 Shopper Spectrum • Powered by Streamlit")
