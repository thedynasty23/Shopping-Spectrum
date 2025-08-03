# app.py
# -------------------------------------------------------------------
# Shopper Spectrum Dashboard – Streamlit v2 (Fixed & Deployment-ready)
# -------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# -------------------------------------------------------------------
# 1.   Helpers & Caching
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_customer_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_resource(show_spinner=False)
def build_similarity_and_lookup(df: pd.DataFrame):
    # Ensure Description / StockCode columns exist
    if "Description" not in df.columns or "StockCode" not in df.columns:
        if "Rec1_Description" in df.columns and "Rec1_StockCode" in df.columns:
            df["Description"] = df["Rec1_Description"]
            df["StockCode"] = df["Rec1_StockCode"]
        else:
            raise ValueError("DataFrame must contain either 'Description' and 'StockCode' or 'Rec1_' fallback columns.")

    lookup = df[["Description", "StockCode"]].drop_duplicates().set_index("Description")["StockCode"]
    reverse_lookup = lookup.reset_index().set_index("StockCode")["Description"]

    rec_cols = [col for col in df.columns if col.startswith("Rec") and col.endswith("StockCode")]
    long_df = pd.melt(df[["CustomerID"] + rec_cols], id_vars="CustomerID", value_vars=rec_cols,
                      value_name="StockCode").dropna().drop_duplicates()

    cust_ids = long_df["CustomerID"].astype("category").cat.codes
    item_ids = long_df["StockCode"].astype("category").cat.codes
    mat = csr_matrix((np.ones(len(long_df), dtype=np.int8), (cust_ids, item_ids)))

    sim = cosine_similarity(mat.T, dense_output=False)
    labels = long_df["StockCode"].astype("category").cat.categories
    sim_df = pd.DataFrame(sim.toarray(), index=labels, columns=labels)

    return sim_df, lookup, reverse_lookup

@st.cache_resource(show_spinner=False)
def load_bundle(pkl_path: str):
    return joblib.load(pkl_path)

# -------------------------------------------------------------------
# 2.   Page Setup & Theme
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* ----------  existing rules ---------- */
    [data-testid="stAppViewContainer"] {background:white;}
    [data-testid="stSidebar"]{background:#f8f9fa;padding-top:1.5rem;}
    h1,h2,h3{color:#111!important;}
    .stButton>button{width:100%;background:#4f8bf9;color:white;}
    .stButton>button:hover{background:#3b6ec9;color:white;}
    div[data-testid="stSpinner"] > div > div {color:#4f8bf9;}
    .nav-btn{cursor:pointer;padding:0.6rem 1.2rem;border-radius:6px;
             display:flex;align-items:center;gap:0.4rem;font-weight:600;
             color:#555;font-size:0.9rem;margin-bottom:0.5rem;}
    .nav-btn:hover{background:#f0f4ff;color:#1f52ff;}
    .nav-btn.selected{background:#1f52ff;color:#fff;}
    .icon-circle{width:12px;height:12px;border-radius:50%;}
    
    /* ----------  NEW rule: force body text to black ---------- */
    body, div[data-testid="stAppViewContainer"] * {
        color:#000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# 3.   Sidebar Navigation
# -------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "rec"

qs = st.query_params
if "page" in qs:
    st.session_state.page = qs["page"] if isinstance(qs["page"], str) else qs["page"][0]

# -------------------------------------------------------------------
# 3.   Sidebar Navigation
# -------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "rec"          # default tab

qs = st.query_params
if "page" in qs:
    # qs["page"] can be list or str → normalise to str
    st.session_state.page = qs["page"] if isinstance(qs["page"], str) else qs["page"][0]

page = st.session_state.page              

def nav_button(key, label, color):
    selected = "selected" if st.session_state.page == key else ""
    # When clicked, the link itself changes the query string **and**
    # triggers a full page reload → Streamlit reruns automatically.
    st.sidebar.markdown(
        f"""
        <a href="?page={key}" style="text-decoration:none">
            <div class="nav-btn {selected}">
                <div class="icon-circle" style="background:{color};"></div>{label}
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    if st.button("🛍️ Product Recommendation", key="nav_rec"):
        st.session_state.page = "rec"
        st.rerun()

    if st.button("👥 Customer Segmentation", key="nav_seg"):
        st.session_state.page = "seg"
        st.rerun()
# ===================================================================
# 4A.  PRODUCT RECOMMENDATION PAGE
# ===================================================================
if page == "rec":
    st.header("🔍 Product Recommendation")
    data_path = Path("customer_data_with_recommendations.csv")

    if not data_path.exists():
        st.error("Required data file 'customer_data_with_recommendations.csv' not found.")
        st.stop()

    with st.spinner("Loading data and similarity matrix..."):
        df = load_customer_csv(data_path)
        sim_df, desc2code, code2desc = build_similarity_and_lookup(df)

    description = st.selectbox(
        "Select a product description:",
        sorted(desc2code.index),
        help="Dropdown of available product descriptions"
    )

    k = st.slider("Number of similar products to show", 1, 10, 5)

    if st.button("Recommend"):
        try:
            code = desc2code[description]
            scores = sim_df.loc[code].drop(code).sort_values(ascending=False)
            top_codes = scores.head(k).index
            st.subheader("Products customers also buy:")
            for c in top_codes:
                st.write(f"• **{code2desc.get(c, c)}**")
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

# ===================================================================
# 4B.  CUSTOMER SEGMENTATION PAGE
# ===================================================================
elif page == "seg":
    st.header("👥 Customer Segmentation")

    pkl_path = Path("kmeans_rfm_model.pkl")
    if not pkl_path.exists():
        st.error("Trained model 'kmeans_rfm_model.pkl' not found.")
        st.stop()

    kmeans = load_bundle(pkl_path)

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (orders)", min_value=0, value=5)
    with col3:
        monetary = st.number_input("Monetary (₹)", min_value=0.0, value=500.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        basket_avg = st.number_input("Avg. Basket Value (₹)", min_value=0.0, value=100.0)
    with col5:
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
    with col6:
        returns = st.number_input("Returns", min_value=0, value=0)

    if st.button("Predict segment"):
        features = [[recency, frequency, monetary, basket_avg, tenure, returns]]
        try:
            cluster = int(kmeans.predict(features)[0])
            seg_names = {
                0: "High-Value",
                1: "Regular",
                2: "Occasional",
                3: "At-Risk"
            }
            label = seg_names.get(cluster, f"Cluster {cluster}")
            st.success(f"Predicted segment → **{label}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------------------------------------------
# 5.   Footer
# -------------------------------------------------------------------
st.markdown("<br><hr><br>", unsafe_allow_html=True)
st.caption("© 2025 Shopper Spectrum • Powered by Streamlit")

