# -------------------------------------------------------------------
# Shopper Spectrum Dashboard  –  Streamlit v3 (clean white theme)
# -------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# -------------------------------------------------------------------
# 1.  Helpers & Caching
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_customer_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

@st.cache_resource(show_spinner=False)
def build_similarity_and_lookup(df: pd.DataFrame):
    # guarantee we have Description & StockCode columns
    if {"Description", "StockCode"}.issubset(df.columns) is False:
        if {"Rec1_Description", "Rec1_StockCode"}.issubset(df.columns):
            df["Description"] = df["Rec1_Description"]
            df["StockCode"]   = df["Rec1_StockCode"]
        else:
            raise ValueError("No Description/StockCode columns found.")

    # forward & reverse look-ups
    desc2code = (
        df[["Description", "StockCode"]]
        .drop_duplicates()
        .set_index("Description")["StockCode"]
    )
    code2desc = desc2code.reset_index().set_index("StockCode")["Description"]

    # customer-item matrix for cosine similarity
    rec_cols = [c for c in df.columns if c.startswith("Rec") and c.endswith("StockCode")]
    long_df  = (
        pd.melt(df[["CustomerID"] + rec_cols],
                id_vars="CustomerID",
                value_vars=rec_cols,
                value_name="StockCode")
        .dropna()
        .drop_duplicates()
    )
    cust_ids = long_df["CustomerID"].astype("category").cat.codes
    item_ids = long_df["StockCode"].astype("category").cat.codes
    mat      = csr_matrix((np.ones(len(long_df), dtype=np.int8), (cust_ids, item_ids)))
    sim      = cosine_similarity(mat.T, dense_output=False)
    labels   = long_df["StockCode"].astype("category").cat.categories
    sim_df   = pd.DataFrame(sim.toarray(), index=labels, columns=labels)

    # ensure EVERY code in the similarity matrix maps to a description
    missing = sim_df.index.difference(code2desc.index)
    if len(missing):
        code2desc = pd.concat(
            [code2desc,
             pd.Series(missing, index=missing, name="Description")]
        )

    return sim_df, desc2code, code2desc

@st.cache_resource(show_spinner=False)
def load_bundle(pkl_path: str):
    return joblib.load(pkl_path)

# -------------------------------------------------------------------
# 2.  Page Setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# 3.  Global CSS  (sidebar, widgets, dropdown font)
# -------------------------------------------------------------------
ACCENT      = "#4095FF"   # brand blue
SIDEBAR_BG  = "#f2f2f2"   # light-grey sidebar
DROPDOWN_FG = "#ffffff"   # white font for selectbox

st.markdown(
    f"""
    <style>
        /* GLOBAL */
        body, .stApp {{ background:#ffffff !important; color:#000000 !important;
                        font-family:"Inter",sans-serif; }}

        /* SIDEBAR */
        section[data-testid="stSidebar"] > div:first-child {{
            background:{SIDEBAR_BG} !important; border-right:1px solid #e0e0e0;
        }}

        .sidebar-btn {{ display:flex; align-items:center; padding:0.6rem 1rem;
                        margin:0.2rem 0; border-radius:6px; cursor:pointer;
                        font-weight:600; color:#000000; text-decoration:none; }}
        .sidebar-btn:hover   {{ background:#e8e8e8; }}
        .sidebar-btn.selected{{ background:{ACCENT}; color:#ffffff; }}

        /* ACCENTED WIDGETS */
        div.stButton > button,
        .stNumberInput input,
        .stAlert,
        .stMetric {{
            background:{ACCENT} !important; color:#ffffff !important;
            border:1px solid {ACCENT} !important; }}
        div.stButton > button:hover {{ background:#2577EB !important; }}

        /* SELECTBOX FONT COLOR */
        .stSelectbox div[data-baseweb="select"] div[role="combobox"],
        .stSelectbox div[data-baseweb="select"] div[role="option"] {{
            color:{DROPDOWN_FG} !important; }}

        /* keep headers black */
        label,h1,h2,h3,h4,h5,h6,p,span,div {{ color:#000000; }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# 4.  Sidebar Navigation
# -------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "rec"

qs = st.query_params
if "page" in qs:
    st.session_state.page = qs["page"] if isinstance(qs["page"], str) else qs["page"][0]

page = st.session_state.page

with st.sidebar:
    st.title("🛒 Shopper Spectrum")
    if st.button("🛍️  Product Recommendation", use_container_width=True):
        st.session_state.page = "rec"; st.rerun()
    if st.button("👥  Customer Segmentation",  use_container_width=True):
        st.session_state.page = "seg"; st.rerun()
    st.markdown("---")
    st.caption("© 2025 Shopper Spectrum")

# -------------------------------------------------------------------
# 5A.  PRODUCT RECOMMENDATION PAGE
# -------------------------------------------------------------------
if page == "rec":
    st.header("🛍️ Product Recommender")
    st.markdown("Choose a product and see what other customers buy alongside it.")

    csv_path = Path("customer_data_with_recommendations.csv")
    if not csv_path.exists():
        st.error("CSV ‘customer_data_with_recommendations.csv’ not found."); st.stop()

    df = load_customer_csv(csv_path)
    sim_df, desc2code, code2desc = build_similarity_and_lookup(df)

    product_list = sorted(desc2code.index.unique())
    description  = st.selectbox("Choose a Product", product_list, index=0)
    k            = st.slider("Number of recommendations", 1, 10, 5, key="k")

    if st.button("Recommend"):
        code      = desc2code[description]
        scores    = sim_df.loc[code].drop(code).sort_values(ascending=False)
        top_codes = scores.head(k).index

        st.subheader("Recommended Products")
        for sc in top_codes:
            st.write(f"• **{code2desc[sc]}**")

# -------------------------------------------------------------------
# 5B.  CUSTOMER SEGMENTATION PAGE
# -------------------------------------------------------------------
elif page == "seg":
    st.header("👥 Customer Segmentation")
    st.markdown("Predict which customer segment a shopper belongs to using recent RFM stats.")

    pkl_path = Path("kmeans_rfm_model.pkl")
    if not pkl_path.exists():
        st.error("Trained model ‘kmeans_rfm_model.pkl’ not found."); st.stop()

    kmeans = load_bundle(pkl_path)

    col1, col2, col3 = st.columns(3)
    with col1: recency   = st.number_input("Recency (days)", 0, value=30)
    with col2: frequency = st.number_input("Frequency (orders)", 0, value=5)
    with col3: monetary  = st.number_input("Monetary (₹)", 0.0, value=500.0)

    col4, col5, col6 = st.columns(3)
    with col4: basket_avg = st.number_input("Avg. Basket Value (₹)", 0.0, value=100.0)
    with col5: tenure     = st.number_input("Tenure (months)", 0, value=12)
    with col6: returns    = st.number_input("Returns", 0, value=0)

    if st.button("Predict Segment"):
        try:
            cluster = int(kmeans.predict([[recency, frequency, monetary,
                                           basket_avg, tenure, returns]])[0])
            seg_map = {0:"High-Value", 1:"Regular", 2:"Occasional", 3:"At-Risk"}
            st.success(f"This customer belongs to: **{seg_map.get(cluster, f'Cluster {cluster}')}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------------------------------------------
# 6.  Footer
# -------------------------------------------------------------------
st.markdown("<br><hr style='border:0;border-top:1px solid #e0e0e0'><br>", unsafe_allow_html=True)
st.caption("Built with ❤️ and Streamlit")
