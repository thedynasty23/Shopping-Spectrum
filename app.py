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
        /* ---------- GLOBAL ---------- */
        body, .stApp {                       /* whole page */
            background:#ffffff !important;   /* white page */
            color:#000000 !important;        /* black text */
        }

        /* ---------- RECTANGLE-STYLE WIDGETS ---------- */
        /* Dark background + white text */
        div.stButton>button,
        .stNumberInput  input,               /* number_input box */
        .stAlert,
        .stMetric {
            background:#1e1e1e !important;   /* dark rectangle */
            color:#ffffff !important;        /* white text      */
            border:1px solid #1e1e1e !important;
        }

        /* ---------- KEEP LABELS / HEADERS DEFAULT ---------- */
        label, h1, h2, h3, h4, h5, h6, p, span, div {
            color:#000000 !important;        /* black everywhere else */
        }
    </style>
    """,
    unsafe_allow_html=True
)

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
# app.py  (excerpt)

# … previous code …
if st.button("Recommend"):          # <-- line 145
    try:                            # <-- start of indented block
        code = desc2code[description]
        scores = (
            sim_df.loc[code]
                  .drop(code)
                  .sort_values(ascending=False)
        )
        top_codes = scores.head(k).index

        # build full StockCode ➔ Description mapping
        code2desc_full = (
            df[
                [
                    'Rec1_StockCode', 'Rec1_Description',
                    'Rec2_StockCode', 'Rec2_Description',
                    'Rec3_StockCode', 'Rec3_Description',
                    'Rec4_StockCode', 'Rec4_Description',
                    'Rec5_StockCode', 'Rec5_Description',
                    'Rec6_StockCode', 'Rec6_Description',
                    'Rec7_StockCode', 'Rec7_Description',
                    'Rec8_StockCode', 'Rec8_Description',
                    'Rec9_StockCode', 'Rec9_Description',
                    'Rec10_StockCode', 'Rec10_Description',
                ]
            ]
            .melt(value_vars=[
                'Rec1_StockCode', 'Rec2_StockCode', 'Rec3_StockCode',
                'Rec4_StockCode', 'Rec5_StockCode', 'Rec6_StockCode',
                'Rec7_StockCode', 'Rec8_StockCode', 'Rec9_StockCode',
                'Rec10_StockCode'
            ],
            var_name='dummy',
            value_name='StockCode')
            .join(
                df.melt(
                    value_vars=[
                        'Rec1_Description', 'Rec2_Description',
                        'Rec3_Description', 'Rec4_Description',
                        'Rec5_Description', 'Rec6_Description',
                        'Rec7_Description', 'Rec8_Description',
                        'Rec9_Description', 'Rec10_Description'
                    ],
                    value_name='Description'
                )['Description']
            )
            .dropna()
        )
        csv_lookup = dict(
            zip(code2desc_full['StockCode'], code2desc_full['Description'])
        )

        st.subheader("Products customers also buy:")
        for c in top_codes:
            desc = csv_lookup.get(c)
            if desc:                # only display if description exists
                st.write(f"• **{desc}**")
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

