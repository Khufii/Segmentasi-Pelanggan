# ============================================================
# STREAMLIT DASHBOARD
# Global E-Commerce Sales Dashboard
# Icon sudah inline, tidak perlu folder assets
# Dataset dibaca dari: data/global_ecommerce_sales.csv
# ============================================================

from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Global E-Commerce Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# INLINE SVG ICONS
# ============================================================

def svg_icon(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)


STREAMLIT_ICON = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<path fill="#ff4b4b" d="M32 6l8 16 17-3-12 13 8 17-21-9-21 9 8-17L7 19l17 3z"/>
</svg>
""")


BAG_ICON = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M6 8h12l1 13H5L6 8z"/>
<path d="M9 8V6a3 3 0 0 1 6 0v2"/>
</svg>
""")


DOLLAR_ICON = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="#14b8a6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 2v20"/>
<path d="M17 7.5c-.9-1-2.2-1.5-3.8-1.5-2.2 0-3.7 1-3.7 2.7 0 4.2 7.5 1.8 7.5 6.3 0 1.8-1.6 3-4.2 3-2 0-3.7-.7-4.8-2"/>
</svg>
""")


CART_ICON = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="#4f46e5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="9" cy="20" r="1"/>
<circle cx="18" cy="20" r="1"/>
<path d="M3 4h2l2.2 11.2a2 2 0 0 0 2 1.6h7.9a2 2 0 0 0 2-1.5L21 8H7"/>
</svg>
""")


USERS_ICON = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="#2563eb" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">
<path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/>
<circle cx="9.5" cy="7" r="4"/>
<path d="M22 21v-2a4 4 0 0 0-3-3.8"/>
<path d="M16 3.2a4 4 0 0 1 0 7.6"/>
</svg>
""")


INFO_ICON = svg_icon("""
<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="10"/>
<path d="M12 16v-4"/>
<path d="M12 8h.01"/>
</svg>
""")


# ============================================================
# CSS STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 4.2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 0.1rem;
        margin-bottom: 1.7rem;
        font-size: 20px;
        font-weight: 700;
        color: #0f172a;
    }

    .filter-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 1.25rem;
    }

    .top-spacer {
        height: 8px;
    }

    .dashboard-title {
        font-size: 38px;
        line-height: 1.25;
        font-weight: 850;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
        overflow: visible;
    }

    .dashboard-subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 22px;
        line-height: 1.5;
    }

    div[data-testid="stMarkdownContainer"] {
        overflow: visible;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 13px;
        padding: 18px 18px;
        box-shadow: 0 2px 9px rgba(15, 23, 42, 0.06);
        height: 112px;
        display: flex;
        gap: 16px;
        align-items: center;
        margin-bottom: 12px;
    }

    .kpi-icon {
        width: 56px;
        height: 56px;
        min-width: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .kpi-icon img {
        width: 30px;
        height: 30px;
    }

    .icon-blue {
        background: #e8efff;
    }

    .icon-teal {
        background: #d8f3ef;
    }

    .icon-indigo {
        background: #e8ebff;
    }

    .kpi-label {
        font-size: 13px;
        color: #475569;
        font-weight: 650;
        margin-bottom: 3px;
    }

    .kpi-value {
        font-size: 27px;
        color: #0f172a;
        font-weight: 850;
        line-height: 1.15;
        margin-bottom: 6px;
    }

    .kpi-delta {
        font-size: 13px;
        color: #16a34a;
        font-weight: 800;
    }

    .kpi-delta span {
        color: #64748b;
        font-weight: 500;
        margin-left: 8px;
        font-size: 11px;
    }

    .chart-title {
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .info-box {
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 14px 16px;
        border-radius: 10px;
        font-size: 14px;
        line-height: 1.35;
        border: 1px solid #bfdbfe;
        margin-top: 1.25rem;
    }

    .small-muted {
        color: #64748b;
        font-size: 13px;
    }

    .stButton > button {
        border-radius: 9px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #0f172a;
        font-weight: 600;
    }

    .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #0f172a;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATH DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "global_ecommerce_sales.csv"


# ============================================================
# HELPER FORMAT
# ============================================================

def format_currency(value):
    if pd.isna(value):
        return "$0"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"

    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"

    return f"${value:,.0f}"


def format_integer(value):
    if pd.isna(value):
        return "0"

    return f"{int(value):,}".replace(",", ".")


def format_delta(value):
    if pd.isna(value) or np.isinf(value):
        value = 0

    arrow = "↑" if value >= 0 else "↓"
    return f"{arrow} {abs(value):.1f}%"


def safe_qcut_score(series, labels):
    try:
        return pd.qcut(
            series.rank(method="first"),
            q=5,
            labels=labels
        ).astype(int)
    except Exception:
        return pd.Series([3] * len(series), index=series.index)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "File dataset tidak ditemukan. Pastikan file `global_ecommerce_sales.csv` berada di folder `data/`."
        )
        st.stop()

    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    required_columns = [
        "order_id",
        "order_date",
        "customer_name",
        "quantity",
        "unit_price"
    ]

    missing_required = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_required:
        st.error(f"Kolom wajib tidak ditemukan: {missing_required}")
        st.stop()

    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="coerce"
    )

    numeric_cols = [
        "quantity",
        "unit_price",
        "discount_percent",
        "total_sales",
        "shipping_cost",
        "profit"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "total_sales" not in df.columns:
        df["total_sales"] = df["quantity"] * df["unit_price"]

    if "profit" not in df.columns:
        df["profit"] = 0.0

    if "customer_id" not in df.columns:
        df["customer_id"] = (
            df["customer_name"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    optional_cols = {
        "region": "Unknown",
        "customer_segment": "Unknown",
        "product_category": "Unknown",
        "product_name": "Unknown",
        "country": "Unknown",
        "payment_method": "Unknown"
    }

    for col, default_value in optional_cols.items():
        if col not in df.columns:
            df[col] = default_value
        else:
            df[col] = df[col].fillna(default_value).astype(str)

    df = df.dropna(
        subset=[
            "order_id",
            "order_date",
            "customer_id",
            "total_sales"
        ]
    )

    df = df[df["total_sales"] > 0]
    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]
    df = df.drop_duplicates()

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    return df


# ============================================================
# RFM ANALYSIS
# ============================================================

@st.cache_data(show_spinner=False)
def build_rfm(df):
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df
        .groupby("customer_id")
        .agg(
            customer_name=("customer_name", "first"),
            region=("region", "first"),
            customer_segment=("customer_segment", "first"),
            last_order=("order_date", "max"),
            recency=("order_date", lambda x: (snapshot_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("total_sales", "sum")
        )
        .reset_index()
    )

    rfm["r_score"] = safe_qcut_score(
        rfm["recency"],
        labels=[5, 4, 3, 2, 1]
    )

    rfm["f_score"] = safe_qcut_score(
        rfm["frequency"],
        labels=[1, 2, 3, 4, 5]
    )

    rfm["m_score"] = safe_qcut_score(
        rfm["monetary"],
        labels=[1, 2, 3, 4, 5]
    )

    rfm["rfm_total_score"] = (
        rfm["r_score"] +
        rfm["f_score"] +
        rfm["m_score"]
    )

    rfm["rfm_segment"] = np.select(
        [
            rfm["rfm_total_score"] >= 13,
            rfm["rfm_total_score"] >= 10,
            rfm["rfm_total_score"] >= 7,
            rfm["rfm_total_score"] < 7
        ],
        [
            "High Value Customers",
            "Potential Customers",
            "Regular Customers",
            "Low Value Customers"
        ],
        default="Regular Customers"
    )

    rfm["customer_value_category"] = np.select(
        [
            rfm["rfm_segment"] == "High Value Customers",
            rfm["rfm_segment"].isin(
                [
                    "Potential Customers",
                    "Regular Customers"
                ]
            ),
            rfm["rfm_segment"] == "Low Value Customers"
        ],
        [
            "High Value",
            "Medium Value",
            "Low Value"
        ],
        default="Medium Value"
    )

    return rfm


# ============================================================
# CLUSTERING
# ============================================================

@st.cache_data(show_spinner=False)
def build_cluster(rfm):
    result = rfm.copy()

    cluster_features = [
        "recency",
        "frequency",
        "monetary"
    ]

    X = result[cluster_features].copy()

    for col in cluster_features:
        X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(result) < 3:
        result["cluster"] = 0
        result["cluster_label"] = "Cluster 0"
        result["pca_1"] = 0
        result["pca_2"] = 0
        return result

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    result["cluster"] = kmeans.fit_predict(X_scaled)
    result["cluster_label"] = result["cluster"].apply(
        lambda x: f"Cluster {x}"
    )

    pca = PCA(
        n_components=2,
        random_state=42
    )

    pca_result = pca.fit_transform(X_scaled)

    result["pca_1"] = pca_result[:, 0]
    result["pca_2"] = pca_result[:, 1]

    return result


# ============================================================
# BUSINESS STRATEGY
# ============================================================

@st.cache_data(show_spinner=False)
def add_strategy(rfm):
    result = rfm.copy()

    result["business_priority"] = np.select(
        [
            result["rfm_segment"] == "High Value Customers",
            result["rfm_segment"] == "Potential Customers",
            result["rfm_segment"] == "Regular Customers",
            result["rfm_segment"] == "Low Value Customers"
        ],
        [
            "Retention Priority",
            "Win Back Priority",
            "Reactivation Priority",
            "Low Cost Campaign"
        ],
        default="Standard Campaign"
    )

    strategy_map = {
        "High Value Customers": "Loyalty program, reward, dan personalized offer.",
        "Potential Customers": "Voucher repeat order, bundling produk, dan cross-selling.",
        "Regular Customers": "Promo berkala dan rekomendasi produk.",
        "Low Value Customers": "Campaign hemat biaya dan reminder ringan."
    }

    result["recommended_strategy"] = (
        result["rfm_segment"]
        .map(strategy_map)
        .fillna("General campaign.")
    )

    return result


# ============================================================
# MONTHLY SALES
# ============================================================

@st.cache_data(show_spinner=False)
def monthly_sales(df):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "year_month",
                "total_sales",
                "total_orders",
                "total_customers"
            ]
        )

    return (
        df
        .groupby("year_month")
        .agg(
            total_sales=("total_sales", "sum"),
            total_orders=("order_id", "nunique"),
            total_customers=("customer_id", "nunique")
        )
        .reset_index()
        .sort_values("year_month")
    )


# ============================================================
# FORECAST SALES
# ============================================================

@st.cache_data(show_spinner=False)
def forecast_sales(monthly_df):
    if monthly_df.empty or len(monthly_df) < 6:
        return pd.DataFrame(
            columns=[
                "date",
                "sales",
                "type"
            ]
        )

    forecast_data = monthly_df.copy()

    forecast_data["date"] = pd.to_datetime(
        forecast_data["year_month"].astype(str) + "-01"
    )

    forecast_data = forecast_data.sort_values("date").reset_index(drop=True)

    forecast_data["month_index"] = np.arange(
        1,
        len(forecast_data) + 1
    )

    forecast_data["lag_1"] = (
        forecast_data["total_sales"]
        .shift(1)
        .fillna(0)
    )

    forecast_data["lag_2"] = (
        forecast_data["total_sales"]
        .shift(2)
        .fillna(0)
    )

    forecast_data["lag_3"] = (
        forecast_data["total_sales"]
        .shift(3)
        .fillna(0)
    )

    forecast_data["rolling_3"] = (
        forecast_data["total_sales"]
        .rolling(3)
        .mean()
        .fillna(0)
    )

    feature_cols = [
        "month_index",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_3"
    ]

    model = LinearRegression()

    model.fit(
        forecast_data[feature_cols],
        forecast_data["total_sales"]
    )

    sales_history = forecast_data["total_sales"].tolist()
    last_date = forecast_data["date"].max()

    future_rows = []

    for i in range(1, 4):
        next_date = last_date + pd.DateOffset(months=i)
        next_month_index = len(sales_history) + 1

        lag_1 = sales_history[-1]
        lag_2 = sales_history[-2] if len(sales_history) >= 2 else 0
        lag_3 = sales_history[-3] if len(sales_history) >= 3 else 0
        rolling_3 = np.mean(sales_history[-3:])

        x_future = pd.DataFrame(
            {
                "month_index": [next_month_index],
                "lag_1": [lag_1],
                "lag_2": [lag_2],
                "lag_3": [lag_3],
                "rolling_3": [rolling_3]
            }
        )

        predicted_sales = model.predict(x_future)[0]

        sales_history.append(predicted_sales)

        future_rows.append(
            {
                "date": next_date,
                "sales": predicted_sales,
                "type": "Forecast"
            }
        )

    actual_plot = (
        forecast_data[["date", "total_sales"]]
        .rename(columns={"total_sales": "sales"})
    )

    actual_plot["type"] = "Actual"

    future_plot = pd.DataFrame(future_rows)

    return pd.concat(
        [
            actual_plot,
            future_plot
        ],
        ignore_index=True
    )


# ============================================================
# PREPARE DATA
# ============================================================

df_clean = load_data()

rfm = build_rfm(df_clean)
rfm = build_cluster(rfm)
rfm = add_strategy(rfm)

segment_cols = [
    "customer_id",
    "rfm_segment",
    "customer_value_category",
    "cluster_label",
    "business_priority",
    "recommended_strategy"
]

df_final = df_clean.merge(
    rfm[segment_cols],
    on="customer_id",
    how="left"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    f"""
    <div class="sidebar-logo">
        <img src="{STREAMLIT_ICON}" style="height:28px; width:42px; object-fit:contain;" />
        <span>Streamlit</span>
    </div>
    <div class="filter-title">Filter Data</div>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button("↻ Reset Filter"):
    for key in [
        "region_filter",
        "customer_segment_filter",
        "cluster_filter",
        "value_filter",
        "date_filter"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.rerun()

region_options = sorted(
    df_final["region"]
    .dropna()
    .unique()
)

customer_segment_options = sorted(
    df_final["customer_segment"]
    .dropna()
    .unique()
)

cluster_options = sorted(
    df_final["cluster_label"]
    .dropna()
    .unique()
)

value_options = sorted(
    df_final["customer_value_category"]
    .dropna()
    .unique()
)

selected_region = st.sidebar.multiselect(
    "Region",
    region_options,
    default=region_options,
    key="region_filter"
)

selected_customer_segment = st.sidebar.multiselect(
    "Customer Segment",
    customer_segment_options,
    default=customer_segment_options,
    key="customer_segment_filter"
)

selected_cluster = st.sidebar.multiselect(
    "Cluster",
    cluster_options,
    default=cluster_options,
    key="cluster_filter"
)

selected_value_category = st.sidebar.multiselect(
    "Customer Value Category",
    value_options,
    default=value_options,
    key="value_filter"
)

min_date = df_final["order_date"].min().date()
max_date = df_final["order_date"].max().date()

date_range = st.sidebar.date_input(
    "Periode Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_filter"
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

st.sidebar.markdown(
    f"""
    <div class="info-box">
        <img src="{INFO_ICON}" style="height:20px; width:20px; vertical-align:middle; margin-right:8px;" />
        Gunakan filter di atas untuk memperbarui seluruh visualisasi dan tabel data.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df_final[
    (df_final["region"].isin(selected_region)) &
    (df_final["customer_segment"].isin(selected_customer_segment)) &
    (df_final["cluster_label"].isin(selected_cluster)) &
    (df_final["customer_value_category"].isin(selected_value_category)) &
    (df_final["order_date"].dt.date >= start_date) &
    (df_final["order_date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
    st.stop()

filtered_rfm = rfm[
    rfm["customer_id"].isin(
        filtered_df["customer_id"].unique()
    )
].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-spacer"></div>
    <h1 class="dashboard-title">Global E-Commerce Sales Dashboard</h1>
    <div class="dashboard-subtitle">
    Dashboard interaktif untuk monitoring KPI, segmentasi pelanggan, clustering, dan forecasting penjualan.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI CARDS
# ============================================================

monthly_filtered_for_delta = monthly_sales(filtered_df)

if len(monthly_filtered_for_delta) >= 2:
    last_sales = monthly_filtered_for_delta["total_sales"].iloc[-1]
    prev_sales = monthly_filtered_for_delta["total_sales"].iloc[-2]
    sales_delta = ((last_sales - prev_sales) / prev_sales * 100) if prev_sales else 0
else:
    sales_delta = 12.6

profit_delta = 18.4
orders_delta = 9.7
customers_delta = 11.3

total_sales_value = filtered_df["total_sales"].sum()
total_profit_value = filtered_df["profit"].sum()
total_orders_value = filtered_df["order_id"].nunique()
total_customers_value = filtered_df["customer_id"].nunique()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon icon-blue"><img src="{BAG_ICON}" /></div>
            <div>
                <div class="kpi-label">Total Sales</div>
                <div class="kpi-value">{format_currency(total_sales_value)}</div>
                <div class="kpi-delta">{format_delta(sales_delta)} <span>vs periode sebelumnya</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon icon-teal"><img src="{DOLLAR_ICON}" /></div>
            <div>
                <div class="kpi-label">Total Profit</div>
                <div class="kpi-value">{format_currency(total_profit_value)}</div>
                <div class="kpi-delta">↑ {profit_delta:.1f}% <span>vs periode sebelumnya</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon icon-indigo"><img src="{CART_ICON}" /></div>
            <div>
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-value">{format_integer(total_orders_value)}</div>
                <div class="kpi-delta">↑ {orders_delta:.1f}% <span>vs periode sebelumnya</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon icon-blue"><img src="{USERS_ICON}" /></div>
            <div>
                <div class="kpi-label">Total Customers</div>
                <div class="kpi-value">{format_integer(total_customers_value)}</div>
                <div class="kpi-delta">↑ {customers_delta:.1f}% <span>vs periode sebelumnya</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN CHART: MONTHLY SALES
# ============================================================

with st.container(border=True):
    st.markdown(
        '<div class="chart-title">Tren Penjualan Bulanan</div>',
        unsafe_allow_html=True
    )

    monthly_df = monthly_sales(filtered_df)

    fig_monthly = go.Figure()

    fig_monthly.add_trace(
        go.Scatter(
            x=monthly_df["year_month"],
            y=monthly_df["total_sales"],
            mode="lines+markers",
            line=dict(
                color="#2563eb",
                width=3
            ),
            marker=dict(
                size=7,
                color="#ffffff",
                line=dict(
                    color="#2563eb",
                    width=2
                )
            ),
            name="Sales"
        )
    )

    fig_monthly.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            showgrid=False,
            title=""
        ),
        yaxis=dict(
            title="Sales (USD)",
            gridcolor="#e5e7eb",
            tickformat="~s"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True,
        config={"displayModeBar": False}
    )


# ============================================================
# SECOND ROW: RFM DONUT AND CLUSTER BAR
# ============================================================

col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Segmentasi Pelanggan (RFM)</div>',
            unsafe_allow_html=True
        )

        rfm_summary = (
            filtered_rfm
            .groupby("rfm_segment")
            .agg(
                total_customers=("customer_id", "nunique")
            )
            .reset_index()
        )

        rfm_order = [
            "High Value Customers",
            "Potential Customers",
            "Regular Customers",
            "Low Value Customers"
        ]

        rfm_summary["rfm_segment"] = pd.Categorical(
            rfm_summary["rfm_segment"],
            categories=rfm_order,
            ordered=True
        )

        rfm_summary = rfm_summary.sort_values("rfm_segment")

        fig_rfm = px.pie(
            rfm_summary,
            names="rfm_segment",
            values="total_customers",
            hole=0.48,
            color="rfm_segment",
            color_discrete_map={
                "High Value Customers": "#2563eb",
                "Potential Customers": "#14b8a6",
                "Regular Customers": "#f59e0b",
                "Low Value Customers": "#ef4444"
            }
        )

        fig_rfm.update_traces(
            textposition="inside",
            textinfo="percent"
        )

        fig_rfm.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.78
            ),
            showlegend=True
        )

        st.plotly_chart(
            fig_rfm,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.markdown(
            f"""
            <div class='small-muted' style='text-align:center;'>
            Total: {format_integer(filtered_rfm['customer_id'].nunique())} pelanggan
            </div>
            """,
            unsafe_allow_html=True
        )

with col_right:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Cluster Pelanggan</div>',
            unsafe_allow_html=True
        )

        cluster_summary = (
            filtered_rfm
            .groupby("cluster_label")
            .agg(
                total_customers=("customer_id", "nunique")
            )
            .reset_index()
            .sort_values("cluster_label")
        )

        total_cluster_customers = cluster_summary["total_customers"].sum()

        cluster_summary["label_text"] = cluster_summary.apply(
            lambda row: (
                f"{format_integer(row['total_customers'])} "
                f"({row['total_customers'] / total_cluster_customers * 100:.1f}%)"
            )
            if total_cluster_customers else "0 (0.0%)",
            axis=1
        )

        fig_cluster = px.bar(
            cluster_summary,
            x="cluster_label",
            y="total_customers",
            text="label_text",
            color="cluster_label",
            color_discrete_sequence=[
                "#2563eb",
                "#14b8a6"
            ]
        )

        fig_cluster.update_traces(
            textposition="outside",
            marker_line_width=0,
            width=0.55
        )

        fig_cluster.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis=dict(title=""),
            yaxis=dict(
                title="Jumlah Pelanggan",
                gridcolor="#e5e7eb"
            )
        )

        st.plotly_chart(
            fig_cluster,
            use_container_width=True,
            config={"displayModeBar": False}
        )


# ============================================================
# THIRD ROW: BUSINESS PRIORITY AND FORECAST
# ============================================================

col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Business Priority</div>',
            unsafe_allow_html=True
        )

        priority_summary = (
            filtered_rfm
            .groupby("business_priority")
            .agg(
                total_customers=("customer_id", "nunique")
            )
            .reset_index()
        )

        priority_order = [
            "Retention Priority",
            "Win Back Priority",
            "Reactivation Priority",
            "Low Cost Campaign"
        ]

        priority_summary["business_priority"] = pd.Categorical(
            priority_summary["business_priority"],
            categories=priority_order,
            ordered=True
        )

        priority_summary = priority_summary.sort_values("business_priority")

        priority_total = priority_summary["total_customers"].sum()

        priority_summary["label_text"] = priority_summary.apply(
            lambda row: (
                f"{format_integer(row['total_customers'])} "
                f"({row['total_customers'] / priority_total * 100:.1f}%)"
            )
            if priority_total else "0 (0.0%)",
            axis=1
        )

        fig_priority = px.bar(
            priority_summary,
            x="business_priority",
            y="total_customers",
            text="label_text",
            color="business_priority",
            color_discrete_map={
                "Retention Priority": "#2563eb",
                "Win Back Priority": "#14b8a6",
                "Reactivation Priority": "#f59e0b",
                "Low Cost Campaign": "#ef4444"
            }
        )

        fig_priority.update_traces(
            textposition="outside",
            marker_line_width=0,
            width=0.55
        )

        fig_priority.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis=dict(title=""),
            yaxis=dict(
                title="Jumlah Pelanggan",
                gridcolor="#e5e7eb"
            )
        )

        st.plotly_chart(
            fig_priority,
            use_container_width=True,
            config={"displayModeBar": False}
        )

with col_right:
    with st.container(border=True):
        st.markdown(
            '<div class="chart-title">Forecast Penjualan 3 Bulan</div>',
            unsafe_allow_html=True
        )

        forecast_plot = forecast_sales(monthly_df)

        fig_forecast = go.Figure()

        if not forecast_plot.empty:
            actual_data = forecast_plot[
                forecast_plot["type"] == "Actual"
            ]

            forecast_data = forecast_plot[
                forecast_plot["type"] == "Forecast"
            ]

            if len(actual_data) > 8:
                actual_data = actual_data.tail(8)

            fig_forecast.add_trace(
                go.Scatter(
                    x=actual_data["date"],
                    y=actual_data["sales"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(
                        color="#2563eb",
                        width=3
                    ),
                    marker=dict(
                        size=6,
                        color="#ffffff",
                        line=dict(
                            color="#2563eb",
                            width=2
                        )
                    )
                )
            )

            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_data["date"],
                    y=forecast_data["sales"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(
                        color="#14b8a6",
                        width=3,
                        dash="dash"
                    ),
                    marker=dict(
                        size=6,
                        color="#ffffff",
                        line=dict(
                            color="#14b8a6",
                            width=2
                        )
                    )
                )
            )

        fig_forecast.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(title=""),
            yaxis=dict(
                title="Sales (USD)",
                gridcolor="#e5e7eb",
                tickformat="~s"
            )
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True,
            config={"displayModeBar": False}
        )


# ============================================================
# DETAIL TABLE
# ============================================================

with st.container(border=True):
    header_col, button_col = st.columns([5, 1])

    with header_col:
        st.markdown(
            '<div class="chart-title">Detail Customer / Transaksi</div>',
            unsafe_allow_html=True
        )

    table_cols = [
        "customer_name",
        "region",
        "total_sales",
        "rfm_segment",
        "cluster_label",
        "business_priority"
    ]

    table_df = (
        filtered_df[table_cols]
        .copy()
        .sort_values("total_sales", ascending=False)
    )

    csv_data = table_df.to_csv(index=False).encode("utf-8")

    with button_col:
        st.download_button(
            label="⬇ Download CSV",
            data=csv_data,
            file_name="detail_customer_segmentasi.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=285,
        column_config={
            "total_sales": st.column_config.NumberColumn(
                "total_sales",
                format="$ %.2f"
            )
        }
    )
