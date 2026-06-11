# ============================================================
# STREAMLIT DASHBOARD
# Fokus: Segmentasi Pelanggan E-Commerce
# Dataset: Global E-Commerce Sales & Customer Data
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dashboard Segmentasi Pelanggan",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 42px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
    }

    .dashboard-subtitle {
        font-size: 16px;
        color: #64748b;
        margin-bottom: 24px;
    }

    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
        height: 135px;
    }

    .kpi-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 28px;
        color: #0f172a;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .kpi-note {
        font-size: 13px;
        color: #16a34a;
        font-weight: 600;
    }

    .info-box {
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 14px;
        border-radius: 10px;
        font-size: 14px;
        border: 1px solid #bfdbfe;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTION
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

@st.cache_data
def load_data():
    file_path = Path("data/global_ecommerce_sales.csv")

    if not file_path.exists():
        st.error(
            "File dataset tidak ditemukan. "
            "Pastikan file `global_ecommerce_sales.csv` berada di folder `data/`."
        )
        st.stop()

    df_raw = pd.read_csv(file_path)

    df = df_raw.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

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
        df["profit"] = 0

    if "customer_name" not in df.columns:
        df["customer_name"] = "Customer"

    if "customer_id" not in df.columns:
        df["customer_id"] = (
            df["customer_name"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    optional_cols = {
        "region": "Tidak Diketahui",
        "customer_segment": "Tidak Diketahui",
        "product_category": "Tidak Diketahui",
        "product_name": "Tidak Diketahui",
        "payment_method": "Tidak Diketahui",
        "country": "Tidak Diketahui"
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

    if "quantity" in df.columns:
        df = df[df["quantity"] > 0]

    if "unit_price" in df.columns:
        df = df[df["unit_price"] > 0]

    df = df.drop_duplicates()

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    return df


# ============================================================
# RFM ANALYSIS
# ============================================================

@st.cache_data
def build_rfm(df):
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df
        .groupby("customer_id")
        .agg(
            customer_name=("customer_name", "first"),
            region=("region", "first"),
            customer_segment=("customer_segment", "first"),
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
            "Pelanggan Bernilai Tinggi",
            "Pelanggan Potensial",
            "Pelanggan Reguler",
            "Pelanggan Bernilai Rendah"
        ],
        default="Pelanggan Reguler"
    )

    return rfm


# ============================================================
# CLUSTERING
# ============================================================

@st.cache_data
def build_cluster(rfm):
    rfm_clustered = rfm.copy()

    cluster_features = [
        "recency",
        "frequency",
        "monetary"
    ]

    X = rfm_clustered[cluster_features].copy()

    for col in cluster_features:
        X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if len(rfm_clustered) < 2:
        rfm_clustered["cluster"] = 0
        rfm_clustered["cluster_label"] = "Cluster 0"
        rfm_clustered["pca_1"] = 0
        rfm_clustered["pca_2"] = 0
        return rfm_clustered

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    rfm_clustered["cluster"] = kmeans.fit_predict(X_scaled)
    rfm_clustered["cluster_label"] = rfm_clustered["cluster"].apply(
        lambda x: f"Cluster {x}"
    )

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_scaled)

    rfm_clustered["pca_1"] = pca_result[:, 0]
    rfm_clustered["pca_2"] = pca_result[:, 1]

    return rfm_clustered


# ============================================================
# BUSINESS STRATEGY
# ============================================================

@st.cache_data
def add_strategy(rfm):
    rfm_strategy = rfm.copy()

    rfm_strategy["business_priority"] = np.select(
        [
            rfm_strategy["rfm_segment"] == "Pelanggan Bernilai Tinggi",
            rfm_strategy["rfm_segment"] == "Pelanggan Potensial",
            rfm_strategy["rfm_segment"] == "Pelanggan Reguler",
            rfm_strategy["rfm_segment"] == "Pelanggan Bernilai Rendah"
        ],
        [
            "Prioritas Retensi",
            "Prioritas Pengembangan",
            "Prioritas Keterlibatan",
            "Kampanye Biaya Rendah"
        ],
        default="Prioritas Standar"
    )

    strategy_map = {
        "Pelanggan Bernilai Tinggi": "Pertahankan dengan loyalty program, reward, dan personalized offer.",
        "Pelanggan Potensial": "Dorong repeat order melalui voucher pembelian berikutnya dan bundling produk.",
        "Pelanggan Reguler": "Tingkatkan engagement dengan rekomendasi produk dan promosi berkala.",
        "Pelanggan Bernilai Rendah": "Gunakan campaign hemat biaya seperti promo sederhana atau reminder ringan."
    }

    rfm_strategy["recommended_strategy"] = (
        rfm_strategy["rfm_segment"]
        .map(strategy_map)
        .fillna("Gunakan strategi pemasaran umum.")
    )

    return rfm_strategy


# ============================================================
# MONTHLY SALES
# ============================================================

def build_monthly_sales(df):
    monthly_sales = (
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

    return monthly_sales


# ============================================================
# PREPARE DATA
# ============================================================

df_clean = load_data()

rfm = build_rfm(df_clean)
rfm = build_cluster(rfm)
rfm = add_strategy(rfm)

segment_result = rfm[
    [
        "customer_id",
        "rfm_segment",
        "cluster_label",
        "business_priority",
        "recommended_strategy"
    ]
]

df_final = df_clean.merge(
    segment_result,
    on="customer_id",
    how="left"
)


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.markdown("## Filter Data")

region_options = sorted(df_final["region"].dropna().unique())
customer_segment_options = sorted(df_final["customer_segment"].dropna().unique())
cluster_options = sorted(df_final["cluster_label"].dropna().unique())
rfm_options = sorted(df_final["rfm_segment"].dropna().unique())

selected_region = st.sidebar.multiselect(
    "Region",
    options=region_options,
    default=region_options
)

selected_customer_segment = st.sidebar.multiselect(
    "Customer Segment",
    options=customer_segment_options,
    default=customer_segment_options
)

selected_cluster = st.sidebar.multiselect(
    "Cluster",
    options=cluster_options,
    default=cluster_options
)

selected_rfm_segment = st.sidebar.multiselect(
    "RFM Segment",
    options=rfm_options,
    default=rfm_options
)

min_date = df_final["order_date"].min().date()
max_date = df_final["order_date"].max().date()

date_range = st.sidebar.date_input(
    "Periode Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

if st.sidebar.button("Reset Filter"):
    st.rerun()

st.sidebar.markdown(
    """
    <div class="info-box">
    Gunakan filter di atas untuk memperbarui seluruh visualisasi dan tabel data.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLY FILTER
# ============================================================

filtered_df = df_final[
    (df_final["region"].isin(selected_region)) &
    (df_final["customer_segment"].isin(selected_customer_segment)) &
    (df_final["cluster_label"].isin(selected_cluster)) &
    (df_final["rfm_segment"].isin(selected_rfm_segment)) &
    (df_final["order_date"].dt.date >= start_date) &
    (df_final["order_date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("Tidak ada data yang sesuai dengan filter.")
    st.stop()

filtered_customer_ids = filtered_df["customer_id"].unique()

filtered_rfm = rfm[
    rfm["customer_id"].isin(filtered_customer_ids)
].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-title">Dashboard Segmentasi Pelanggan E-Commerce</div>
    <div class="dashboard-subtitle">
    Dashboard interaktif untuk melihat segmentasi pelanggan berdasarkan RFM, cluster pelanggan, dan rekomendasi strategi.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI CARDS
# ============================================================

total_sales = filtered_df["total_sales"].sum()
total_profit = filtered_df["profit"].sum()
total_orders = filtered_df["order_id"].nunique()
total_customers = filtered_df["customer_id"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Sales</div>
            <div class="kpi-value">{format_currency(total_sales)}</div>
            <div class="kpi-note">Nilai transaksi bersih</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Profit</div>
            <div class="kpi-value">{format_currency(total_profit)}</div>
            <div class="kpi-note">Total keuntungan</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{format_integer(total_orders)}</div>
            <div class="kpi-note">Jumlah transaksi unik</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{format_integer(total_customers)}</div>
            <div class="kpi-note">Jumlah pelanggan unik</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MONTHLY SALES
# ============================================================

st.markdown("### Tren Penjualan Bulanan")

monthly_filtered = build_monthly_sales(filtered_df)

fig_monthly = px.line(
    monthly_filtered,
    x="year_month",
    y="total_sales",
    markers=True,
    labels={
        "year_month": "Bulan",
        "total_sales": "Total Sales"
    }
)

fig_monthly.update_layout(
    height=330,
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig_monthly, use_container_width=True)


# ============================================================
# RFM SEGMENT AND CLUSTER
# ============================================================

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Segmentasi Pelanggan (RFM)")

    rfm_summary = (
        filtered_rfm
        .groupby("rfm_segment")
        .agg(
            total_customers=("customer_id", "nunique"),
            total_monetary=("monetary", "sum")
        )
        .reset_index()
        .sort_values("total_customers", ascending=False)
    )

    fig_rfm = px.pie(
        rfm_summary,
        names="rfm_segment",
        values="total_customers",
        hole=0.45
    )

    fig_rfm.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text="RFM Segment"
    )

    st.plotly_chart(fig_rfm, use_container_width=True)

with right_col:
    st.markdown("### Cluster Pelanggan")

    cluster_summary = (
        filtered_rfm
        .groupby("cluster_label")
        .agg(
            total_customers=("customer_id", "nunique"),
            total_monetary=("monetary", "sum")
        )
        .reset_index()
        .sort_values("cluster_label")
    )

    fig_cluster = px.bar(
        cluster_summary,
        x="cluster_label",
        y="total_customers",
        text="total_customers",
        labels={
            "cluster_label": "Cluster",
            "total_customers": "Jumlah Pelanggan"
        }
    )

    fig_cluster.update_traces(textposition="outside")
    fig_cluster.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_cluster, use_container_width=True)


# ============================================================
# BUSINESS PRIORITY AND PCA
# ============================================================

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Prioritas Bisnis")

    priority_summary = (
        filtered_rfm
        .groupby("business_priority")
        .agg(
            total_customers=("customer_id", "nunique")
        )
        .reset_index()
        .sort_values("total_customers", ascending=False)
    )

    fig_priority = px.bar(
        priority_summary,
        x="business_priority",
        y="total_customers",
        text="total_customers",
        labels={
            "business_priority": "Prioritas Bisnis",
            "total_customers": "Jumlah Pelanggan"
        }
    )

    fig_priority.update_traces(textposition="outside")
    fig_priority.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_priority, use_container_width=True)

with right_col:
    st.markdown("### Visualisasi Cluster PCA")

    fig_pca = px.scatter(
        filtered_rfm,
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_data=[
            "customer_name",
            "rfm_segment",
            "recency",
            "frequency",
            "monetary"
        ],
        labels={
            "pca_1": "PCA 1",
            "pca_2": "PCA 2",
            "cluster_label": "Cluster"
        }
    )

    fig_pca.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig_pca, use_container_width=True)


# ============================================================
# SUMMARY TABLES
# ============================================================

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Ringkasan Segmen RFM")

    rfm_summary_table = (
        filtered_rfm
        .groupby("rfm_segment")
        .agg(
            total_customers=("customer_id", "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum")
        )
        .reset_index()
        .sort_values("total_monetary", ascending=False)
    )

    st.dataframe(
        rfm_summary_table,
        use_container_width=True,
        hide_index=True
    )

with right_col:
    st.markdown("### Ringkasan Cluster")

    cluster_profile_table = (
        filtered_rfm
        .groupby("cluster_label")
        .agg(
            total_customers=("customer_id", "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_monetary=("monetary", "sum")
        )
        .reset_index()
        .sort_values("total_monetary", ascending=False)
    )

    st.dataframe(
        cluster_profile_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DETAIL TABLE
# ============================================================

st.markdown("### Detail Customer / Transaksi")

display_cols = [
    "customer_name",
    "region",
    "customer_segment",
    "product_category",
    "product_name",
    "quantity",
    "total_sales",
    "profit",
    "rfm_segment",
    "cluster_label",
    "business_priority",
    "recommended_strategy"
]

display_cols = [col for col in display_cols if col in filtered_df.columns]

st.dataframe(
    filtered_df[display_cols].sort_values("total_sales", ascending=False),
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_df[display_cols].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="hasil_segmentasi_pelanggan.csv",
    mime="text/csv"
)
