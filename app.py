# ============================================================
# DASHBOARD STREAMLIT
# Project: Segmentasi Pelanggan E-Commerce Berbasis RFM dan K-Means
# Fokus utama: Segmentasi Pelanggan
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Segmentasi Pelanggan",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# STYLE SEDERHANA
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #64748b;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0px 2px 10px rgba(15, 23, 42, 0.06);
    }
    .small-note {
        font-size: 13px;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNGSI UTILITAS
# ============================================================

@st.cache_data(show_spinner=False)
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def to_numeric_safe(series: pd.Series) -> pd.Series:
    if series.dtype == "object":
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
    return pd.to_numeric(series, errors="coerce")


def safe_qcut_score(series: pd.Series, labels: list[int]) -> pd.Series:
    return pd.qcut(
        series.rank(method="first"),
        q=5,
        labels=labels,
        duplicates="drop"
    ).astype(int)


@st.cache_data(show_spinner=True)
def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw_df)

    required_cols = ["order_id", "order_date", "customer_name", "quantity", "unit_price"]
    missing_required = [col for col in required_cols if col not in df.columns]

    if "total_sales" not in df.columns and {"quantity", "unit_price"}.issubset(df.columns):
        df["total_sales"] = to_numeric_safe(df["quantity"]) * to_numeric_safe(df["unit_price"])

    if missing_required:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(missing_required)
        )

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    numeric_cols = ["quantity", "unit_price", "total_sales"]
    for col in numeric_cols:
        df[col] = to_numeric_safe(df[col])

    if "customer_id" not in df.columns:
        df["customer_id"] = (
            df["customer_name"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    df["gross_sales"] = df["quantity"] * df["unit_price"]

    df = df.dropna(subset=[
        "order_id",
        "order_date",
        "customer_id",
        "customer_name",
        "quantity",
        "unit_price",
        "total_sales"
    ])

    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] > 0]
    df = df[df["total_sales"] > 0]
    df = df.drop_duplicates()

    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    return df


@st.cache_data(show_spinner=True)
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            customer_name=("customer_name", "first"),
            recency=("order_date", lambda x: (snapshot_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("total_sales", "sum")
        )
        .reset_index()
    )

    rfm["r_score"] = safe_qcut_score(rfm["recency"], [5, 4, 3, 2, 1])
    rfm["f_score"] = safe_qcut_score(rfm["frequency"], [1, 2, 3, 4, 5])
    rfm["m_score"] = safe_qcut_score(rfm["monetary"], [1, 2, 3, 4, 5])

    rfm["rfm_total_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

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


@st.cache_data(show_spinner=True)
def build_rfm_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rfm.groupby("rfm_segment")
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

    summary["customer_percentage"] = (
        summary["total_customers"] / rfm["customer_id"].nunique() * 100
    )

    summary["monetary_percentage"] = (
        summary["total_monetary"] / rfm["monetary"].sum() * 100
    )

    return summary


@st.cache_data(show_spinner=True)
def evaluate_and_cluster(rfm: pd.DataFrame):
    cluster_features = ["recency", "frequency", "monetary"]
    X = rfm[cluster_features].copy()

    for col in cluster_features:
        X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k = min(6, len(rfm) - 1)
    results = []

    if max_k >= 2:
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

            results.append({
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": score
            })

        cluster_eval = pd.DataFrame(results)
        best_k = int(cluster_eval.sort_values("silhouette_score", ascending=False).iloc[0]["k"])
    else:
        cluster_eval = pd.DataFrame({"k": [1], "inertia": [0], "silhouette_score": [0]})
        best_k = 1

    if best_k > 1:
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
    else:
        labels = np.zeros(len(rfm), dtype=int)

    rfm_clustered = rfm.copy()
    rfm_clustered["cluster"] = labels
    rfm_clustered["cluster_label"] = rfm_clustered["cluster"].apply(lambda x: f"Cluster {x}")

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_scaled)
    rfm_clustered["pca_1"] = pca_result[:, 0]
    rfm_clustered["pca_2"] = pca_result[:, 1]

    cluster_profile = (
        rfm_clustered.groupby("cluster_label")
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

    return rfm_clustered, cluster_eval, cluster_profile, best_k


@st.cache_data(show_spinner=True)
def build_strategy(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    strategy = rfm_clustered.copy()

    strategy["customer_priority"] = np.select(
        [
            strategy["rfm_segment"] == "Pelanggan Bernilai Tinggi",
            strategy["rfm_segment"] == "Pelanggan Potensial",
            strategy["rfm_segment"] == "Pelanggan Reguler",
            strategy["rfm_segment"] == "Pelanggan Bernilai Rendah"
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

    strategy["recommended_strategy"] = strategy["rfm_segment"].map(strategy_map)

    return strategy


@st.cache_data(show_spinner=True)
def merge_to_transactions(df: pd.DataFrame, strategy: pd.DataFrame) -> pd.DataFrame:
    segment_result = strategy[
        [
            "customer_id",
            "rfm_total_score",
            "rfm_segment",
            "cluster_label",
            "customer_priority",
            "recommended_strategy"
        ]
    ]

    final_df = df.merge(segment_result, on="customer_id", how="left")
    return final_df


# ============================================================
# LOAD DATA DARI FILE LOKAL ATAU UPLOAD
# ============================================================

st.markdown('<p class="main-title">Dashboard Segmentasi Pelanggan E-Commerce</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Fokus analisis: segmentasi pelanggan berdasarkan Recency, Frequency, dan Monetary.</p>',
    unsafe_allow_html=True
)

DATA_PATH = Path("data/global_ecommerce_sales.csv")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset CSV",
    type=["csv"],
    help="Upload dataset jika file CSV belum tersedia di repository GitHub."
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset berhasil diupload.")
elif DATA_PATH.exists():
    raw_df = pd.read_csv(DATA_PATH)
    st.sidebar.success("Dataset dibaca dari folder data/.")
else:
    st.info(
        "Upload file CSV melalui sidebar, atau simpan file dataset di GitHub pada path: data/global_ecommerce_sales.csv"
    )
    st.stop()


# ============================================================
# PIPELINE ANALISIS
# ============================================================

try:
    df_clean = prepare_data(raw_df)
    rfm = build_rfm(df_clean)
    rfm_summary = build_rfm_summary(rfm)
    rfm_clustered, cluster_eval, cluster_profile, best_k = evaluate_and_cluster(rfm)
    customer_strategy = build_strategy(rfm_clustered)
    final_df = merge_to_transactions(df_clean, customer_strategy)
except Exception as error:
    st.error(f"Terjadi error saat memproses data: {error}")
    st.stop()


# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("Filter Segmentasi")

segment_options = sorted(customer_strategy["rfm_segment"].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Segmen RFM",
    segment_options,
    default=segment_options
)

cluster_options = sorted(customer_strategy["cluster_label"].dropna().unique().tolist())
selected_clusters = st.sidebar.multiselect(
    "Cluster",
    cluster_options,
    default=cluster_options
)

min_score, max_score = int(customer_strategy["rfm_total_score"].min()), int(customer_strategy["rfm_total_score"].max())
score_range = st.sidebar.slider(
    "Rentang skor RFM",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score)
)

search_name = st.sidebar.text_input("Cari nama pelanggan")

filtered_strategy = customer_strategy[
    (customer_strategy["rfm_segment"].isin(selected_segments)) &
    (customer_strategy["cluster_label"].isin(selected_clusters)) &
    (customer_strategy["rfm_total_score"].between(score_range[0], score_range[1]))
].copy()

if search_name.strip():
    filtered_strategy = filtered_strategy[
        filtered_strategy["customer_name"].str.contains(search_name, case=False, na=False)
    ]

filtered_customers = filtered_strategy["customer_id"].unique()
filtered_transactions = final_df[final_df["customer_id"].isin(filtered_customers)].copy()


# ============================================================
# KPI UTAMA SEGMENTASI
# ============================================================

st.subheader("Ringkasan Segmentasi")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Pelanggan", f"{filtered_strategy['customer_id'].nunique():,.0f}")
col2.metric("Total Monetary", f"{filtered_strategy['monetary'].sum():,.0f}")
col3.metric("Rata-rata Frequency", f"{filtered_strategy['frequency'].mean():.2f}")
col4.metric("Cluster Terbaik", f"{best_k}")

st.caption("Catatan: clustering menggunakan tiga variabel inti, yaitu recency, frequency, dan monetary.")


# ============================================================
# TABS DASHBOARD
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Segmentasi RFM",
    "Clustering Pelanggan",
    "Strategi Pelanggan",
    "Data Detail"
])


with tab1:
    st.subheader("Segmentasi RFM")

    rfm_summary_filtered = (
        filtered_strategy.groupby("rfm_segment")
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

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            rfm_summary_filtered,
            x="total_customers",
            y="rfm_segment",
            orientation="h",
            title="Jumlah Pelanggan per Segmen RFM",
            labels={
                "total_customers": "Jumlah Pelanggan",
                "rfm_segment": "Segmen RFM"
            }
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            rfm_summary_filtered,
            x="total_monetary",
            y="rfm_segment",
            orientation="h",
            title="Total Monetary per Segmen RFM",
            labels={
                "total_monetary": "Total Monetary",
                "rfm_segment": "Segmen RFM"
            }
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(rfm_summary_filtered, use_container_width=True)


with tab2:
    st.subheader("Clustering Pelanggan")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            cluster_eval,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Method"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(
            cluster_eval,
            x="k",
            y="silhouette_score",
            markers=True,
            title="Silhouette Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered_strategy,
        x="pca_1",
        y="pca_2",
        color="cluster_label",
        hover_data=["customer_name", "rfm_segment", "recency", "frequency", "monetary"],
        title="Visualisasi Cluster Pelanggan dengan PCA"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Profil Cluster")
    cluster_profile_filtered = (
        filtered_strategy.groupby("cluster_label")
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
    st.dataframe(cluster_profile_filtered, use_container_width=True)


with tab3:
    st.subheader("Prioritas dan Rekomendasi Strategi")

    strategy_summary = (
        filtered_strategy.groupby(["rfm_segment", "customer_priority", "recommended_strategy"])
        .agg(
            total_customers=("customer_id", "nunique"),
            total_monetary=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean")
        )
        .reset_index()
        .sort_values("total_monetary", ascending=False)
    )

    fig = px.bar(
        strategy_summary,
        x="total_customers",
        y="customer_priority",
        orientation="h",
        color="rfm_segment",
        title="Prioritas Pelanggan Berdasarkan Segmen"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(strategy_summary, use_container_width=True)


with tab4:
    st.subheader("Data Detail Pelanggan dan Transaksi")

    st.write("Data pelanggan hasil segmentasi")
    st.dataframe(
        filtered_strategy[
            [
                "customer_name",
                "recency",
                "frequency",
                "monetary",
                "r_score",
                "f_score",
                "m_score",
                "rfm_total_score",
                "rfm_segment",
                "cluster_label",
                "customer_priority",
                "recommended_strategy"
            ]
        ],
        use_container_width=True
    )

    st.write("Data transaksi setelah diberi label segmentasi")
    cols_to_show = [
        "order_id",
        "order_date",
        "customer_name",
        "quantity",
        "unit_price",
        "total_sales",
        "rfm_segment",
        "cluster_label",
        "customer_priority"
    ]
    existing_cols = [col for col in cols_to_show if col in filtered_transactions.columns]
    st.dataframe(filtered_transactions[existing_cols], use_container_width=True)

    csv_data = filtered_transactions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data transaksi hasil segmentasi",
        data=csv_data,
        file_name="hasil_segmentasi_pelanggan.csv",
        mime="text/csv"
    )
