# Dashboard Segmentasi Pelanggan E-Commerce

Dashboard ini fokus pada mata kuliah **Segmentasi Pelanggan**. Analisis menggunakan metode **RFM** dan **K-Means Clustering**.

## Fokus Analisis

Pelanggan disegmentasikan berdasarkan tiga variabel inti:

1. **Recency**: jarak hari sejak pembelian terakhir.
2. **Frequency**: jumlah transaksi pelanggan.
3. **Monetary**: total nilai pembelian pelanggan.

## Struktur File

```text
streamlit_segmentasi_pelanggan/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
└── data/
    └── README_DATA.txt
```

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Deploy ke Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Upload semua file project ke repository tersebut.
3. Simpan dataset di `data/global_ecommerce_sales.csv`, atau gunakan fitur upload CSV di sidebar aplikasi.
4. Buka Streamlit Community Cloud.
5. Klik **Create app**.
6. Pilih repository GitHub.
7. Isi main file path dengan:

```text
app.py
```

8. Klik **Deploy**.

## Catatan

File `requirements.txt` wajib ada agar Streamlit Cloud tahu library Python yang perlu diinstal.
