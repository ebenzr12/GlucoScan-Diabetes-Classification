# Klasifikasi Status Diabetes (Streamlit)

Aplikasi web untuk memprediksi status pasien — **Non-Diabetes**, **Prediabetes**, atau **Diabetes** —
menggunakan model Decision Tree dengan 5 fitur: HbA1c, BMI, AGE, Chol, dan TG.

> Hanya untuk tujuan edukasi, bukan alat diagnosis medis.

## Struktur
```
├── .streamlit/config.toml   # tema aplikasi
├── artifacts/               # model.pkl, label_encoder.pkl, fitur.pkl (hasil train_model.py)
├── data/                    # taruh dataset .csv di sini
├── train_model.py           # melatih model & menyimpan artifacts
├── app.py                   # aplikasi Streamlit
└── requirements.txt
```

## Cara menjalankan
```bash
pip install -r requirements.txt
python train_model.py     # sekali saja, menghasilkan folder artifacts/
streamlit run app.py
```

## Deploy
Upload semua file (termasuk folder `artifacts/`) ke GitHub, lalu deploy lewat
[Streamlit Community Cloud](https://share.streamlit.io) dengan main file `app.py`.
