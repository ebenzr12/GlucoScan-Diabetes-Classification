import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model_diabetes.pkl")

model = joblib.load(model_path)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

    /* Background */
    .stApp {
        background: #f5f7fb;
    }

    /* Container */
    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .header {
        background: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
    }

    .header h1 {
        margin: 0;
        color: #263b80;
        font-size: 38px;
        font-weight: 700;
    }

    .header p {
        margin-top: 10px;
        color: #667085;
        font-size: 16px;
    }

    /* Form */
    .form-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .section-title {
        color: #263b80;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Label */
    label {
        font-weight: 600 !important;
        color: #344054 !important;
    }

    /* Input */
    div[data-baseweb="input"] {
        border-radius: 10px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 50px;
        border-radius: 12px;
        background: #4169e1;
        color: white;
        border: none;
        font-size: 17px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #3155c7;
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <h1>🩺 Prediksi Diabetes</h1>
    <p>
        Sistem prediksi kondisi diabetes berdasarkan data kesehatan pasien
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# FORM
# =========================
st.markdown("""
<div class="form-card">
    <div class="section-title">📋 Data Pasien</div>
</div>
""", unsafe_allow_html=True)


# Dua kolom
col1, col2 = st.columns(2)

with col1:

    HbA1c = st.number_input(
        "HbA1c",
        min_value=0.0,
        step=0.01,
        help="Masukkan nilai HbA1c pasien"
    )

    AGE = st.number_input(
        "AGE",
        min_value=0,
        step=1,
        help="Masukkan usia pasien"
    )

    Chol = st.number_input(
        "Chol",
        min_value=0.0,
        step=0.01,
        help="Masukkan nilai kolesterol"
    )


with col2:

    BMI = st.number_input(
        "BMI",
        min_value=0.0,
        step=0.01,
        help="Masukkan nilai BMI"
    )

    TG = st.number_input(
        "TG",
        min_value=0.0,
        step=0.01,
        help="Masukkan nilai triglycerides"
    )


# =========================
# PREDICTION
# =========================
if st.button("🔍  Prediksi Sekarang"):

    data = pd.DataFrame(
        [[HbA1c, BMI, AGE, Chol, TG]],
        columns=["HbA1c", "BMI", "AGE", "Chol", "TG"]
    )

    prediction = model.predict(data)[0]

    kelas = {
        0: "Diabetes",
        1: "Non-Diabetes",
        2: "Prediabetes"
    }

    hasil = kelas[prediction]

    st.markdown("---")

    if prediction == 0:
        st.error(f"🔴 Hasil Prediksi: **{hasil}**")

    elif prediction == 1:
        st.success(f"🟢 Hasil Prediksi: **{hasil}**")

    elif prediction == 2:
        st.warning(f"🟡 Hasil Prediksi: **{hasil}**")


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    Machine Learning • Streamlit • Sistem Prediksi Diabetes
</div>
""", unsafe_allow_html=True)