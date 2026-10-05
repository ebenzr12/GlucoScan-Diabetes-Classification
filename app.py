import glob

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =====================  SESUAIKAN BAGIAN INI  =====================
MODEL_PATH = "artifacts/model.pkl"             # model Decision Tree hasil train_model.py
ENCODER_PATH = "artifacts/label_encoder.pkl"   # untuk mengubah kode prediksi jadi nama kelas
DATA_PATH = "data/diabetes.csv"                # dataset (ganti nama file CSV jadi ini)
FEATURES = ["HbA1c", "BMI", "Usia", "Chol", "TG"]  # dipakai bila model tidak menyimpan nama kolom
TARGET = "CLASS"                               # kolom label di dataset
LABEL_MAP = {"n": "Non-Diabetes", "p": "Prediabetes", "y": "Diabetes"}  # isi kolom CLASS
TEST_ACC = "98,80%"
CV_ACC = "95,3%"
CV_STD = "±6,19%"
# ==================================================================

BLUE, TEAL, AMBER, INK, MUTED = "#1F6FB2", "#2A9D8F", "#E9A23B", "#18324A", "#6B8197"
ORANGE = "#E0703A"

st.set_page_config(page_title="GlucoScan", page_icon="🩸", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:wght@600;700&display=swap');
html, body, [class*="css"], p, label, span, div { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .brand-name, .kpi-value { font-family: 'Fraunces', serif !important; }
.stApp { background: #EEF4F8; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1150px; }

section[data-testid="stSidebar"] { background: #F8FBFD; border-right: 1px solid #DCE7EF; }
.brand { display:flex; align-items:center; gap:12px; margin: 6px 0 2px 0; }
.logo { width:38px; height:38px; border-radius:50%; border:7px solid #1F6FB2; box-sizing:border-box; }
.brand-name { font-size:1.45rem; color:#18324A; font-weight:700; line-height:1; }
.brand-sub { color:#6B8197; font-size:.85rem; margin: 6px 0 20px 50px; }
.side-label { font-weight:700; color:#18324A; margin-top:18px; font-size:.9rem; }
.side-val { color:#6B8197; font-size:.85rem; }

h2 { color:#18324A !important; margin-bottom:0 !important; }
.page-sub { color:#6B8197; margin: 2px 0 1.4rem 0; }
.section-title { font-family:'Fraunces',serif; font-size:1.25rem; color:#1F6FB2; margin: 1.6rem 0 .5rem 0; }

.card { background:#fff; border:1px solid #DCE7EF; border-top:4px solid #1F6FB2;
        border-radius:14px; padding:16px 20px; box-shadow:0 2px 8px rgba(24,50,74,.05); }
.card.teal { border-top-color:#2A9D8F; }
.card.amber { border-top-color:#E9A23B; }
.card.orange { border-top-color:#E0703A; }
.kpi-label { color:#6B8197; font-size:.85rem; }
.kpi-value { color:#18324A; font-size:2rem; font-weight:700; line-height:1.25; }
.kpi-note { color:#6B8197; font-size:.78rem; }

div[data-testid="stForm"] { background:#fff; border:1px solid #DCE7EF; border-radius:14px; padding:20px; }
.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  background:#1F6FB2; color:#fff; border:none; border-radius:999px; padding:.55rem 1.6rem; font-weight:700; }
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background:#17598F; color:#fff; }

.result { border-radius:14px; padding:20px 24px; margin-top:18px; border:1px solid; }
.result.dia { background:#FCE8DD; border-color:#E0703A; }
.result.pre { background:#FDF1DC; border-color:#E9A23B; }
.result.neg { background:#DFF3F0; border-color:#2A9D8F; }
.result h3 { margin:0 0 4px 0; color:#18324A; }
.bar-bg { background:#fff; border-radius:8px; height:10px; margin-top:10px; }
.bar-fg { height:10px; border-radius:8px; }
.range-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px dashed #DCE7EF; }
.footer { border-top:1px solid #DCE7EF; margin-top:3rem; padding-top:1rem; text-align:center; color:#6B8197; font-size:.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------- helper -----------------------------
@st.cache_resource
def load_model():
    path = MODEL_PATH
    if path is None:
        found = sorted(glob.glob("artifacts/*.joblib") + glob.glob("artifacts/*.pkl"))
        if not found:
            raise FileNotFoundError("Tidak ada file .joblib atau .pkl di folder artifacts")
        path = found[0]
    return joblib.load(path)


@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)


def build_X(model, values):
    """Susun input sesuai nama kolom yang dipakai saat training."""
    alias = {"hba1c": "HbA1c", "bmi": "BMI", "usia": "Usia", "age": "Usia", "chol": "Chol", "tg": "TG"}
    cols = list(getattr(model, "feature_names_in_", FEATURES))
    row = []
    for c in cols:
        key = alias.get(str(c).strip().lower())
        if key is None:
            raise ValueError(f"Kolom '{c}' dipakai model tetapi tidak ada di form")
        row.append(values[key])
    return pd.DataFrame([row], columns=cols)


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception:
        return None
    if TARGET not in df.columns:
        return None
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"AGE": "Usia"})
    df = df[df["Chol"] != 0].reset_index(drop=True)  # sama dengan saat training
    df["Status"] = df[TARGET].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["Status"])
    return df


def style_fig(fig, h=360):
    fig.update_layout(
        template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fff",
        font=dict(family="DM Sans", color=INK), margin=dict(l=10, r=10, t=30, b=10),
        legend_title_text="",
    )
    return fig


COLORS = {"Diabetes": ORANGE, "Prediabetes": AMBER, "Non-Diabetes": TEAL}


def kpi(label, value, note="", tone=""):
    st.markdown(
        f'<div class="card {tone}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def title(text, sub):
    st.markdown(f"## {text}")
    st.markdown(f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


df = load_data()

# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="logo"></div><div class="brand-name">GlucoScan</div></div>'
        '<div class="brand-sub">Skrining Diabetes berbasis ML</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Menu",
        ["Dashboard", "Eksplorasi Data", "Prediksi", "Penjelasan"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="side-label">Tugas</div><div class="side-val">Non-Diabetes / Prediabetes / Diabetes</div>'
        '<div class="side-label">Fitur</div><div class="side-val">HbA1c, BMI, Usia, Kolesterol, Trigliserida</div>',
        unsafe_allow_html=True,
    )

# ----------------------------- Dashboard -----------------------------
if page == "Dashboard":
    title("Dashboard", "Ringkasan data dan performa model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Jumlah Data", f"{len(df):,}" if df is not None else "-", "Baris pada dataset")
    with c2:
        pct = (df["Status"].eq("Diabetes").mean() * 100) if df is not None else None
        kpi("Proporsi Diabetes", f"{pct:.1f}%" if pct is not None else "-", "Dari seluruh data", "amber")
    with c3:
        kpi("Akurasi Data Uji", TEST_ACC, "Hold-out test set", "teal")
    with c4:
        kpi("Akurasi Cross-Validation", CV_ACC, f"Simpangan {CV_STD}", "teal")

    if df is None:
        st.warning(f"Dataset belum terbaca. Taruh file CSV di `{DATA_PATH}` dan pastikan kolom `{TARGET}` ada.")
    else:
        l, r = st.columns(2)
        with l:
            section("Distribusi Kelas")
            cnt = df["Status"].value_counts().reset_index()
            cnt.columns = ["Status", "Jumlah"]
            fig = px.bar(cnt, x="Status", y="Jumlah", color="Status", color_discrete_map=COLORS, text="Jumlah")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with r:
            section("Proporsi Kelas")
            fig = px.pie(cnt, names="Status", values="Jumlah", hole=0.55, color="Status", color_discrete_map=COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        section("Contoh Data")
        st.dataframe(df.head(10), use_container_width=True)
    st.info("Cross-validation lebih mewakili performa sebenarnya dibanding akurasi data uji. Simpangan yang besar "
            "berarti hasil bisa berubah antar pembagian data. Data juga timpang: kelas Diabetes mendominasi, "
            "sedangkan Prediabetes dan Non-Diabetes jauh lebih sedikit.")

# ----------------------------- Eksplorasi -----------------------------
elif page == "Eksplorasi Data":
    title("Eksplorasi Data", "Pola dan hubungan antar variabel pada data pasien")
    if df is None:
        st.warning(f"Dataset belum terbaca. Taruh file CSV di `{DATA_PATH}` dan pastikan kolom `{TARGET}` ada.")
    else:
        feats = [f for f in FEATURES if f in df.columns]
        section("Rata-rata Fitur per Kelas")
        st.dataframe(df.groupby("Status")[feats].mean().round(2), use_container_width=True)

        section("Korelasi Fitur terhadap Status Diabetes")
        tmp = df[feats].copy()
        tmp["Diabetes"] = (df["Status"] == "Diabetes").astype(int)
        corr = tmp.corr()["Diabetes"].drop("Diabetes").sort_values()
        fig = px.bar(corr, orientation="h", labels={"value": "Korelasi", "index": ""})
        fig.update_traces(marker_color=[BLUE if v >= 0 else AMBER for v in corr.values])
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

        l, r = st.columns(2)
        with l:
            section("Distribusi Fitur")
            f1 = st.selectbox("Pilih fitur", feats, key="hist")
            fig = px.histogram(df, x=f1, color="Status", barmode="overlay", opacity=0.7,
                               color_discrete_map=COLORS, nbins=30)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with r:
            section("Sebaran per Kelas (Box Plot)")
            f2 = st.selectbox("Pilih fitur", feats, key="box")
            fig = px.box(df, x="Status", y=f2, color="Status", color_discrete_map=COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), use_container_width=True)

        section("Hubungan Dua Fitur")
        a, b = st.columns(2)
        xa = a.selectbox("Sumbu X", feats, index=0)
        ya = b.selectbox("Sumbu Y", feats, index=min(1, len(feats) - 1))
        fig = px.scatter(df, x=xa, y=ya, color="Status", color_discrete_map=COLORS, opacity=0.7)
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

        section("Heatmap Korelasi Antar Fitur")
        fig = px.imshow(df[feats].corr().round(2), text_auto=True, color_continuous_scale="Blues", zmin=-1, zmax=1)
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

# ----------------------------- Prediksi -----------------------------
elif page == "Prediksi":
    title("Prediksi Diabetes", "Masukkan data pasien untuk memprediksi Non-Diabetes, Prediabetes, atau Diabetes")
    with st.form("form"):
        a, b, c = st.columns(3)
        hba1c = a.number_input("HbA1c (%)", 3.0, 20.0, 5.5, 0.1)
        bmi = b.number_input("BMI (kg/m²)", 10.0, 60.0, 24.0, 0.1)
        usia = c.number_input("Usia (tahun)", 1, 120, 40)
        d, e, _ = st.columns(3)
        chol = d.number_input("Kolesterol total (mmol/L)", 1.0, 15.0, 4.5, 0.1)
        tg = e.number_input("Trigliserida (mmol/L)", 0.1, 15.0, 1.5, 0.1)
        submit = st.form_submit_button("Prediksi Sekarang")

    if submit:
        try:
            model = load_model()
            X = build_X(model, {"HbA1c": hba1c, "BMI": bmi, "Usia": usia, "Chol": chol, "TG": tg})
            le = load_encoder()
            pred = model.predict(X)[0]
            label = le.inverse_transform([pred])[0]
            cls, col = {"Diabetes": ("dia", ORANGE), "Prediabetes": ("pre", AMBER)}.get(label, ("neg", TEAL))
            bar = ""
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                prob = float(proba.max())
                detail = " · ".join(f"{c}: {p*100:.0f}%" for c, p in zip(le.classes_, proba))
                bar = (f'<div>Keyakinan model: <b>{prob*100:.1f}%</b></div><div class="bar-bg">'
                       f'<div class="bar-fg" style="width:{prob*100:.0f}%;background:{col}"></div></div>'
                       f'<div class="kpi-note" style="margin-top:8px">{detail}</div>')
            st.markdown(f'<div class="result {cls}"><h3>{label}</h3>{bar}</div>', unsafe_allow_html=True)
            if hba1c >= 6.5:
                st.caption("Catatan: HbA1c ≥ 6,5% sudah masuk ambang diagnosis diabetes menurut pedoman ADA.")
            st.caption("Hasil ini alat bantu, bukan diagnosis medis. Konsultasikan dengan tenaga kesehatan.")
        except Exception as ex:
            st.error(f"Gagal memuat atau menjalankan model: {ex}")

# ----------------------------- Penjelasan -----------------------------
else:
    title("Penjelasan", "Arti tiap fitur, cara membaca hasil, dan batasan model")

    section("Arti Setiap Fitur")
    with st.expander("HbA1c: rata-rata gula darah 2-3 bulan terakhir", expanded=True):
        st.markdown(
            "Mengukur persentase hemoglobin yang berikatan dengan glukosa. Biasanya fitur paling berpengaruh.\n"
            '<div class="range-row"><span>Normal</span><b>&lt; 5,7%</b></div>'
            '<div class="range-row"><span>Prediabetes</span><b>5,7 – 6,4%</b></div>'
            '<div class="range-row"><span>Diabetes</span><b>≥ 6,5%</b></div>',
            unsafe_allow_html=True,
        )
    with st.expander("BMI: indeks massa tubuh"):
        st.markdown(
            "Berat (kg) dibagi tinggi (m) kuadrat. Kelebihan berat badan meningkatkan resistensi insulin.\n"
            '<div class="range-row"><span>Kurus</span><b>&lt; 18,5</b></div>'
            '<div class="range-row"><span>Normal</span><b>18,5 – 24,9</b></div>'
            '<div class="range-row"><span>Berlebih</span><b>25 – 29,9</b></div>'
            '<div class="range-row"><span>Obesitas</span><b>≥ 30</b></div>'
            "<br><small>Kategori WHO. Untuk populasi Asia batasnya sering lebih rendah (23 dan 25).</small>",
            unsafe_allow_html=True,
        )
    with st.expander("Usia"):
        st.write("Risiko diabetes tipe 2 naik seiring usia, terutama setelah 45 tahun.")
    with st.expander("Kolesterol total"):
        st.write("Kadar lemak dalam darah. Nilai yang diinginkan di bawah 5,2 mmol/L (sekitar 200 mg/dL). "
                 "Diabetes sering disertai profil lemak darah yang tidak sehat.")
    with st.expander("Trigliserida (TG)"):
        st.write("Jenis lemak darah yang biasanya naik saat gula darah tidak terkontrol. "
                 "Normal di bawah 1,7 mmol/L (sekitar 150 mg/dL).")

    section("Cara Membaca Hasil")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Non-Diabetes", "Risiko rendah", "Pola data mirip pasien non-diabetes", "teal")
    with c2:
        kpi("Prediabetes", "Perlu waspada", "Gula darah di atas normal, belum masuk ambang diabetes", "amber")
    with c3:
        kpi("Diabetes", "Risiko tinggi", "Pola data mirip pasien diabetes", "orange")

    section("Performa dan Batasan")
    st.markdown(
        f'<div class="card">Akurasi data uji <b>{TEST_ACC}</b>, cross-validation <b>{CV_ACC}</b> ({CV_STD}). '
        "Model dilatih pada satu dataset tertentu, jadi belum tentu berlaku untuk populasi lain. "
        "Aplikasi ini untuk keperluan pembelajaran dan bukan pengganti pemeriksaan laboratorium atau dokter.</div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="footer">GlucoScan — Klasifikasi Diabetes | Dibangun dengan Streamlit</div>', unsafe_allow_html=True)
