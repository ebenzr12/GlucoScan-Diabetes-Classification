"""Latih model Decision Tree klasifikasi status diabetes dan simpan ke artifacts/.

Jalankan sekali dari folder proyek:
    python train_model.py
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Urutan fitur ini HARUS sama dengan yang dipakai di app.py
FITUR = ["HbA1c", "BMI", "AGE", "Chol", "TG"]

LABEL_MAP = {"N": "Non-Diabetes", "P": "Prediabetes", "Y": "Diabetes"}


def cari_dataset() -> Path:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Tidak ada file .csv di folder {DATA_DIR}. "
            "Letakkan dataset diabetes di folder data/ lalu jalankan ulang."
        )
    return csv_files[0]


def main() -> None:
    path = cari_dataset()
    print(f"Membaca dataset: {path.name}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Rapikan label
    df["CLASS"] = df["CLASS"].astype(str).str.strip().str.upper().map(LABEL_MAP)
    df = df.dropna(subset=["CLASS"])

    # Hapus nilai mustahil (Chol = 0)
    sebelum = len(df)
    df = df[df["Chol"] != 0].reset_index(drop=True)
    print(f"Baris dihapus karena Chol = 0: {sebelum - len(df)} | Sisa: {len(df)}")

    X = df[FITUR]
    le = LabelEncoder()
    y = le.fit_transform(df["CLASS"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nAkurasi data uji: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    cv = cross_val_score(
        DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
        X, y, cv=5,
    )
    print(f"Cross-validation 5-fold: {cv.mean():.2%} (+/- {cv.std():.2%})")

    # Simpan model akhir (dilatih ulang pada seluruh data) + pelengkapnya
    final_model = DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"
    ).fit(X, y)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, ARTIFACTS_DIR / "model.pkl")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.pkl")
    joblib.dump(FITUR, ARTIFACTS_DIR / "fitur.pkl")

    print(f"\nArtifacts tersimpan di: {ARTIFACTS_DIR}")
    print(f"Versi scikit-learn: {sklearn.__version__}  <- samakan di requirements.txt")


if __name__ == "__main__":
    main()
