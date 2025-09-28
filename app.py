# app.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", layout="centered")

# =========================
# 1) Load model & feature list (dengan guard)
# =========================
@st.cache_resource
def load_all():
    try:
        assert os.path.exists("model_churn_histgb.pkl"), \
            "model_churn_histgb.pkl tidak ditemukan di root repo"
        pipe = joblib.load("model_churn_histgb.pkl")
    except Exception as e:
        st.error("❌ Gagal memuat model. Cek versi Python/sklearn & lokasi file.")
        st.write("CWD:", os.getcwd())
        st.write("Files:", os.listdir("."))
        st.exception(e)
        st.stop()

    try:
        with open("features_15.json") as f:
            feats = json.load(f)
        assert isinstance(feats, list) and len(feats) >= 15, "features_15.json invalid."
    except Exception as e:
        st.error("❌ Gagal memuat features_15.json.")
        st.exception(e)
        st.stop()

    return pipe, feats

pipe, FEATURES_15 = load_all()

# =========================
# 2) Konstanta & helper
# =========================
CORE12 = [
    "tenure","contract","monthly_charges","payment_method","paperless_billing",
    "internet_service","phone_service","multiple_lines",
    "online_security","device_protection","premium_tech_support","streaming_tv",
]
ADDON_COLS = ["online_security","device_protection","premium_tech_support","streaming_tv"]

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Bikin 3 fitur turunan + normalisasi tipe data ringan."""
    df = df.copy()

    # normalisasi string
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).strip()

    # fitur turunan
    miss_for_addons = [c for c in ADDON_COLS if c not in df.columns]
    for c in miss_for_addons:
        df[c] = np.nan
    df["addons_count"] = (df[ADDON_COLS] == "Yes").sum(axis=1).astype("Int64")

    if "payment_method" not in df.columns:
        df["payment_method"] = np.nan
    df["auto_pay"] = df["payment_method"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype("Int64")

    if "contract" not in df.columns:
        df["contract"] = np.nan
    df["contract_mtm"] = (df["contract"] == "Month-to-Month").astype("Int64")

    # cast numerik aman
    for c in ["tenure", "monthly_charges", "addons_count", "auto_pay", "contract_mtm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def ensure_schema(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pastikan semua kolom FEATURES_15 ada (kalau hilang, isi NaN) & urut sesuai model."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

# =========================
# 3) UI
# =========================
st.title("📉 Telco Churn Predictor")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("", ["Single customer", "Batch CSV"])
    with st.expander("ℹ️ Info"):
        st.write("- Model: **HistGradientBoosting** (pipeline OHE + imputer).")
        st.write("- Fitur (15): 12 inti + 3 turunan (`addons_count`, `auto_pay`, `contract_mtm`).")
        st.write("- File yang dibaca: `model_churn_histgb.pkl`, `features_15.json` (harus di root).")

# ---------- Single ----------
if mode == "Single customer":
    st.subheader("Single input")

    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=70.0, step=0.1)
        contract = st.selectbox("Contract", ["Month-to-Month","One year","Two year"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes","No"])
        internet_service  = st.selectbox("Internet service", ["DSL","Fiber optic","No"])
    with c2:
        phone_service     = st.selectbox("Phone service", ["Yes","No"])
        multiple_lines    = st.selectbox("Multiple lines", ["Yes","No"])
        online_security   = st.selectbox("Online security", ["Yes","No"])
        device_protection = st.selectbox("Device protection", ["Yes","No"])
        premium_tech_support = st.selectbox("Premium tech support", ["Yes","No"])
        streaming_tv      = st.selectbox("Streaming TV", ["Yes","No"])

    row = pd.DataFrame([{
        "tenure": tenure, "monthly_charges": monthly_charges,
        "contract": contract, "payment_method": payment_method, "paperless_billing": paperless_billing,
        "internet_service": internet_service, "phone_service": phone_service,
        "multiple_lines": multiple_lines, "online_security": online_security,
        "device_protection": device_protection, "premium_tech_support": premium_tech_support,
        "streaming_tv": streaming_tv
    }])

    feats = ensure_schema(add_engineered(row), FEATURES_15)

    st.divider()
    th = st.slider("Decision threshold", 0.05, 0.90, 0.35, 0.01)
    if st.button("Predict", use_container_width=True):
        try:
            proba = float(pipe.predict_proba(feats)[:, 1][0])
            pred  = ("🛑 **Churn**", "✅ **Stay**")[proba < th]
            st.metric("Churn probability", f"{proba:.3f}")
            st.write("Prediction:", pred, f"(threshold {th:.2f})")
        except Exception as e:
            st.error("Inference error.")
            st.exception(e)

# ---------- Batch ----------
else:
    st.subheader("Batch scoring (CSV)")
    st.markdown(
        "Kolom **minimal** (12 fitur inti):  \n"
        "`tenure, contract, monthly_charges, payment_method, paperless_billing, "
        "internet_service, phone_service, multiple_lines, online_security, "
        "device_protection, premium_tech_support, streaming_tv`  \n"
        "(Opsional `customer_id` ikut dibawa ke output.)"
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    col1, col2 = st.columns(2)
    with col1:
        topk = st.slider("Top-K (%) untuk shortlist", 5, 30, 10, 1)
    with col2:
        th = st.slider("Threshold (opsional)", 0.05, 0.90, 0.35, 0.01)

    if file is not None:
        try:
            raw = pd.read_csv(file)
        except Exception as e:
            st.error("Gagal membaca CSV.")
            st.exception(e)
            st.stop()

        id_col = "customer_id" if "customer_id" in raw.columns else None
        missing = [c for c in CORE12 if c not in raw.columns]
        if missing:
            st.error(f"Kolom wajib hilang: {missing}")
            st.stop()

        feats = ensure_schema(add_engineered(raw), FEATURES_15)
        try:
            score = pipe.predict_proba(feats)[:, 1]
        except Exception as e:
            st.error("Inference error saat batch.")
            st.exception(e)
            st.stop()

        out = pd.DataFrame({
            "score": score,
            "label_pred": (score >= th).astype(int)
        })
        if id_col:
            out[id_col] = raw[id_col].values

        out = out.sort_values("score", ascending=False)
        k = int(np.ceil(len(out) * (topk / 100)))
        st.success(f"Top-{topk}% shortlist (n={k})")
        st.dataframe(out.head(k), use_container_width=True)

        st.download_button(
            "⬇️ Download all scores (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="scores_churn.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer kecil
st.markdown(
    "<br><sub>Model pipeline di-load dari <code>model_churn_histgb.pkl</code> · "
    "Fitur align dengan <code>features_15.json</code>.</sub>", unsafe_allow_html=True
)
