# streamlit_catboost_churn.py
# Clean final app: uses saved model; single & batch predict; persistent Excel store; delete works reliably;
# compact charts, clear prediction outputs, no demo buttons.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------------- CONFIG ----------------
MODEL_CANDIDATES = [
    "./catboost_churn.pkl",
    "./churn_pipeline.pkl",
    "./churn_pipeline_colab.pkl",
    "/mnt/data/catboost_churn.pkl",
    "/content/catboost_churn.pkl"
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
STORE_PATH = "predictions_store.xlsx"
CSV_BACKUP = "predictions_store.csv"

TELCO_COLUMNS = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
]

# ---------------- Load model ----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = None
if MODEL_PATH:
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model at {MODEL_PATH}: {e}")
        model = None
else:
    st.warning("No saved model found. Place catboost_churn.pkl or churn_pipeline.pkl in app folder.")

# ---------------- Store helpers ----------------
def read_store():
    if os.path.exists(STORE_PATH):
        try:
            return pd.read_excel(STORE_PATH)
        except Exception:
            try:
                return pd.read_csv(CSV_BACKUP)
            except Exception:
                return pd.DataFrame()
    else:
        return pd.DataFrame()

def write_store(df):
    try:
        df.to_excel(STORE_PATH, index=False)
    except Exception as e:
        st.error(f"Failed to save Excel: {e}")
    try:
        df.to_csv(CSV_BACKUP, index=False)
    except Exception as e:
        st.error(f"Failed to save CSV backup: {e}")

def append_to_store(new_df):
    df = read_store()
    if df.empty:
        out = new_df.reset_index(drop=True)
    else:
        out = pd.concat([df, new_df], ignore_index=True)
    write_store(out)
    return out

# ---------------- Prediction helper ----------------
def predict_df(mdl, X):
    if mdl is None:
        raise RuntimeError("Model not loaded.")
    try:
        if hasattr(mdl, "predict_proba"):
            preds = mdl.predict(X)
            probas = mdl.predict_proba(X)[:,1]
            return np.array(preds), np.array(probas)
        else:
            preds = mdl.predict(X)
            probas = mdl.predict_proba(X)[:,1]
            return np.array(preds), np.array(probas)
    except Exception:
        # fallback: ensure expected columns exist
        X2 = X.copy()
        for c in TELCO_COLUMNS:
            if c not in X2.columns:
                X2[c] = 0
        preds = mdl.predict(X2)
        probas = mdl.predict_proba(X2)[:,1]
        return np.array(preds), np.array(probas)

# ---------------- UI polish ----------------
st.markdown(
    """
    <style>
    .title {font-size:24px; font-weight:700; margin-bottom:0.2rem;}
    .subtitle {color:#6c757d; margin-top:0; margin-bottom:0.6rem;}
    .card {background:#f7f9fc; padding:10px; border-radius:8px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}
    .small {font-size:13px; color:#6c757d;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">📡 Telecom Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Use the saved model. Single or batch predictions will be saved in a persistent store. Visuals show trends based on stored predictions.</div>', unsafe_allow_html=True)
st.markdown("---")

# Top KPIs
store_df = read_store()
total_recs = len(store_df)
churn_recs = int(store_df['predicted_churn'].sum()) if ('predicted_churn' in store_df.columns and total_recs>0) else 0
avg_prob = float(store_df['churn_probability'].mean()) if ('churn_probability' in store_df.columns and total_recs>0) else 0.0
last_time = store_df['pred_time'].max() if ('pred_time' in store_df.columns and total_recs>0) else "N/A"

k1, k2, k3, k4 = st.columns([1,1,1,1])
k1.metric("Stored rows", total_recs)
k2.metric("Predicted churners", f"{churn_recs}")
k3.metric("Avg churn probability", f"{avg_prob:.3f}")
k4.metric("Last update", last_time)

st.markdown("---")

# ---------------- Single prediction ----------------
st.header("1) Single prediction (Predict & Save)")
with st.form("single_form"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Male","Female"])
        senior = st.selectbox("Senior Citizen", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
        partner = st.selectbox("Partner", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes","No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service","No","Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    with right:
        online_security = st.selectbox("Online Security", ["No internet service","No","Yes"])
        online_backup = st.selectbox("Online Backup", ["No internet service","No","Yes"])
        device_protection = st.selectbox("Device Protection", ["No internet service","No","Yes"])
        tech_support = st.selectbox("Tech Support", ["No internet service","No","Yes"])
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, format="%.2f")
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, format="%.2f")

    submit_single = st.form_submit_button("Predict & Save")

if submit_single:
    if model is None:
        st.error("Model not loaded. Place catboost_churn.pkl or churn_pipeline.pkl in the folder.")
    else:
        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])
        try:
            preds, probs = predict_df(model, input_df)
            pred = int(preds[0]); proba = float(probs[0])
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_row = input_df.copy()
            save_row["predicted_churn"] = pred
            save_row["churn_probability"] = proba
            save_row["pred_time"] = now
            new_store = append_to_store(save_row)
            # Clear and show clear formatted output
            st.success("Prediction saved to store ✅")
            # Nicely formatted output:
            st.markdown("**Prediction result**")
            st.markdown(f"- **Churn probability:** {proba:.2%}")
            st.markdown(f"- **Predicted class:** {'🔴 CHURN' if pred==1 else '🟢 NO CHURN'}")
            if proba >= 0.75:
                st.warning("**Action:** High risk — Immediate retention recommended (call + offer).")
            elif proba >= 0.50:
                st.info("**Action:** Medium risk — Consider targeted retention campaign.")
            else:
                st.success("**Action:** Low risk — No immediate action required.")
            # update KPIs
            store_df = new_store
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")

# ---------------- Batch prediction ----------------
st.header("2) Batch prediction (upload CSV)")
uploaded = st.file_uploader("Upload CSV with feature columns (same schema)", type=["csv"])
if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        st.info(f"Loaded file: {uploaded.name} — rows: {batch_df.shape[0]}")
        st.dataframe(batch_df.head(5), use_container_width=True)
        if st.button("Run batch predict and save"):
            if model is None:
                st.error("Model not loaded.")
            else:
                try:
                    preds, probs = predict_df(model, batch_df)
                    out = batch_df.copy()
                    out["predicted_churn"] = preds
                    out["churn_probability"] = probs
                    out["pred_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_store = append_to_store(out)
                    st.success(f"Batch predictions appended: {len(out)} rows. Store size: {len(new_store)}")
                    store_df = new_store
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.markdown("---")

# ---------------- Store manager (view / delete / download) ----------------
st.header("3) Stored predictions — view / delete / download")
store_df = read_store()
if store_df.empty:
    st.info("No stored predictions yet.")
else:
    # show lightweight filters
    cols = store_df.columns.tolist()
    # date filters if available
    if 'pred_time' in store_df.columns:
        store_df['pred_time'] = pd.to_datetime(store_df['pred_time'])
        min_date = st.date_input("From date", value=store_df['pred_time'].dt.date.min())
        max_date = st.date_input("To date", value=store_df['pred_time'].dt.date.max())
        df_filtered = store_df[(store_df['pred_time'].dt.date >= min_date) & (store_df['pred_time'].dt.date <= max_date)]
    else:
        df_filtered = store_df.copy()

    # contract filter
    if 'Contract' in store_df.columns:
        contract_vals = ['All'] + sorted(store_df['Contract'].astype(str).unique().tolist())
        sel_contract = st.selectbox("Contract filter", contract_vals, index=0)
        if sel_contract != 'All':
            df_filtered = df_filtered[df_filtered['Contract'] == sel_contract]

    st.dataframe(df_filtered.reset_index(drop=False).rename(columns={"index":"store_index"}), use_container_width=True, height=300)

    # delete by store_index (which is original index)
    st.markdown("**Delete rows** — select store_index values (leftmost column) to remove permanently from store.")
    available_indexes = df_filtered.reset_index().set_index('index').index.tolist()  # original indices
    # Provide a multiselect of the original indices (as strings to avoid confusion)
    selectable = df_filtered.reset_index()[['index']].rename(columns={'index':'store_index'})
    selectable_values = selectable['store_index'].astype(str).tolist()
    to_delete = st.multiselect("Select store_index to delete", options=selectable_values)
    if st.button("Delete selected rows"):
        if not to_delete:
            st.warning("No rows selected to delete.")
        else:
            # convert back to integers and drop from original store_df
            idxs = [int(x) for x in to_delete]
            orig = read_store()
            new_orig = orig.drop(index=idxs).reset_index(drop=True)
            write_store(new_orig)
            st.success(f"Deleted {len(idxs)} rows from store.")
            store_df = new_orig

    if st.button("Clear entire store (DELETE ALL)"):
        confirm = st.checkbox("I confirm delete ALL stored predictions", key="confirm_delete_all")
        if confirm:
            if os.path.exists(STORE_PATH):
                os.remove(STORE_PATH)
            if os.path.exists(CSV_BACKUP):
                os.remove(CSV_BACKUP)
            st.success("All stored predictions removed.")
            store_df = pd.DataFrame()

    # download
    if not store_df.empty:
        csv_bytes = store_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download stored predictions (CSV)", data=csv_bytes, file_name="predictions_store.csv", mime="text/csv")

st.markdown("---")

# ---------------- Visuals (compact charts) ----------------
st.header("4) Visual trends & analysis (from stored predictions)")
store_df = read_store()
if store_df.empty:
    st.info("No stored data to show charts.")
else:
    dfp = store_df.copy()
    if 'pred_time' in dfp.columns:
        dfp['pred_time'] = pd.to_datetime(dfp['pred_time'])
    else:
        dfp['pred_time'] = pd.to_datetime(datetime.now())

    total = len(dfp)
    churners = int(dfp['predicted_churn'].sum()) if 'predicted_churn' in dfp.columns else 0
    churn_rate = churners / total if total>0 else 0

    st.subheader("Overview")
    o1, o2, o3 = st.columns(3)
    o1.metric("Total records", total)
    o2.metric("Predicted churners", churners)
    o3.metric("Churn rate", f"{churn_rate:.1%}")

    st.subheader("Compact charts")
    # churn distribution (small)
    if 'predicted_churn' in dfp.columns:
        fig_pie = px.pie(dfp, names=dfp['predicted_churn'].map({0:"No churn",1:"Churn"}), title="Predicted churn distribution", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # tenure histogram (small)
    if 'tenure' in dfp.columns:
        fig_t = px.histogram(dfp, x="tenure", color=dfp['predicted_churn'].map({0:"No churn",1:"Churn"}), title="Tenure by predicted churn", height=300)
        st.plotly_chart(fig_t, use_container_width=True)

    # monthly charges box (small)
    if 'MonthlyCharges' in dfp.columns:
        fig_box = px.box(dfp, x=dfp['predicted_churn'].map({0:"No churn",1:"Churn"}), y="MonthlyCharges", title="Monthly Charges by predicted churn", height=300)
        st.plotly_chart(fig_box, use_container_width=True)

    # churn over time (compact)
    try:
        df_time = dfp.set_index('pred_time').resample('D').agg({'predicted_churn':'sum'}).reset_index()
        fig_time = px.line(df_time, x='pred_time', y='predicted_churn', markers=True, title="Predicted churn count over time", height=300)
        st.plotly_chart(fig_time, use_container_width=True)
    except Exception:
        st.info("Not enough date data for churn-over-time chart.")

st.markdown("---")

# ---------------- Model insights ----------------
st.header("5) Model insights (feature importance if available)")
if model is None:
    st.info("No model loaded.")
else:
    try:
        estimator = model
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            estimator = model.named_steps["classifier"]
        fi = None
        fnames = None
        if hasattr(estimator, "get_feature_importance"):
            fi = estimator.get_feature_importance(prettified=False)
            try:
                fnames = list(estimator.feature_names_)
            except Exception:
                fnames = TELCO_COLUMNS[:len(fi)]
        elif hasattr(estimator, "feature_importances_"):
            fi = estimator.feature_importances_
            fnames = TELCO_COLUMNS[:len(fi)]
        if fi is not None:
            fi_series = pd.Series(fi, index=fnames).sort_values(ascending=False).head(12)
            st.bar_chart(fi_series)
        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

st.markdown("---")
st.caption("Stored data files: predictions_store.xlsx and predictions_store.csv. For production, use a database instead of Excel.")
