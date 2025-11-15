# app.py
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from google.cloud import storage

# MUST be the first Streamlit command (call once)
st.set_page_config(page_title="Housing Price Prediction — Holdout Explorer", layout="wide")

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
GCS_BUCKET = os.getenv("GCS_BUCKET", "housing-regression-data")
# Optional: path to service account key (when running locally). In Cloud Run, use Workload Identity or set
# GOOGLE_APPLICATION_CREDENTIALS as secret/volume if you prefer a key file.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)

# Robust lazy storage client (does NOT create a client at import time)
def get_storage_client():
    """
    Create and return a google.cloud.storage.Client in a robust way:
    - If GOOGLE_APPLICATION_CREDENTIALS points to a key file, use it.
    - If GCP_PROJECT or GOOGLE_CLOUD_PROJECT env vars exist, use them.
    - If project not available but key file contains project_id, use it.
    - Otherwise attempt default Client() which will use ADC/metadata server.
    """
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_env = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")

    if key_path:
        key_path = str(key_path)
        # try to read project_id from key file (if present)
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                js = json.load(f)
                json_project = js.get("project_id")
        except Exception:
            json_project = None

        final_project = project_env or json_project

        # Create client explicitly from service account file (prefer explicit project)
        if final_project:
            return storage.Client.from_service_account_json(key_path, project=final_project)
        return storage.Client.from_service_account_json(key_path)

    # No key file: prefer explicit project if provided
    if project_env:
        return storage.Client(project=project_env)

    # Last resort: let google client determine project via ADC or metadata server
    return storage.Client()

def load_from_gcs(blob_name: str, local_path: str) -> str:
    """
    Download an object from Google Cloud Storage into a local path if not cached locally.
    Creates the storage client lazily to avoid project-detection errors at import time.
    """
    local_path = Path(local_path)
    if local_path.exists():
        return str(local_path)

    # create client here (lazy)
    client = get_storage_client()
    bucket_name = os.getenv("GCS_BUCKET", GCS_BUCKET)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"📥 Downloading {blob_name} from GCS bucket {bucket_name}..."):
        try:
            blob.download_to_filename(str(local_path))
            st.success(f"Downloaded {blob_name}")
        except Exception as e:
            st.error(f"Failed to download {blob_name}: {e}")
            raise
    return str(local_path)

# Paths (fetch from GCS if missing)
HOLDOUT_ENGINEERED_PATH = load_from_gcs(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)
HOLDOUT_META_PATH = load_from_gcs(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv"
)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# ============================
# UI
# ============================
st.title("🏠 Housing Price Prediction — Holdout Explorer")

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")

        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = fe_df.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)

            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
                idx_region = yearly_data.index
                payload_region = fe_df.loc[idx_region].to_dict(orient="records")

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_region, index=yearly_data.index).astype(float)

            # Aggregate by month
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            # Highlight selected month
            monthly_avg["highlight"] = monthly_avg["month"].apply(lambda m: "Selected" if m == month else "Other")

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}"
            )

            # Add highlight with background shading
            highlight_month = month
            fig.add_vrect(
                x0=highlight_month - 0.5,
                x1=highlight_month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")
