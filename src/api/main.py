# src/api/main.py
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from google.cloud import storage
from google.oauth2 import service_account

# Import your model / batch functions (rename to avoid collisions)
from src.inference_pipeline.inference import predict as model_predict
from src.batch.run_monthly import run_monthly_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration via env
GCS_BUCKET = os.getenv("GCS_BUCKET", "housing-regression-data")
GCP_SA_KEY = os.getenv("GCP_SA_KEY")  # optional path to service account JSON
GCP_PROJECT = os.getenv("GCP_PROJECT")  # optional project id

# Helper: lazy GCS client creation
def get_gcs_client() -> storage.Client:
    """
    Lazily create a google.cloud.storage.Client.

    Priority:
     1. Use GCP_SA_KEY (path to JSON) if provided.
     2. Else rely on GOOGLE_APPLICATION_CREDENTIALS or ADC.
    Pass GCP_PROJECT to client if provided to avoid 'project not found' errors.
    """
    if GCP_SA_KEY:
        sa_path = Path(GCP_SA_KEY)
        if not sa_path.exists():
            raise FileNotFoundError(f"GCP_SA_KEY set to {GCP_SA_KEY} but file not found.")
        creds = service_account.Credentials.from_service_account_file(str(sa_path))
        return storage.Client(project=GCP_PROJECT, credentials=creds)

    # Let google library pick up GOOGLE_APPLICATION_CREDENTIALS or ADC.
    try:
        return storage.Client(project=GCP_PROJECT)
    except Exception as exc:
        raise EnvironmentError(
            "Could not create a GCS client. Set one of:\n"
            " - GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json\n"
            " - GCP_SA_KEY=/path/to/sa.json\n"
            " - run `gcloud auth application-default login` for ADC\n"
            "Optionally set GCP_PROJECT to your project id.\n"
            f"Original error: {exc}"
        ) from exc

# Utility: download if not cached locally
def load_from_gcs(key: str, local_path: str, bucket_name: Optional[str] = None) -> str:
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        bname = bucket_name or GCS_BUCKET
        client = get_gcs_client()
        bucket = client.bucket(bname)
        blob = bucket.blob(key)
        if not blob.exists(client):
            raise FileNotFoundError(f"Object gs://{bname}/{key} not found")
        logger.info("Downloading gs://%s/%s to %s", bname, key, str(local_path))
        blob.download_to_filename(str(local_path))
    else:
        logger.info("Using cached file: %s", str(local_path))
    return str(local_path)

# Paths (will attempt to download if missing)
try:
    MODEL_PATH = Path(load_from_gcs("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))
except Exception as exc:
    # Do not crash server import — keep MODEL_PATH as a Path to the expected file but log the issue.
    logger.warning("Could not download model at startup: %s", exc)
    MODEL_PATH = Path("models/xgb_best_model.pkl")

try:
    TRAIN_FE_PATH = Path(load_from_gcs("processed/feature_engineered_train.csv", "data/processed/feature_engineered_train.csv"))
except Exception as exc:
    logger.warning("Could not download training features at startup: %s", exc)
    TRAIN_FE_PATH = Path("data/processed/feature_engineered_train.csv")

# Load expected training features for alignment (if available)
TRAIN_FEATURE_COLUMNS = None
try:
    if TRAIN_FE_PATH.exists():
        _train_col = pd.read_csv(TRAIN_FE_PATH, nrows=1)
        TRAIN_FEATURE_COLUMNS = [c for c in _train_col.columns if c != "price"]
        logger.info("Loaded training feature columns (%d)", len(TRAIN_FEATURE_COLUMNS))
    else:
        logger.info("Training features file not present at %s", TRAIN_FE_PATH)
except Exception as exc:
    logger.warning("Failed to read training features: %s", exc)
    TRAIN_FEATURE_COLUMNS = None

# FastAPI app
app = FastAPI(title="House Regression API")

@app.get("/")
async def root():
    return {"message": "Housing Regression API is running"}

@app.get("/health")
async def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model Not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status

@app.post("/predict")
async def predict_batch(data: List[Dict[str, Any]]):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Model not found at {str(MODEL_PATH)}")

    df = pd.DataFrame(data)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data provided")

    try:
        preds_df = model_predict(df, model_path=MODEL_PATH)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    if "predicted_price" not in preds_df.columns:
        raise HTTPException(status_code=500, detail="Model output missing 'predicted_price' column")

    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()
    return resp

@app.post("/run_batch")
async def run_batch():
    try:
        preds = run_monthly_predictions()
    except Exception as exc:
        logger.exception("Batch run failed")
        raise HTTPException(status_code=500, detail=f"Batch run failed: {exc}")

    return {
        "status": "success",
        "rows_predicted": int(len(preds)) if preds is not None else 0,
        "output_dir": "Data/predictions/"
    }

@app.get("/latest_predictions")
async def latest_predictions(limit: int = 5):
    pred_dir = Path("Data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        raise HTTPException(status_code=404, detail="No predictions found")

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }
