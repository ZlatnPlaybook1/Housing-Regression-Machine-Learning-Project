## House Price Prediction Oroject

## Project Overview

Housing Regression MLE is a complete machine learning pipeline designed to predict housing prices using XGBoost. It adheres to ML engineering best practices through modular pipeline design, experiment tracking with MLflow, containerization, deployment on Google Cloud, and thorough testing. The project features both a REST API and a Streamlit dashboard, enabling seamless and interactive price predictions.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

### Core Modules

- **`src/feature_pipeline`**: Data loading, Preprocessing and Feature Engineering
    - `load.py`: Time-aware data splitting (train <2020, eval 2020-21, holdout ≥2022)
    -  `preprocess.py`: City normalization, deduplication, outlier removal  
    - `feature_engineering.py`: Date features, frequency encoding (zipcode), target encoding (city_full)

- **`src/training_pipeline/`**: Model training and hyperparameter optimization
  - `train.py`: Baseline XGBoost training with configurable parameters
  - `tune.py`: Optuna-based hyperparameter tuning with MLflow integration
  - `eval.py`: Model evaluation and metrics calculation

- **`src/inference_pipeline/`**: Production inference
  - `inference.py`: Applies same preprocessing/encoding transformations using saved encoders

- **`src/batch/`**: Batch prediction processing
  - `run_monthly.py`: Generates monthly predictions on holdout data

- **`src/api/`**: FastAPI web service
  - `main.py`: REST API with S3 integration, health checks, prediction endpoints, and batch processing

### Web Applications

- **`app.py`**: Streamlit dashboard for interactive housing price predictions
  - Real-time predictions via FastAPI integration
  - Interactive filtering by year, month, and region
  - Visualization of predictions vs actuals with metrics (MAE, RMSE, % Error)
  - Yearly trend analysis with highlighted selected periods

### Cloud Infrastructure & Deployment

The project is deployed on Google Cloud Platform (GCP) with the following architecture:

#### **Container Registry**
- Docker images stored in Google Container Registry (GCR)
- Automated builds triggered via Cloud Build or CI/CD pipeline
```bash
# Tag and push API image to GCR
docker tag housing-regression gcr.io/[PROJECT_ID]/housing-regression:latest
docker push gcr.io/[PROJECT_ID]/housing-regression:latest

# Tag and push Streamlit image to GCR
docker tag housing-streamlit gcr.io/[PROJECT_ID]/housing-streamlit:latest
docker push gcr.io/[PROJECT_ID]/housing-streamlit:latest
```

#### **Cloud Run Services**
- **API Service**: FastAPI deployed on Cloud Run for serverless inference
- **Dashboard Service**: Streamlit app deployed on Cloud Run for interactive predictions
```bash
# Deploy API to Cloud Run
gcloud run deploy housing-api \
  --image gcr.io/[PROJECT_ID]/housing-regression:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2

# Deploy Streamlit dashboard to Cloud Run
gcloud run deploy housing-dashboard \
  --image gcr.io/[PROJECT_ID]/housing-streamlit:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars API_URL=https://housing-api-[hash].run.app
```


### Data Leakage Prevention

The project implements strict data leakage prevention:
- Time-based splits (not random)
- Encoders fitted only on training data
- Leakage-prone columns dropped before training
- Schema alignment enforced between train/eval/inference

## Common Commands

### Environment Setup

```bash
# Install dependencies using uv
uv sync
```

### Testing
```bash
# Run all tests
pytest

# Run specific test modules  
pytest tests/test_features.py
pytest tests/test_training.py
pytest tests/test_inference.py

# Run with verbose output
pytest -v


### Data Pipeline
```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

### Training Pipeline
```bash
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py

# Model evaluation
python src/training_pipeline/eval.py
```

### Inference
```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv

# Batch monthly predictions
python src/batch/run_monthly.py

### API Service
```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit Dashboard
```bash
# Start Streamlit dashboard locally
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker
```bash
# Build API container
docker build -t housing-regression .

# Build Streamlit container  
docker build -t housing-streamlit -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 housing-regression

# Run Streamlit container
docker run -p 8501:8501 housing-streamlit
```

### MLflow Tracking
```bash
# Start MLflow UI (view experiments)
mlflow ui
```

## Key Design Patterns

### Pipeline Modularity
Each pipeline component can be run independently with consistent interfaces. All modules accept configurable input/output paths for testing isolation.

### Cloud-Native Architecture
- **GCS-First Storage**: Models and data automatically sync from Google Cloud Storage buckets
- **Containerized Services**: Both API and dashboard run in Docker containers deployed on Cloud Run
- **Auto-scaling Infrastructure**: Cloud Run provides serverless container scaling with automatic scale-to-zero
- **Environment-based Configuration**: Separate configs for local development and production using Cloud Run environment variables

### Encoder Persistence  
Frequency and target encoders are saved as pickle files during training and loaded during inference to ensure consistent transformations. Encoders are stored in GCS for persistence across container instances.

### Configuration Management
Model parameters, file paths, and pipeline settings use sensible defaults but can be overridden through function parameters or environment variables. Production deployments use GCP environment variables and Secret Manager for sensitive configurations.

### Testing Strategy
- Unit tests for individual pipeline components
- Integration tests for end-to-end pipeline flows  
- Smoke tests for inference pipeline
- All tests use temporary directories to avoid touching production data
- CI/CD pipeline via Cloud Build runs full test suite before deployment

### Cloud Storage Integration
- **Model Artifacts**: XGBoost models and preprocessing objects stored in `gs://[PROJECT_ID]-housing-models/`
- **Data Buckets**: Raw and processed data in `gs://[PROJECT_ID]-housing-data/`
- **Automatic Sync**: Services pull latest models from GCS on startup
- **Versioning**: Model versions tracked via MLflow with GCS backend storage


## Dependencies
Key production dependencies (see `pyproject.toml`):
- **ML/Data**: `xgboost==3.1.1`, `scikit-learn`, `pandas==2.3.3`, `numpy==2.3.4`
- **API**: `fastapi`, `uvicorn`
- **Dashboard**: `streamlit`, `plotly`
- **Cloud**: `google-cloud-storage`, `google-cloud-secret-manager` (GCP integration)
- **Experimentation**: `mlflow`, `optuna`
- **Quality**: `great-expectations`, `evidently`

## File Structure Notes
- **`Data/`**: Raw, processed, and prediction data (time-structured, GCS-synced)
- **`models/`**: Trained models and encoders (pkl files, GCS-synced)
- **`mlruns/`**: MLflow experiment tracking data (backed by GCS)
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Comprehensive test suite with sample data
- **`Dockerfile`**: API service containerization
- **`Dockerfile.streamlit`**: Dashboard service containerization
- **CI/CD**: `.github/workflows/ci.yml` or Cloud Build triggers for automated deployment to Cloud Run