from google.cloud import storage
from pathlib import Path
import sys

# ---- Config ----
bucket_name = "housing-regression-data"

# Use the file location to compute project root reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
local_data_dir = PROJECT_ROOT / "Data" / "processed"
local_model_dir = PROJECT_ROOT / "models"

def get_client():
    try:
        # Increase timeout to 600 seconds (10 minutes)
        client = storage.Client()
        print("✅ storage.Client() created. Active project:", client.project)
        return client
    except Exception as e:
        print("❌ Failed to create storage client:", e)
        sys.exit(1)

def upload_file(bucket, local_path: Path, gcs_blob_path: str):
    if not local_path.exists():
        print(f"❌ File not found: {local_path}")
        return False
    
    try:
        # Get file size for progress tracking
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"⬆️ Uploading {local_path.name} ({file_size_mb:.2f} MB) → gs://{bucket.name}/{gcs_blob_path}")
        
        blob = bucket.blob(gcs_blob_path)
        
        # Upload with increased timeout (10 minutes)
        blob.upload_from_filename(str(local_path), timeout=600)
        
        print(f" ✅ uploaded successfully")
        return True
    except Exception as e:
        print(f"❌ Upload failed for {local_path.name}: {e}")
        print(f"   Try uploading this file manually or check your internet connection")
        return False

def main():
    client = get_client()
    bucket = client.bucket(bucket_name)
    
    # Check bucket exists
    try:
        if not bucket.exists():
            print(f"❌ Bucket '{bucket_name}' does not exist or you don't have permission.")
            sys.exit(1)
        else:
            print(f"✅ Bucket '{bucket_name}' found.\n")
    except Exception as e:
        print("❌ Error checking bucket existence:", e)
        sys.exit(1)
    
    # List of files to upload
    files_to_upload = [
        (local_data_dir / "feature_engineered_holdout.csv", "processed/feature_engineered_holdout.csv"),
        (local_data_dir / "cleaning_holdout.csv", "processed/cleaning_holdout.csv"),
        (local_data_dir / "feature_engineered_train.csv", "processed/feature_engineered_train.csv"),
        (local_model_dir / "xgb_best_model.pkl", "models/xgb_best_model.pkl"),
    ]
    
    # Track results
    success_count = 0
    fail_count = 0
    
    for local_path, gcs_path in files_to_upload:
        if upload_file(bucket, local_path, gcs_path):
            success_count += 1
        else:
            fail_count += 1
        print()  # Empty line between uploads
    
    # Summary
    print("="*50)
    print(f"✅ Successfully uploaded: {success_count} files")
    print(f"❌ Failed uploads: {fail_count} files")
    print("="*50)

if __name__ == "__main__":
    main()