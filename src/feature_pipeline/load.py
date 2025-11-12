"""
Load & time-split the raw dataset

- Production default to writes to Data/raw/
- Tests can pass a temp `output_dir` so nothing in Data/ is touched

"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("Data/raw")

def load_and_split_data(
  raw_path: str = "Data/HouseTS.csv"  ,
  output_dir : Path | str = DATA_DIR  ):
    """Load raw dataset, split into train/eval/holdout by date, ans save to output_dir"""
    df = pd.read_csv(raw_path)

    # Ensure datetime + sort 
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Cutoffs
    cutoff_data_eval = pd.Timestamp("2020-01-01")
    cutoff_data_holdout = pd.Timestamp("2022-01-01")

    # Splits
    train_df = df[df["date"] < cutoff_data_eval]
    eval_df = df[(df["date"] >= cutoff_data_eval) & (df["date"] < cutoff_data_holdout) ]
    holdout_df = df[df["date"] >= cutoff_data_holdout]

    # Save 
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv" , index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv" , index=False)


    print(f"Data split completed (saved to {outdir}).")
    print(f"Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")


    return train_df, eval_df, holdout_df

if __name__== "__main__":
    load_and_split_data()