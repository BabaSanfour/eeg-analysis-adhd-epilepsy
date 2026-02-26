#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, balanced_accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Eyes Open vs Eyes Closed classifier.")
    parser.add_argument("--csv", default="embeddings_motor.csv", help="Path to embeddings csv")
    args = parser.parse_args()

    LOG.info(f"Loading data from {args.csv}")
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        LOG.error(f"File {args.csv} not found. Please run make_embeddings_motor.py first.")
        return

    # Check required columns
    required_cols = ["subject", "run", "segment"]
    if not all(col in df.columns for col in required_cols):
        LOG.error(f"Missing one of {required_cols} in CSV.")
        return

    # Filter for R01 and R02 just in case
    # Runs are usually ints or strings like "01", so convert target to consistency
    df["run"] = df["run"].astype(str).str.zfill(2)
    
    valid_runs = ["01", "02"]
    df = df[df["run"].isin(valid_runs)].copy()
    
    if df.empty:
        LOG.error("No data found for runs 01 or 02.")
        return
    
    LOG.info(f"Found {len(df)} samples from {df['subject'].nunique()} subjects.")

    # Create Classification Target
    # 0 = Eyes Open (R01), 1 = Eyes Closed (R02)
    y = (df["run"] == "02").astype(int)
    groups = df["subject"]

    # Features: all cbramod columns
    feat_cols = [c for c in df.columns if c.startswith("cbramod_")]
    X = df[feat_cols]
    
    LOG.info(f"Training on {len(X)} samples with {len(feat_cols)} features.")
    LOG.info(f"Class balance: {y.value_counts().to_dict()} (0=Open, 1=Closed)")

    # Classifier
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    # Cross-Validation
    # GroupKFold to ensure segments from same subject don't leak into validation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Handle small datasets (e.g. only 2 subjects) where 5-fold is impossible
    n_subs = df["subject"].nunique()
    if n_subs < 5:
        n_splits = n_subs
        LOG.warning(f"Only {n_subs} subjects available. Reducing CV to {n_splits}-fold.")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    LOG.info(f"Running {cv.get_n_splits(X, y, groups)}-fold Cross-Validation...")
    
    scores = cross_val_score(
        clf, X, y, groups=groups, cv=cv, scoring="balanced_accuracy", n_jobs=-1
    )

    LOG.info("-" * 40)
    LOG.info(f"Mean Balanced Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    LOG.info("-" * 40)
    LOG.info(f"Individual Fold Scores: {scores}")

if __name__ == "__main__":
    main()
