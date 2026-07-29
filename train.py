"""Reproducible training pipeline for the fraud detection model.

Loads data/paysim.csv, replicates the cleaning + feature engineering from
notebook/fraud_detection.ipynb, trains a Random Forest and an XGBoost model,
compares them on a held-out test set, and saves the better one to models/
along with an evaluation report in reports/.

Usage:
    python train.py
"""

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from fraud_detection.features import MODEL_FEATURE_ORDER, TX_TYPE_MAP

DATA_PATH = "data/paysim.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
SUBSAMPLE_SIZE = 300_000
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.30


def load_and_engineer(path):
    df = pd.read_csv(path)
    df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], inplace=True)
    df.drop_duplicates(inplace=True)

    df["type"] = df["type"].map(TX_TYPE_MAP)

    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["sender_drained"] = (df["newbalanceOrig"] < 0.1 * df["oldbalanceOrg"]).astype(int)
    df["dest_was_zero"] = (df["oldbalanceDest"] == 0).astype(int)
    df["balance_error_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    df["balance_error_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    return df


def evaluate(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    print(f"\n{name}: {json.dumps(metrics, indent=2)}")
    return metrics, y_prob, y_pred


def main():
    import os

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    t0 = time.time()
    df = load_and_engineer(DATA_PATH)
    print(f"Loaded + engineered {len(df):,} rows in {time.time() - t0:.1f}s")

    X = df[MODEL_FEATURE_ORDER]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    X_train_small = X_train.sample(
        min(SUBSAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE
    )
    y_train_small = y_train.loc[X_train_small.index]

    rf = RandomForestClassifier(
        criterion="gini",
        max_depth=6,
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_small, y_train_small)
    rf_metrics, rf_prob, rf_pred = evaluate("Random Forest", rf, X_test, y_test)

    scale_pos_weight = (y_train_small == 0).sum() / (y_train_small == 1).sum()
    xgb = XGBClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(X_train_small, y_train_small)
    xgb_metrics, xgb_prob, xgb_pred = evaluate("XGBoost", xgb, X_test, y_test)

    candidates = {"random_forest": (rf, rf_metrics, rf_prob, rf_pred),
                  "xgboost": (xgb, xgb_metrics, xgb_prob, xgb_pred)}
    best_name = max(candidates, key=lambda k: candidates[k][1]["f1"])
    best_model, best_metrics, best_prob, best_pred = candidates[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, f"{MODELS_DIR}/fraud_model.pkl")
    joblib.dump(MODEL_FEATURE_ORDER, f"{MODELS_DIR}/model_columns.pkl")

    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "decision_threshold": DECISION_THRESHOLD,
                "random_forest": rf_metrics,
                "xgboost": xgb_metrics,
            },
            f,
            indent=2,
        )

    with open(f"{REPORTS_DIR}/model_comparison.md", "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(f"Decision threshold: `{DECISION_THRESHOLD}` | Test set size: {len(X_test):,}\n\n")
        f.write("| Model | Precision | Recall | F1 | ROC-AUC |\n")
        f.write("|---|---|---|---|---|\n")
        for name, (_, m, _, _) in candidates.items():
            f.write(
                f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['roc_auc']:.4f} |\n"
            )
        f.write(f"\n**Selected model: `{best_name}`** (highest F1 on the fraud class)\n")

    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fraud"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, best_prob, ax=ax, name=best_name)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/roc_curve.png", dpi=120)
    plt.close(fig)

    print(f"\nSaved model artifacts to {MODELS_DIR}/ and report to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
