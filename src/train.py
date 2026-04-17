"""
train.py — Train, evaluate, and save sentiment classification models.

Usage:
    python train.py

Outputs:
    model/logreg_model.pkl
    model/linear_svc_model.pkl

Models trained:
    - LogisticRegression  → used in PRODUCTION (has predict_proba)
    - LinearSVC           → used in NOTEBOOKS only (no predict_proba)
"""

import sys
import os
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, os.path.abspath("."))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import mlflow
import mlflow.sklearn

from src.preprocessing import TextPreprocessor

# Config
DATA_PATH   = Path("data/amazon_reviews.csv")
MODEL_DIR   = Path("model")
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# TF-IDF settings shared across all pipelines
TFIDF_PARAMS = dict(
    max_features=20000,
    ngram_range=(1, 3),   # unigrams, bigrams, trigrams
    min_df=2,             # ignore very rare terms
    max_df=0.9,           # ignore near-universal terms
    sublinear_tf=True,    # log-scale TF
)

# Hyperparameter grids per model
PARAM_GRIDS = {
    "logreg":     {"clf__C": [0.1, 1, 5, 10]},
    "linear_svc": {"clf__C": [0.1, 1, 5, 10]},
}


# Data loading
def load_data(path: Path) -> tuple:
    """Load CSV, encode sentiment, drop neutral (3★) and missing rows."""
    df = pd.read_csv(path)

    required = {"overall", "reviewText"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")

    # 1 = positive (4–5★), 0 = negative (1–2★), None = neutral (3★) → dropped
    df["sentiment"] = df["overall"].apply(
        lambda x: 1 if x >= 4 else (0 if x <= 2 else None)
    )

    before = len(df)
    df = df.dropna(subset=["sentiment", "reviewText"])
    dropped = before - len(df)

    df["sentiment"] = df["sentiment"].astype(int)

    print(f"\n── Data loaded ──────────────────────────────")
    print(f"  Total usable rows : {len(df):,}  ({dropped:,} neutral/missing dropped)")
    print(f"  Class distribution:")
    print(df["sentiment"].value_counts().rename({1: "Positive", 0: "Negative"}).to_string())
    print()

    return df["reviewText"].astype(str), df["sentiment"]


# Pipeline definitions
def build_pipelines() -> dict[str, Pipeline]:
    """
    Returns two pipelines:

    logreg      → Production model. Has predict_proba(), used in API & Streamlit.
    linear_svc  → Notebook/research model. No predict_proba(), NOT used in production.
    """
    return {
        "logreg": Pipeline([
            ("preprocess", TextPreprocessor()),
            ("tfidf",      TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf",        LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
            )),
        ]),

        "linear_svc": Pipeline([
            ("preprocess", TextPreprocessor()),
            ("tfidf",      TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf",        LinearSVC(
                class_weight="balanced",
                max_iter=1000,
            )),
        ]),
    }


# Training loop
def train() -> None:
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,          # preserve class ratio in both splits
        random_state=RANDOM_SEED,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    pipelines = build_pipelines()

    mlflow.set_experiment("amazon sentiment analysis")
    for name, pipeline in pipelines.items():

        if mlflow.active_run():
            mlflow.end_run() # End previous run to avoid nested runs in next iteration

        # Start a new MLflow run for this model
        with mlflow.start_run(run_name=name):

            mlflow.log_params({
                "model": name,
                "test_size": TEST_SIZE,
                "random_seed": RANDOM_SEED,
                "tfidf_max_features": TFIDF_PARAMS["max_features"],
                "tfidf_ngram_range": str(TFIDF_PARAMS["ngram_range"]),
                "tfidf_min_df": TFIDF_PARAMS["min_df"],
                "tfidf_max_df": TFIDF_PARAMS["max_df"],
                "tfidf_sublinear_tf": TFIDF_PARAMS["sublinear_tf"],
                "cv_folds": 3,
                "cv_scoring": "f1_macro"
                })

            print(f"── Training {name} {'─' * (30 - len(name))}")

            # Grid search with 3-fold CV on training split
            grid = GridSearchCV(
                pipeline,
                param_grid=PARAM_GRIDS[name],
                cv=3,
                scoring="f1_macro",   # best metric for imbalanced data
                n_jobs=-1,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            best = grid.best_estimator_

        mlflow.log_param("best_C",grid.best_params_["clf__C"])
        mlflow.log_metric("best_cv_f1", round(grid.best_score_,4), step=0)

        print(f"  Best params : {grid.best_params_}")
        print(f"  Best CV F1  : {grid.best_score_:.4f}")

        # Evaluate on held-out test set
        preds = best.predict(X_test)
        print(classification_report(
            y_test, preds,
            target_names=["Negative", "Positive"],
        ))

        # Log test metrics to MLflow
        report = classification_report(
            y_test, preds,
            target_names=["Negative", "Positive"],
            output_dict=True
        )
        mlflow.log_metric("test_f1_macro", round(report["macro avg"]["f1-score"], 4), step=0)
        mlflow.log_metric("test_f1_positive", round(report["Positive"]["f1-score"], 4), step=0)
        mlflow.log_metric("test_f1_negative", round(report["Negative"]["f1-score"], 4), step=0)
        # Retrain best pipeline on FULL dataset before saving
        # (more data = better generalisation for production)
        print(f"  Retraining {name} on full dataset...")
        best.fit(X, y)

#        mlflow.sklearn(log_model=best, artifact_path=f"{name}_model")

        out_path = MODEL_DIR / f"{name}_model.pkl"
        joblib.dump(best, out_path)
        print(f"  ✅ Saved → {out_path}\n")

    print("🎉 All models trained and saved.")


# ── Entry point ──
if __name__ == "__main__":
    train()