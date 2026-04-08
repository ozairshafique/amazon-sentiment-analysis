"""
model_loader.py — Lazy model loading and inference for production.

Only LogisticRegression is loaded here because:
  - It has predict_proba() → needed for confidence scores in API/Streamlit
  - LinearSVC does NOT have predict_proba() → notebooks only

Usage:
    from src.model_loader import predict

    result = predict("This product is amazing!")
    # → {"model": "logreg", "sentiment": "positive", "confidence": 0.97}
"""

import joblib
from functools import lru_cache
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

MODEL_DIR = BASE_DIR / "model"

# Models available in production
# LinearSVC is intentionally excluded — no predict_proba support
AVAILABLE_MODELS = {"logreg"}


@lru_cache(maxsize=None)
def _load(name: str):
    """Load and cache a model from disk. Called once per model per session."""
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Model '{name}' is not available for production. "
            f"Available: {AVAILABLE_MODELS}. "
            f"LinearSVC is for notebooks only."
        )

    path = MODEL_DIR / f"{name}_model.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run `python train.py` first to generate models."
        )

    return joblib.load(path)


def predict(text: str, model_name: str = "logreg") -> dict:
    """
    Run sentiment inference on a single text string.

    Args:
        text:        Raw review text (preprocessing handled inside pipeline).
        model_name:  Which model to use. Default: 'logreg'.

    Returns:
        dict with keys: model, sentiment, confidence
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    model  = _load(model_name)
    proba  = model.predict_proba([text.strip()])[0]
    label  = int(proba.argmax())

    return {
        "model":      model_name,
        "sentiment":  "positive" if label == 1 else "negative",
        "confidence": round(float(proba.max()), 4),
    }