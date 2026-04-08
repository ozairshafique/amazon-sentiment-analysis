"""
preprocessing.py — Text cleaning transformer for sklearn pipelines.

Used in:
  - train.py              (training)
  - src/model_loader.py   (production inference)
  - notebooks             (exploration)
"""

import re
import nltk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data once at import time
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible text cleaner.

    Steps:
        1. Lowercase
        2. Remove digits
        3. Remove punctuation / special characters
        4. Remove English stopwords

    Works identically in training, inference, and notebooks.
    """

    def fit(self, X, y=None):
        # Stateless — nothing to learn. Required by sklearn API.
        return self

    def _clean(self, text) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"\d+", "", text)        # remove digits
        text = re.sub(r"[^\w\s]", "", text)    # remove punctuation
        text = " ".join(
            w for w in text.split() if w not in STOPWORDS
        )
        return text

    def transform(self, X, y=None):
        return [self._clean(t) for t in X]