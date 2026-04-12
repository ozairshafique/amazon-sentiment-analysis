from src.train import build_pipelines
import pytest

def test_build_piplines():
    piplines = build_pipelines()
    assert isinstance(piplines, dict)
    assert "logreg" in piplines
    assert hasattr(piplines["logreg"], "predict")
    assert hasattr(piplines["logreg"], "predict_proba")
