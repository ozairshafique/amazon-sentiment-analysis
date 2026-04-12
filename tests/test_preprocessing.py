from src.preprocessing import TextPreprocessor
import pytest

def test_preprocessor():
    preprocessor = TextPreprocessor()
    result = preprocessor.transform(["This is a TEST review! 1246"])
    assert isinstance(result, list)
    assert isinstance(result[0], str)

    assert result[0] ==result[0].lower()

    assert "1246" not in result[0]
    assert "!" not in result[0]
    assert "this" not in result[0]

def test_preprocessor_empty():
    preprocessor = TextPreprocessor()
    result = preprocessor.transform([""])
    assert result[0] == ""

def test_preprocessor_none():
    preprocessor = TextPreprocessor()
    result = preprocessor.transform([None])
    assert result[0] == ""

def test_preprocessor_multiple():
    preprocessor = TextPreprocessor()
    texts = ["Amazing product!", "Terrible quality!"]
    result = preprocessor.transform(texts)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)

def test_preprocessor_self():
    preprocessor = TextPreprocessor()
    result = preprocessor.fit(["some text"])
    assert result is preprocessor