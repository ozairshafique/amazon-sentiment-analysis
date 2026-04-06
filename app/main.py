from fastapi import FastAPI, HTTPException
from .schemas import ReviewRequest, ReviewResponse
from src.model_loader import MODELS

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Welcome to the Amazon Sentiment Analysis API!"}



@app.post("/predict", response_model=ReviewResponse)
def prediction(model_name: str, request: ReviewRequest):
    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} not found "
        )
    model = MODELS[model_name]
    predict = model.predict([request.text])[0]
    probability = model.predict_proba([request.text])[0].max()

    return ReviewResponse(
        model=model_name,
        sentiment="positive" if predict == 1 else "negative",
        confidence=probability
    )
