
''' FastAPI application for Amazon Sentiment Analysis API.

Usage:
     uvicorn api.main:app --reload       # development
     uvicorn api.main:app --host 0.0.0.0  # production

Endpoints:
     GET  /           → Health check
     POST /predict    → Predict sentiment for a given review text and model.

Intractive API docs available at:
     http://localhost:8000/docs  (Swagger UI)
     http://localhost:8000/redoc  (ReDoc)

'''

from fastapi import FastAPI, HTTPException ,Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.model_loader import AVAILABLE_MODELS, _load, predict
import logging
import time
from .schemas import (
    ReviewRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ErrorResponse
    )

# Logging configuration

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# API
API_VERSION = "1.0.1"

# Lifespan: Load models at startup, log startup time, and handle graceful shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up Loading models...")
    try:
        for name in AVAILABLE_MODELS:
            _load(name)
            log.info(f"Model '{name}' loaded successfully.")
    except FileNotFoundError as e:
        log.error(f"Model file not found {name}")
        raise RuntimeError(f"Startup failed: {e} one or more model files are missing.") from e

    log.info(f"All models loaded. Startup complete.")
    yield
    log.info("Shutting down")

# CORS configuration (allow all origins for simplicity)
app = FastAPI(
    title="Amazon Sentiment Analysis API",
    version=API_VERSION,
    description="Predict sentiment of Amazon reviews using pre-trained models.",
    lifespan=lifespan
)

# Allow CORS from any origin (for testing; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def log_request(request: Request, call_next):
    ''' Log each incoming request with method, path, and processing time '''

    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    log.info(f"{request.method} {request.url.path} completed in {duration:.2f} with status {response.status_code}")
    return response

# Endpoints

@app.get("/", summary="Root", tags=['General'])
def root():
    ''' API entry point – return basic information and available endpoints. '''

    return {
        "message": "Welcome to the Amazon Sentiment Analysis API!",
        "name": "Amazon Sentiment Analysis API",
        "version": API_VERSION,
        "models": list(AVAILABLE_MODELS),
        "endpoints": {
            "single": "POST /predict",
            "docs": "GET /docs",
            "health": "GET /health",
            "batch": " POST /batch/predict"
        }
    }


@app.get("/health", summary="Health Check", tags=['General'])
def health():
    ''' Simple health check endpoint to verify API is running.
        Returns 200 OK if the API is healthy. '''

    return{
        "status": "ok",
        "message": "API is healthy and running.",
        "version": API_VERSION,
        "models": list(AVAILABLE_MODELS)
    }


@app.post("/predict", response_model=PredictionResponse, summary="Single Prediction", tags=['Prediction'],status_code=status.HTTP_200_OK)
def predict_single(request: ReviewRequest):
    ''' Predict sentiment for a single review text using the specified model.

    - **text**: Raw review text to analyze. Validations will be done in preprocessing step. - **model_name**: Model to use for prediction.

      Currently only 'logreg' is available.

      Returns sentiment (`positive` / `negative`) and a confidence score. '''

    try:

        result = predict(request.text, request.model_name)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=(str(e)))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=(str(e)))
    except Exception as e:
        log.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

    return PredictionResponse(
        text= request.text,
        model= result['model'],
        sentiment= result['sentiment'],
        confidence= result['confidence'],
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Batch Prediction", tags=['Prediction'])

def predict_batch(reviews: list[ReviewRequest]):
    ''' Predict sentiment for multiple review texts using the specified model.

    - **texts**: List of raw review texts to analyze.
    - **model_name**: Model to use for prediction.

      Currently only 'logreg' is available.

      Returns a list of sentiment predictions and confidence scores. '''

    if len(reviews) > 100:
        raise HTTPException(status_code=400,
                            detail="Request body must contain at most 100 reviews.")

    if len(reviews) == 0:
        raise HTTPException(status_code=400,
                            detail="Request body must contain at least 1 review.")

    results = []
    for review in reviews:
        try:
            result = predict(review.text, review.model_name)
            results.append(PredictionResponse(
                text = review.text,
                model = result['model'],
                sentiment = result['sentiment'],
                confidence = result['confidence']
            ))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=(str(e)))
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=(str(e)))
        except Exception as e:
            log.warning(f"Batch item failed {e}")

            results.append(PredictionResponse(
                text=review.text,
                model=review.model_name,
                sentiment="error",
                confidence=0.0
            ))

    return BatchPredictionResponse(
        total = len(results),
        results = results
        )
