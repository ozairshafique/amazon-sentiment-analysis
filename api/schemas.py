from pydantic import BaseModel, Field, field_validator

class ReviewRequest(BaseModel):
    text: str = Field(...,min_length=3,
                       max_length=5000,
                       description="Raw review text to analyze. Validations will be done in preprocessing step.",
                       example="This product is amazing! I loved it.")

    model_name: str = Field(default="logreg",
                       description="Model to use for prediction. Currently only 'logreg' is available.",
                       example="logreg")

    @field_validator("text")
    @classmethod
    def validate_text(cls, t: str) -> str:
        if not t.strip():
            raise ValueError("Text must be a non-empty string.")

        return t.strip()

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, t: str) -> str:
        allowed = {"logreg"}
        if t not in allowed:
            raise ValueError(f"Unsupported model '{t}'. Available: {allowed}")
        return t

    model_config = {
        "json_schema_extra":{
            "examples":[
                {
                    "text": "This product is absolutely amazing — best purchase I've made.",
                    "model_name": "logreg"
                }
            ]
        }
    }

# Response schema for predictions
class PredictionResponse(BaseModel):
    ''' Defines the structure of the API response for a sentiment prediction request.'''

    text: str = Field(description="Original review text that was analyzed.")
    model: str = Field(description="Name of the model used for prediction.")
    sentiment: str = Field(description="Predicted sentiment label: 'positive' or 'negative'.")
    confidence: float = Field(description="Confidence score for the prediction.")

class BatchPredictionResponse(BaseModel):
    ''' Defines the structure of the API response for batch sentiment prediction requests.'''

    total: int = Field(description="Total number of reviews processed.")
    results: list[PredictionResponse] = Field(description="List of predictions for each input review.")

class ErrorResponse(BaseModel):
    ''' Defines the structure of the API response for error cases.'''

    detail: str = Field(description="Description of the error.")