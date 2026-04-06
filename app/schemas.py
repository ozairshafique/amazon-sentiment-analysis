from pydantic import BaseModel, Field
from typing import Optional

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, example="This product is amazing! I loved it.")

class ReviewResponse(BaseModel):
    model: str = Field(default="svc")
    sentiment: str
    confidence: float