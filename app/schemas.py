from typing import List
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    sample_rate: int = Field(22050, ge=8000, le=48000)
    duration: float = Field(4.0, gt=0.0, le=30.0)
    n_mels: int = Field(64, ge=16, le=256)
    top_k: int = Field(3, ge=1, le=20)

class SpectrogramRequest(BaseModel):
    sample_rate: int = Field(22050, ge=8000, le=48000)
    duration: float = Field(4.0, gt=0.0, le=30.0)
    n_mels: int = Field(64, ge=16, le=256)

class StreamConfig(BaseModel):
    sample_rate: int = Field(22050, ge=8000, le=48000)
    duration: float = Field(4.0, gt=0.0, le=30.0)
    n_mels: int = Field(64, ge=16, le=256)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")

class LabelsResponse(BaseModel):
    labels: List[str] = Field(..., description="Ordered list of class labels")

class Prediction(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Probability for the label"
    )

class PredictResponse(BaseModel):
    top_prediction: Prediction
    top_k: List[Prediction]

class SpectrogramResponse(BaseModel):
    shape: List[int]
    spectrogram: List[List[float]]
