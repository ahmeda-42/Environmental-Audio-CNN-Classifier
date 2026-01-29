from typing import List
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    sample_rate: int = Field(22050, ge=8000, le=48000)
    duration: float = Field(4.0, gt=0.0, le=30.0)
    n_mels: int = Field(128, ge=16, le=256)
    top_k: int = Field(3, ge=1, le=20)

class SpectrogramRequest(BaseModel):
    sample_rate: int = Field(22050, ge=8000, le=48000)
    duration: float = Field(4.0, gt=0.0, le=30.0)
    n_mels: int = Field(128, ge=16, le=256)

class StreamConfig(BaseModel):
    sample_rate: int = Field(22050, ge=8000, le=48000)
    duration: float = Field(4.0, gt=0.0, le=30.0)
    n_mels: int = Field(128, ge=16, le=256)

class HealthResponse(BaseModel):
    status: str 

class LabelsResponse(BaseModel):
    labels: List[str]

class Prediction(BaseModel):
    label: str 
    confidence: float 

class PredictResponse(BaseModel):
    top_prediction: Prediction
    top_k: List[Prediction]
    spectrogram: "SpectrogramResponse"

class SpectrogramResponse(BaseModel):
    image: str
    features: List[List[float]]
    shape: List[int]
    time_ticks: List[float]
    freq_ticks: List[float]
    db_ticks: List[float]
    sample_rate: int
    hop_length: int
    n_mels: int

PredictResponse.model_rebuild()
