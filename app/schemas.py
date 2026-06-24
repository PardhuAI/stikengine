from pydantic import BaseModel
from typing import Optional
import datetime

class UploadResponse(BaseModel):
    status: str
    category: str
    confidence: Optional[float]

class ModeratedImageResponse(BaseModel):
    id: int
    image_url: str
    upload_time: datetime.datetime
    width: Optional[int]
    height: Optional[int]
    image_hash: Optional[str]
    
    final_category: str
    final_decision: str
    final_score: Optional[float]
    
    nudenet_score: Optional[float]
    clip_score: Optional[float]
    yolo_score: Optional[float]
    gemini_score: Optional[float]
    
    nudenet_output: Optional[str]
    clip_output: Optional[str]
    yolo_output: Optional[str]
    gemini_output: Optional[str]
    
    review_status: str
    human_override: bool
    processing_time: Optional[float]

    class Config:
        from_attributes = True
