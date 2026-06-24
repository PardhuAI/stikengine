from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
import datetime
from .database import Base

class ModeratedImage(Base):
    __tablename__ = "moderated_images"

    id = Column(Integer, primary_key=True, index=True)
    image_url = Column(String, index=True)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Layer metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    image_hash = Column(String, index=True, nullable=True)
    
    # Scores
    nudenet_score = Column(Float, nullable=True)
    clip_score = Column(Float, nullable=True)
    yolo_score = Column(Float, nullable=True)
    gemini_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    
    # Raw JSON outputs
    nudenet_output = Column(String, nullable=True)
    clip_output = Column(String, nullable=True)
    yolo_output = Column(String, nullable=True)
    gemini_output = Column(String, nullable=True)
    
    # Final decisions
    final_category = Column(String) # NUDITY, SUGGESTIVE, SAFE, etc.
    final_decision = Column(String) # SAFE, FLAG
    review_status = Column(String, default="REVIEWED") # PENDING, REVIEWED
    human_override = Column(Boolean, default=False)
    processing_time = Column(Float, nullable=True)
