import os
import shutil
import uuid
import time
import json
import datetime
from typing import List

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import engine, Base, get_db
from .models import ModeratedImage
from .schemas import UploadResponse, ModeratedImageResponse
from .services.validation_service import validate_and_process_image, cleanup_crops
from .services.nudenet_service import moderate_image_nudenet
from .services.clip_service import moderate_image_clip
from .services.yolo_service import moderate_image_yolo
from .services.rules_engine import evaluate_rules
from .services.gemini_service import moderate_image_gemini

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="StikGuard AI V2")

# Mount static files and uploads
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Ensure required directories exist
os.makedirs("uploads/safe", exist_ok=True)
os.makedirs("uploads/flagged", exist_ok=True)
os.makedirs("uploads/review", exist_ok=True)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start_time = time.time()
    
    # Layer 1: Validation & Cropping
    val_data = validate_and_process_image(file)
    original_path = val_data["original_path"]
    crops = val_data["crops"]
    
    # Check for duplicate hash
    existing = db.query(ModeratedImage).filter(ModeratedImage.image_hash == val_data["hash"]).first()
    if existing:
        cleanup_crops(crops)
        if os.path.exists(original_path):
            os.remove(original_path)
        return UploadResponse(
            status=existing.final_decision,
            category=existing.final_category,
            confidence=existing.final_score
        )
    
    # Layer 2: NudeNet Multi-crop
    nudenet_out = moderate_image_nudenet(crops)
    
    # Layer 3: OpenCLIP
    clip_out = moderate_image_clip(original_path)
    
    # Layer 4: YOLO
    yolo_out = moderate_image_yolo(original_path)
    
    # Layer 5: Rules Engine
    category, decision = evaluate_rules(nudenet_out, clip_out, yolo_out)
    
    gemini_out = None
    gemini_score = None
    
    # Layer 6: Gemini Escalation
    if decision == "REVIEW":
        gemini_out = moderate_image_gemini(original_path, nudenet_out, clip_out, yolo_out)
        decision = gemini_out.get("decision", "FLAG")
        category = gemini_out.get("category", "REVIEW")
        gemini_score = gemini_out.get("confidence", 0.0)
        
        # If Gemini fails, explicitly route to human review queue
        if gemini_out.get("reason", "").startswith("Gemini API Error"):
            decision = "REVIEW"
    
    # Clean up temporary crops
    cleanup_crops(crops)
    
    # Move file to final destination
    if decision == "SAFE":
        final_dir = "uploads/safe"
    elif decision == "FLAG":
        final_dir = "uploads/flagged"
    else:
        final_dir = "uploads/review"
        
    filename = os.path.basename(original_path)
    final_path = f"{final_dir}/{filename}"
    os.rename(original_path, final_path)
    
    processing_time = time.time() - start_time
    
    # Extract final score (highest from models)
    final_score = gemini_score if gemini_score else max(nudenet_out.get("nudity_score", 0.0), clip_out.get("clip_score", 0.0), yolo_out.get("weapon_score", 0.0))
    
    # Save to database
    db_image = ModeratedImage(
        image_url=final_path,
        upload_time=datetime.datetime.utcnow(),
        width=val_data["width"],
        height=val_data["height"],
        image_hash=val_data["hash"],
        nudenet_score=nudenet_out.get("nudity_score", 0.0),
        clip_score=clip_out.get("clip_score", 0.0),
        yolo_score=yolo_out.get("weapon_score", 0.0),
        gemini_score=gemini_score,
        final_score=final_score,
        nudenet_output=json.dumps(nudenet_out),
        clip_output=json.dumps(clip_out),
        yolo_output=json.dumps(yolo_out),
        gemini_output=json.dumps(gemini_out) if gemini_out else None,
        final_category=category,
        final_decision=decision,
        review_status="REVIEWED" if decision != "REVIEW" else "PENDING",
        processing_time=processing_time
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    return UploadResponse(
        status=decision,
        category=category,
        confidence=final_score
    )

@app.post("/human-review/{image_id}")
def human_override(image_id: int, action: str, db: Session = Depends(get_db)):
    db_image = db.query(ModeratedImage).filter(ModeratedImage.id == image_id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    if action == "approve":
        new_decision = "SAFE"
    elif action == "reject":
        new_decision = "FLAG"
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
        
    old_path = db_image.image_url
    filename = os.path.basename(old_path)
    final_dir = "uploads/safe" if new_decision == "SAFE" else "uploads/flagged"
    new_path = f"{final_dir}/{filename}"
    
    if os.path.exists(old_path) and old_path != new_path:
        os.rename(old_path, new_path)
        
    db_image.image_url = new_path
    db_image.final_decision = new_decision
    db_image.review_status = "REVIEWED"
    db_image.human_override = True
    
    db.commit()
    return {"status": "success"}

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(ModeratedImage).count()
    safe = db.query(ModeratedImage).filter(ModeratedImage.final_decision == "SAFE").count()
    flagged = db.query(ModeratedImage).filter(ModeratedImage.final_decision == "FLAG").count()
    pending = db.query(ModeratedImage).filter(ModeratedImage.review_status == "PENDING").count()
    
    nudity = db.query(ModeratedImage).filter(ModeratedImage.final_category == "NUDITY").count()
    suggestive = db.query(ModeratedImage).filter(ModeratedImage.final_category == "SUGGESTIVE").count()
    weapons = db.query(ModeratedImage).filter(ModeratedImage.final_category == "WEAPON").count()
    
    gemini_used = db.query(ModeratedImage).filter(ModeratedImage.gemini_output != None).count()
    
    return {
        "total_uploads": total,
        "safe_count": safe,
        "flagged_count": flagged,
        "review_count": pending,
        "violations": {
            "nudity": nudity,
            "suggestive": suggestive,
            "weapons": weapons
        },
        "gemini_usage": gemini_used
    }

@app.get("/images/{status}", response_model=List[ModeratedImageResponse])
def get_images(status: str, db: Session = Depends(get_db)):
    status = status.lower()
    
    if status == "all":
        images = db.query(ModeratedImage).order_by(ModeratedImage.upload_time.desc()).all()
    elif status == "review":
        images = db.query(ModeratedImage).filter(ModeratedImage.review_status == "PENDING").order_by(ModeratedImage.upload_time.desc()).all()
    elif status == "safe":
        images = db.query(ModeratedImage).filter(ModeratedImage.final_decision == "SAFE").order_by(ModeratedImage.upload_time.desc()).all()
    elif status == "flagged" or status == "flag":
        images = db.query(ModeratedImage).filter(ModeratedImage.final_decision == "FLAG").order_by(ModeratedImage.upload_time.desc()).all()
    elif status == "gemini":
        images = db.query(ModeratedImage).filter(ModeratedImage.gemini_output != None).order_by(ModeratedImage.upload_time.desc()).all()
    else:
        raise HTTPException(status_code=400, detail="Invalid status")
        
    return images
