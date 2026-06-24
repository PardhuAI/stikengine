import hashlib
from fastapi import UploadFile, HTTPException
from PIL import Image, ImageOps
import os
import uuid
import shutil

MAX_FILE_SIZE = 10 * 1024 * 1024 # 10 MB

def calculate_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def validate_and_process_image(file: UploadFile) -> dict:
    # Check size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10MB.")
        
    ext = file.filename.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png', 'webp']:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    filename = f"{uuid.uuid4()}.{ext}"
    temp_path = f"uploads/{filename}"
    
    # Save original bytes
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Calculate hash
    img_hash = calculate_hash(temp_path)
    
    # Open, normalize EXIF orientation, and save crops
    try:
        img = Image.open(temp_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save normalized image
        img.save(temp_path)
        width, height = img.size
        
        crops = {
            "original": temp_path
        }
        
        # Generate crops (simple division)
        # top
        top_crop = img.crop((0, 0, width, height // 2))
        top_path = f"uploads/{uuid.uuid4()}_top.jpg"
        top_crop.save(top_path)
        crops["top"] = top_path
        
        # bottom
        bottom_crop = img.crop((0, height // 2, width, height))
        bottom_path = f"uploads/{uuid.uuid4()}_bottom.jpg"
        bottom_crop.save(bottom_path)
        crops["bottom"] = bottom_path
        
        # left
        left_crop = img.crop((0, 0, width // 2, height))
        left_path = f"uploads/{uuid.uuid4()}_left.jpg"
        left_crop.save(left_path)
        crops["left"] = left_path
        
        # right
        right_crop = img.crop((width // 2, 0, width, height))
        right_path = f"uploads/{uuid.uuid4()}_right.jpg"
        right_crop.save(right_path)
        crops["right"] = right_path
        
        # center
        cw, ch = width // 2, height // 2
        cx, cy = width // 4, height // 4
        center_crop = img.crop((cx, cy, cx + cw, cy + ch))
        center_path = f"uploads/{uuid.uuid4()}_center.jpg"
        center_crop.save(center_path)
        crops["center"] = center_path
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    return {
        "original_path": temp_path,
        "hash": img_hash,
        "width": width,
        "height": height,
        "crops": crops
    }

def cleanup_crops(crops: dict):
    for name, path in crops.items():
        if name != "original" and os.path.exists(path):
            os.remove(path)
