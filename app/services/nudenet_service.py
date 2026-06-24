import json
from nudenet import NudeDetector

detector = NudeDetector()

# NudeNet labels in V2 only focus on exposed body parts.
SENSITIVE_LABELS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED"
]

# NEW: Tripwires for sheer/tight clothing
SUGGESTIVE_LABELS = [
    "FEMALE_BREAST_COVERED", 
    "BUTTOCKS_COVERED", 
    "FEMALE_GENITALIA_COVERED"
]

def moderate_image_nudenet(crops: dict) -> dict:
    highest_score = 0.0
    highest_crop = "original"
    has_suggestive = False
    all_detections = {}
    
    try:
        for crop_name, crop_path in crops.items():
            results = detector.detect(crop_path)
            
            crop_detections = []
            for result in results:
                label = result['class']
                score = result['score']
                
                # Check for absolute nudity
                if label in SENSITIVE_LABELS:
                    crop_detections.append(f"{label} ({score:.2f})")
                    if score > highest_score:
                        highest_score = score
                        highest_crop = crop_name
                # NEW: Check for suggestive clothing (tripwire for Gemini)
                elif label in SUGGESTIVE_LABELS and score > 0.60:
                    crop_detections.append(f"{label} ({score:.2f})")
                    has_suggestive = True
            
            all_detections[crop_name] = crop_detections
            
        decision = "SAFE"
        if highest_score >= 0.85:
            decision = "FLAG"
        elif highest_score >= 0.40 or has_suggestive:
            decision = "UNCERTAIN" # This routes the image to the Rules Engine tripwire
            
        return {
            "nudity_score": highest_score,
            "crop": highest_crop,
            "detections": all_detections,
            "decision": decision
        }
            
    except Exception as e:
        return {
            "nudity_score": 0.0,
            "crop": "error",
            "detections": {"error": str(e)},
            "decision": "UNCERTAIN"
        }
