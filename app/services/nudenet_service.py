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

def moderate_image_nudenet(crops: dict) -> dict:
    highest_score = 0.0
    highest_crop = "original"
    all_detections = {}
    
    try:
        for crop_name, crop_path in crops.items():
            results = detector.detect(crop_path)
            
            crop_detections = []
            for result in results:
                label = result['class']
                score = result['score']
                
                # Only care about strictly exposed labels for NudeNet V2
                if label in SENSITIVE_LABELS:
                    crop_detections.append(f"{label} ({score:.2f})")
                    if score > highest_score:
                        highest_score = score
                        highest_crop = crop_name
            
            all_detections[crop_name] = crop_detections
            
        decision = "SAFE"
        if highest_score >= 0.85:
            decision = "FLAG"
        elif highest_score >= 0.40:
            decision = "UNCERTAIN"
            
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
