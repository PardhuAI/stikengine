from ultralytics import YOLO

# Load a pretrained YOLO model (v8 nano for speed)
try:
    model = YOLO("yolov8n.pt")
except:
    model = None

# Standard COCO has classes: person (0), knife (43), baseball bat (34), scissors (76)
TARGET_CLASSES = {
    43: "knife",
    34: "baseball bat",
    76: "scissors"
}

def moderate_image_yolo(image_path: str) -> dict:
    if not model:
        return {"weapon_score": 0.0, "decision": "SAFE", "error": "YOLO not loaded"}
        
    try:
        results = model(image_path)
        
        weapon_score = 0.0
        detections = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                if cls_id in TARGET_CLASSES:
                    class_name = TARGET_CLASSES[cls_id]
                    detections[class_name] = max(detections.get(class_name, 0.0), conf)
                    if conf > weapon_score:
                        weapon_score = conf
                        
        # Mock additional classes requested by the architect (until custom weights exist)
        if "gun" not in detections: detections["gun"] = 0.0
        if "rifle" not in detections: detections["rifle"] = 0.0
        if "syringe" not in detections: detections["syringe"] = 0.0
        if "knife" not in detections: detections["knife"] = 0.0
        
        decision = "SAFE"
        if weapon_score > 0.6:
            decision = "FLAG"
        elif weapon_score > 0.3:
            decision = "UNCERTAIN"
            
        return {
            "weapon_score": weapon_score,
            "violence_score": 0.0,
            "self_harm_score": 0.0,
            "detections": detections,
            "decision": decision
        }
        
    except Exception as e:
        return {
            "weapon_score": 0.0,
            "decision": "UNCERTAIN",
            "error": str(e)
        }
