import torch
import open_clip
from PIL import Image

# Initialize OpenCLIP
device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

CONTEXT_CONCEPTS = [
    "sports photo",
    "family photo",
    "travel photo",
    "vacation",
    "pet photo",
    "food image",
    "marketplace product",
    "beach photo",
    "portrait photo"
]

RISK_CONCEPTS = [
    "upskirt view",
    "camera focused on buttocks",
    "camera focused on crotch area",
    "sexualized pose",
    "revealing pose",
    "provocative pose",
    "micro bikini",
    "extremely revealing clothing",
    "minimal clothing",
    "nearly nude person",
    "transparent clothing",
    "sheer clothing",
    "see-through clothing",
    "mesh clothing"
]

# Precompute independent pairs for multi-label scoring
text_feature_pairs = {}
with torch.no_grad():
    for concept in CONTEXT_CONCEPTS + RISK_CONCEPTS:
        # Compare each concept against a generic null-hypothesis to get an independent 0-1 score
        tokens = tokenizer([concept, "a normal everyday photo"]).to(device)
        feat = model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
        text_feature_pairs[concept] = feat

def moderate_image_clip(image_path: str) -> dict:
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            context_scores = {}
            for concept in CONTEXT_CONCEPTS:
                tf = text_feature_pairs[concept]
                probs = (100.0 * image_features @ tf.T).softmax(dim=-1)
                context_scores[concept] = probs[0][0].item()
                
            risk_scores = {}
            for concept in RISK_CONCEPTS:
                tf = text_feature_pairs[concept]
                probs = (100.0 * image_features @ tf.T).softmax(dim=-1)
                risk_scores[concept] = probs[0][0].item()
                
        # Get top matches
        max_context = max(context_scores.values()) if context_scores else 0.0
        best_context = max(context_scores, key=context_scores.get) if context_scores else ""
        
        max_risk = max(risk_scores.values()) if risk_scores else 0.0
        best_risk = max(risk_scores, key=risk_scores.get) if risk_scores else ""
        
        # Sort and get top 5 context and risk for debugging/dashboard
        sorted_context = dict(sorted(context_scores.items(), key=lambda item: item[1], reverse=True)[:5])
        sorted_risk = dict(sorted(risk_scores.items(), key=lambda item: item[1], reverse=True)[:5])
        
        # We return the scores and let Rules Engine decide
        return {
            "clip_score": max_risk if max_risk > max_context else max_context, # Overall highest score
            "max_risk_score": max_risk,
            "best_risk": best_risk,
            "max_context_score": max_context,
            "best_context": best_context,
            "top_contexts": sorted_context,
            "top_risks": sorted_risk,
            "decision": "UNCERTAIN" # Defer strictly to Rules Engine
        }
    except Exception as e:
        return {
            "clip_score": 0.0,
            "max_risk_score": 0.0,
            "max_context_score": 0.0,
            "decision": "UNCERTAIN",
            "error": str(e)
        }
