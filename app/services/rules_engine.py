def evaluate_rules(nudenet_out: dict, clip_out: dict, yolo_out: dict) -> tuple[str, str]:
    """
    Returns (category, final_decision)
    category: SAFE, NUDITY, NEAR_NUDITY, SUGGESTIVE, WEAPON, VIOLENCE, DRUGS, REVIEW
    decision: SAFE, FLAG, REVIEW
    """
    nude_score = nudenet_out.get("nudity_score", 0.0)
    weapon_score = yolo_out.get("weapon_score", 0.0)
    
    max_risk = clip_out.get("max_risk_score", 0.0)
    max_context = clip_out.get("max_context_score", 0.0)
    
    # ---------------------------------------------------------
    # 1. AUTO-FLAG: Absolute highest priority violations
    # ---------------------------------------------------------
    if nude_score >= 0.85:
        return ("NUDITY", "FLAG")
        
    if weapon_score >= 0.60:
        return ("WEAPON", "FLAG")
        
    # ---------------------------------------------------------
    # 2. AUTO-FLAG: Cross-Model Synergy (Your requested logic)
    # ---------------------------------------------------------
    # If NudeNet sees suspicious skin (0.300+) AND CLIP is highly confident of a risky scene (0.950+)
    if nude_score >= 0.300 and max_risk >= 0.950 and max_risk > max_context:
        return ("SUGGESTIVE", "FLAG")
        
    # ---------------------------------------------------------
    # 3. ESCALATE TO GEMINI: The "Review" Tripwires
    # ---------------------------------------------------------
    # If NudeNet or YOLO specifically marked it as UNCERTAIN
    if nudenet_out.get("decision") == "UNCERTAIN" or yolo_out.get("decision") == "UNCERTAIN":
        return ("REVIEW", "REVIEW")
        
    # If CLIP thinks it's risky (0.50 to 0.949), but it didn't trigger the synergy rule above.
    # This catches the ambiguous middle-ground (e.g., standard bikinis vs. lingerie) for Gemini to decide.
    if max_risk > max_context and max_risk >= 0.50:
        return ("SUGGESTIVE", "REVIEW")
        
    # ---------------------------------------------------------
    # 4. SAFE: Fallback
    # ---------------------------------------------------------
    # If no risk thresholds were met, or Safe Context overpowered the Risk
    return ("SAFE", "SAFE")
