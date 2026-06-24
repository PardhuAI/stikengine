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
    
    # 1. Absolute highest priority violations
    if nude_score >= 0.85:
        return ("NUDITY", "FLAG")
        
    if weapon_score >= 0.60:
        return ("WEAPON", "FLAG")
        
    # 2. Risk overrides Context. If an image is a sports photo BUT has an upskirt view > 0.80, it's flagged.
    if max_risk > 0.80:
        return ("SUGGESTIVE", "FLAG")
        
    # 3. If no high risk, but a strong safe context is identified, mark as SAFE
    if max_context > 0.85:
        return ("SAFE", "SAFE")
        
    # 4. Tie-breaker: strict value comparison between Risk and Context
    if max_risk > max_context:
        return ("SUGGESTIVE", "FLAG")
    else:
        return ("SAFE", "SAFE")
