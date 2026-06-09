def calculate_driver_score(hazards, context):
    """
    Calculates a driver safety score based on detected hazards and situational context.
    
    Args:
        hazards (list): List of dicts with 'severity'.
        context (dict): Dict with 'risk_multiplier'.
        
    Returns:
        dict: contains score, rating, and total_deductions.
    """
    base_score = 100
    severity_weights = {
        "critical": 30,
        "high": 20,
        "medium": 10,
        "low": 5
    }
    
    raw_deductions = 0
    for hazard in hazards:
        severity = hazard.get("severity", "low").lower()
        raw_deductions += severity_weights.get(severity, 5)
        
    risk_multiplier = context.get("risk_multiplier", 1.0)
    total_deductions = int(raw_deductions * risk_multiplier)
    
    score = max(0, base_score - total_deductions)
    
    if score > 80:
        rating = "Excellent"
    elif score > 60:
        rating = "Good"
    elif score > 40:
        rating = "Fair"
    else:
        rating = "Poor"
        
    return {
        "score": score,
        "rating": rating,
        "total_deductions": total_deductions
    }
