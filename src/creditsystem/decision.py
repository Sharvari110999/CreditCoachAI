"""Decision engine module for the Credit System."""

from src.creditsystem.config import (
    CONFIDENCE_THRESHOLD,
    SIMULATION_CONFIDENCE_THRESHOLD,
)


def decide_execution(intent: str, confidence: float) -> str:
    """
    Decide whether to execute locally or use cloud fallback.
    
    Args:
        intent: The classified intent
        confidence: The confidence score from retrieval
    
    Returns:
        "local" or "cloud"
    """
    # Simple initial logic
    if confidence < CONFIDENCE_THRESHOLD:
        return "cloud"
    
    # Simulation queries require higher confidence
    if intent == "simulation" and confidence < SIMULATION_CONFIDENCE_THRESHOLD:
        return "cloud"
    
    return "local"
