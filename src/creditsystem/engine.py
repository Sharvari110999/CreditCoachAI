"""Main engine module for the Credit System."""

from src.creditsystem.config import MODEL
from src.creditsystem.intent import classify_intent
from src.creditsystem.rag import retrieve_context
from src.creditsystem.handlers import route_to_handler
from src.creditsystem.decision import decide_execution
from src.creditsystem.cloud import handle_cloud_execution


class CreditSystemEngine:
    """
    Main engine for the Credit System that ties together all components.
    """
    
    def __init__(self):
        """Initialize the Credit System engine."""
        pass
    
    def process_question(self, question: str) -> str:
        """
        Process a question through the full pipeline.
        
        Args:
            question: The user's question
        
        Returns:
            The response from the appropriate handler
        """
        # Classify intent
        intent = classify_intent(question)
        
        # Retrieve context with confidence
        context, confidence = retrieve_context(question, intent)
        
        # Decide execution path
        decision = decide_execution(intent, confidence)
        
        # Print debug info
        print(f"[Intent: {intent}] [Confidence: {confidence:.3f}] [Decision: {decision}]")
        
        if decision == "local":
            # Route to appropriate handler
            return route_to_handler(intent, question, context)
        else:
            # Use cloud fallback
            print("Escalating to Gemini (Cloud)")
            return handle_cloud_execution(question, intent)
    
    def route_question(self, question: str) -> str:
        """
        Route a question to the appropriate handler (simpler version without confidence).
        
        Args:
            question: The user's question
        
        Returns:
            The response from the appropriate handler
        """
        # Classify intent
        intent = classify_intent(question)
        
        # Simple retrieval without confidence
        from src.creditsystem.rag import simple_retrieve
        context = simple_retrieve(question)
        
        # Route to appropriate handler
        return route_to_handler(intent, question, context)


def create_engine() -> CreditSystemEngine:
    """
    Create and return a CreditSystemEngine instance.
    
    Returns:
        A CreditSystemEngine instance
    """
    return CreditSystemEngine()
