"""Intent classification module for the Credit System."""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM

from src.creditsystem.config import MODEL, ALLOWED_LABELS


# Initialize LLM globally
_llm = None


def get_llm():
    """Get or create the LLM."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM(temperature=0, model=MODEL)
    return _llm


SYSTEM_PROMPT_TEMPLATE = """
You are a strict intent classifier.

Classify the user query into EXACTLY one of these labels:

explanation
advisory
risk_assessment
simulation

Label meanings:

explanation = asking to define or explain something.
advisory = asking what action to take OR how to do something.
risk_assessment = asking if something is bad, harmful, or risky.
simulation = describing a specific hypothetical situation and asking what will happen.

Rules:
- Output ONLY one label.
- No extra words.
- No punctuation.
"""


def classify_intent(question: str) -> str:
    """
    Classify the intent of a question.
    
    Args:
        question: The question to classify
    
    Returns:
        The intent label (one of: explanation, advisory, risk_assessment, simulation)
    """
    llm = get_llm()
    
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(content=question)
    ])
    
    label = response.strip().lower()
    
    if label not in ALLOWED_LABELS:
        return "explanation"  # Default fallback
    
    return label
