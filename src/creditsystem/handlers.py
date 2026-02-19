"""Category handlers for the Credit System."""

from langchain_ollama import OllamaLLM

from src.creditsystem.config import MODEL
from src.creditsystem.rag import simple_retrieve


# Initialize LLM globally
_llm = None


def get_llm():
    """Get or create the LLM."""
    global _llm
    if _llm is None:
        _llm = OllamaLLM(temperature=0, model=MODEL)
    return _llm


def handle_explanation(question: str, context: str):
    """
    Handle explanation-type queries.
    
    Args:
        question: The user's question
        context: Retrieved context from RAG
    
    Returns:
        LLM response
    """
    llm = get_llm()
    
    prompt = f"""
    You are a UK credit education assistant.

    Use the verified knowledge base context below as your primary source.

    Context:
    {context}

    Question:
    {question}

    Provide a structured explanation:

    - Clear Definition
    - Why It Matters
    - Practical Example (if relevant)

    Do NOT invent statistics or exact credit score numbers.
    If context does not contain the answer, say so clearly.
    """

    return llm.invoke(prompt)


def handle_advisory(question: str, context: str):
    """
    Handle advisory-type queries.
    
    Args:
        question: The user's question
        context: Retrieved context from RAG
    
    Returns:
        LLM response
    """
    llm = get_llm()
    
    prompt = f"""
    You are a UK credit advisor.

    Use the following verified UK credit information:

    {context}

    Question:
    {question}

    Format:
    - Goal
    - 3 Action Steps
    - 1 Warning
    """

    return llm.invoke(prompt)


def handle_risk_assessment(question: str, context: str):
    """
    Handle risk_assessment-type queries.
    
    Args:
        question: The user's question
        context: Retrieved context from Returns:
        L RAG
    
   LM response
    """
    llm = get_llm()
    
    prompt = f"""
    You are a UK credit risk assessment engine.

    Use the following verified UK credit knowledge base context:

    {context}

    The user question is:
    {question}

    You will Estimate the following
    Risk Level:
    Baseline Reason: 

    Your task:
    - Validate or adjust the risk level if necessary.
    - Provide clear reasoning grounded in UK credit principles.
    - Do NOT invent numerical score changes.
    - Keep explanation concise and practical.

    Provide output in this format:

    - Risk Level (Low / Medium / High)
    - Why This Is Risky
    - Time Sensitivity (Immediate / Short-Term / Long-Term)
    - Recommended Precaution (1–2 steps)
    """

    return llm.invoke(prompt)


def handle_simulation(question: str, context: str):
    """
    Handle simulation-type queries (what-if scenarios).
    
    Args:
        question: The user's question
        context: Retrieved context from RAG
    
    Returns:
        LLM response
    """
    llm = get_llm()
    
    prompt = f"""
    You are a UK credit system simulation engine.

    Use the following verified UK credit knowledge base context:

    {context}

    The user is asking a what-if scenario.

    Question:
    {question}

    Provide structured output in this format:

    - Scenario Summary
    - Short-Term Impact (0–3 months)
    - Medium-Term Impact (3–12 months)
    - Estimated Risk Level (Low / Medium / High)
    - Recovery Strategy (3 steps)
    - Key Insight

    Use realistic UK principles.
    Do NOT hallucinate exact credit score numbers.
    """

    return llm.invoke(prompt)


# Mapping from intent to handler function
INTENT_HANDLERS = {
    "explanation": handle_explanation,
    "advisory": handle_advisory,
    "risk_assessment": handle_risk_assessment,
    "simulation": handle_simulation,
}


def route_to_handler(intent: str, question: str, context: str):
    """
    Route the question to the appropriate handler based on intent.
    
    Args:
        intent: The classified intent
        question: The user's question
        context: Retrieved context from RAG
    
    Returns:
        LLM response from the appropriate handler
    """
    handler = INTENT_HANDLERS.get(intent, handle_explanation)
    return handler(question, context)
