"""Cloud fallback module for the Credit System using Gemini."""

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from src.creditsystem.config import MODEL


# Load environment variables
load_dotenv()


# Initialize Gemini LLM globally
_gemini_llm = None


def get_gemini_llm():
    """Get or create the Gemini LLM."""
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    return _gemini_llm


def handle_cloud_execution(question: str, intent: str):
    """
    Handle cloud execution using Gemini when local retrieval has low confidence.
    
    Args:
        question: The user's question
        intent: The classified intent
    
    Returns:
        Gemini response
    """
    gemini_llm = get_gemini_llm()
    
    prompt = f"""
    You are an expert UK credit advisor.

    The local system had low retrieval confidence.
    Provide a structured answer for this intent: {intent}

    Question:
    {question}

    Follow the same structured format used by the system.
    Do NOT fabricate exact score numbers.
    """

    response = gemini_llm.invoke(prompt)
    
    return response.content
