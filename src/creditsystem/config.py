"""Configuration settings for the Credit System."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model settings
MODEL = "gemma:2b-instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Database settings
DB_NAME = "vector_db"

# Intent classification
ALLOWED_LABELS = {
    "explanation",
    "advisory",
    "risk_assessment",
    "simulation"
}

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.55
SIMULATION_CONFIDENCE_THRESHOLD = 0.65

# Retrieval settings
RETRIEVAL_K_EXPLANATION = 6
RETRIEVAL_K_SIMULATION = 8
RETRIEVAL_K_DEFAULT = 4

# Chroma distance to confidence conversion
# In Chroma, lower distance = better match
# We convert to similarity-style confidence: max(0, 1 - distance)
