"""Kivi Semantic Memory — Configuration"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model pins (override via .env)
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemini-3.5-flash-lite")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# Temperature
EXTRACTION_TEMPERATURE = 0.1
GENERATION_TEMPERATURE = 0.2
VERIFICATION_TEMPERATURE = 0.1

# --- Retrieval weights (fuzzy_search signal fusion) ---
WEIGHT_SEMANTIC = float(os.getenv("WEIGHT_SEMANTIC", "0.45"))
WEIGHT_LEXICAL = float(os.getenv("WEIGHT_LEXICAL", "0.35"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.20"))

# Recency decay half-life in days
RECENCY_HALF_LIFE_DAYS = 7.0

# --- Rate limiting ---
MAX_RETRIES = 8
INITIAL_RETRY_DELAY = 3.0  # seconds
RETRY_BACKOFF_FACTOR = 1.5

# --- Ingestion ---
EXTRACTION_BATCH_SIZE = 10
EMBEDDING_BATCH_SIZE = 100
INTER_BATCH_DELAY = 1.0  # seconds between batches

# --- Dictionary auto-suggestion ---
DICTIONARY_SUGGESTION_THRESHOLD = 3  # occurrences before suggesting

# --- Database ---
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "data", "kivi.db"))

# --- Server ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
