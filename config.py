"""
Central configuration for the Enterprise Knowledge RAG Assistant.
All tunable parameters live here so the rest of the codebase never
hardcodes a model name, path, or threshold.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load variables from a .env file in the project root, if present.
# Falls back silently to real environment variables (e.g. exported in the
# shell, or set by Colab/Docker/CI) if python-dotenv isn't installed or no
# .env file exists -- so this is safe either way.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env",override=True)
except ImportError:
    pass


# ---------------------------------------------------------------------
# Top-level free/paid switch. This is the one flag to flip when you want
# to compare free-model output against paid-model output.
#   RAG_MODE=free  -> chat via OpenRouter's openai/gpt-oss-20b:free
#                      (no OpenAI billing), embeddings via a local
#                      sentence-transformers model (no API call at all).
#   RAG_MODE=paid  -> chat + embeddings via OpenAI (or Azure OpenAI, see
#                      PROVIDER below) -- this is the original behavior.
# ---------------------------------------------------------------------
RAG_MODE = os.getenv("RAG_MODE", "paid")  # "free" | "paid"

# --- Free mode settings (OpenRouter for chat, local model for embeddings) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-oss-120b")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------
# Provider settings (swap PROVIDER to "azure_openai" to move to Azure
# OpenAI without touching any other file -- see rag/llm_client.py)
# ---------------------------------------------------------------------
PROVIDER = os.getenv("RAG_PROVIDER", "openai")  # "openai" | "azure_openai"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536-dim
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Azure OpenAI (only used when PROVIDER == "azure_openai")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")

# ---------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------
RAW_DOCS_DIR = str(BASE_DIR / "data" / "sample_docs")
CHROMA_DIR = str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = f"enterprise_kb_{RAG_MODE}"
MANIFEST_PATH = str(BASE_DIR / "data" / f"ingestion_manifest_{RAG_MODE}.json")
EVAL_RESULTS_PATH = str(BASE_DIR / "eval" / f"eval_results_{RAG_MODE}.json")
EVAL_TESTSET_PATH = str(BASE_DIR / "eval" / "eval_testset.json")

# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------
CHUNK_SIZE = 800          # characters, not tokens (simple + predictable)
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 50       # drop near-empty trailing chunks

# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------
TOP_K_RETRIEVE = 12       # candidates pulled from vector store before rerank
TOP_K_RERANK = 4          # final chunks passed to the LLM as context
USE_HYBRID_SEARCH = True  # combine dense (embeddings) + sparse (BM25) retrieval
HYBRID_ALPHA = 0.5        # weight of dense score vs sparse score (0=sparse only, 1=dense only)

# Confidence thresholds (based on top reranked similarity score, 0-1)
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.55
# below CONFIDENCE_MEDIUM -> "Low confidence" and assistant hedges / asks to verify

# ---------------------------------------------------------------------
# Access control (simple metadata-based filtering, demonstrates the
# pattern enterprises need -- swap for real ACL/entra groups in prod)
# ---------------------------------------------------------------------
DEPARTMENTS = ["HR", "IT", "Finance", "Operations", "General"]
SENSITIVITY_LEVELS = ["Public", "Internal", "Confidential"]

# ---------------------------------------------------------------------
# Evaluation metrics (ROUGE / BERTScore / LLM-as-judge)
# ---------------------------------------------------------------------
# CPU-friendly by default -- roberta-large (the bert-score package default)
# is accurate but slow without a GPU. Swap to a larger model if you have one.
EVAL_BERTSCORE_MODEL = os.getenv("EVAL_BERTSCORE_MODEL", "distilbert-base-uncased")
EVAL_BERTSCORE_LANG = os.getenv("EVAL_BERTSCORE_LANG", "en")
# Judge reuses the same chat model/provider already configured above (RAG_MODE
# / PROVIDER) so no separate API key is needed.
EVAL_JUDGE_TEMPERATURE = 0.0
