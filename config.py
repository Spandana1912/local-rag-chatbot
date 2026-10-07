"""
Central configuration for Ask My Documents.
Change defaults here instead of hunting through the codebase.
"""
from pathlib import Path

# Paths (anchored to project root so Streamlit cwd changes don't break the DB)
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "my_documents"

# Models (must be pulled via `ollama pull <name>`)
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Retrieval & Advanced RAG Settings
DEFAULT_N_RESULTS = 5
CANDIDATE_POOL = 15
RRF_K = 60
MIN_RELEVANCE_SCORE = 0.01  # below this → treat as "nothing relevant"
ENABLE_MULTI_QUERY = True
ENABLE_HYDE = False  # Hypothetical Document Embeddings
ENABLE_QUERY_REWRITE = True

# Chunking
CHUNK_SIZE = 800
OVERLAP_SENTENCES = 1

# Conversation memory (turns kept for multi-turn context)
HISTORY_WINDOW = 6

# UI
PAGE_TITLE = "Ask My Documents Pro — Advanced Local RAG"
PAGE_ICON = "⚡"
MAX_UPLOAD_MB = 50

