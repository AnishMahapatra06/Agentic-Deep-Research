
# =========================
# PATHS
# =========================

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

RAW_PDF_DIR = f"{PROJECT_PATH}/data/raw_pdfs"
PARSED_DIR = f"{PROJECT_PATH}/data/parsed"
CHUNK_DIR = f"{PROJECT_PATH}/data/chunks"
METADATA_DIR = f"{PROJECT_PATH}/data/metadata"

FAISS_DIR = f"{PROJECT_PATH}/index/faiss"
BM25_DIR = f"{PROJECT_PATH}/index/bm25"

# =========================
# CORPUS SETTINGS
# =========================

START_YEAR = 2024
END_YEAR = 2026

MAX_PAPERS = 700

# =========================
# CHUNKING
# =========================

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# =========================
# RETRIEVAL
# =========================

TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 8

DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# =========================
# AGENT
# =========================

MAX_REFLECTION_ROUNDS = 3

# =========================
# MODELS
# =========================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
