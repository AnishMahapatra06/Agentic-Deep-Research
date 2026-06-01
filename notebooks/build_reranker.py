
from sentence_transformers import CrossEncoder

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

print("Loading reranker...")

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)

print("Reranker loaded successfully.")
