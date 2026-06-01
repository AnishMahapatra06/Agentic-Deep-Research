
import os
import json
import pickle
import numpy as np
import faiss

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

CHUNK_DIR = f"{PROJECT_PATH}/data/chunks"

FAISS_DIR = f"{PROJECT_PATH}/index/faiss"

os.makedirs(FAISS_DIR, exist_ok=True)

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

all_texts = []
metadata = []

for file_name in os.listdir(CHUNK_DIR):

    if not file_name.endswith(".json"):
        continue

    with open(
        os.path.join(CHUNK_DIR, file_name),
        "r"
    ) as f:

        chunks = json.load(f)

    for chunk in chunks:

        all_texts.append(
            chunk["text"]
        )

        metadata.append({
            "arxiv_id": chunk["arxiv_id"],
            "chunk_id": chunk["chunk_id"]
        })

print("Total texts:", len(all_texts))

embeddings = model.encode(
    all_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

index = faiss.IndexFlatIP(
    embeddings.shape[1]
)

index.add(
    embeddings
)

faiss.write_index(
    index,
    f"{FAISS_DIR}/paper_index.faiss"
)

with open(
    f"{FAISS_DIR}/metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )

print("FAISS index saved.")
