
import os
import json
import pickle

from rank_bm25 import BM25Okapi
from tqdm import tqdm

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

CHUNK_DIR = f"{PROJECT_PATH}/data/chunks"
BM25_DIR = f"{PROJECT_PATH}/index/bm25"

os.makedirs(BM25_DIR, exist_ok=True)

documents = []
metadata = []

for file_name in tqdm(os.listdir(CHUNK_DIR)):

    if not file_name.endswith(".json"):
        continue

    with open(
        os.path.join(CHUNK_DIR, file_name),
        "r"
    ) as f:

        chunks = json.load(f)

    for chunk in chunks:

        text = chunk["text"]

        documents.append(
            text.split()
        )

        metadata.append({
            "arxiv_id": chunk["arxiv_id"],
            "chunk_id": chunk["chunk_id"]
        })

print("Building BM25...")

bm25 = BM25Okapi(documents)

with open(
    f"{BM25_DIR}/bm25.pkl",
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )

with open(
    f"{BM25_DIR}/metadata.pkl",
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )

print("BM25 index saved.")
print("Documents:", len(documents))
