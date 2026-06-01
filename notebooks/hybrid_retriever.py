
import os
import json
import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

FAISS_DIR = f"{PROJECT_PATH}/index/faiss"
BM25_DIR = f"{PROJECT_PATH}/index/bm25"
CHUNK_DIR = f"{PROJECT_PATH}/data/chunks"

print("Loading FAISS...")
index = faiss.read_index(
    f"{FAISS_DIR}/paper_index.faiss"
)

with open(
    f"{FAISS_DIR}/metadata.pkl",
    "rb"
) as f:
    faiss_metadata = pickle.load(f)

print("Loading BM25...")
with open(
    f"{BM25_DIR}/bm25.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

with open(
    f"{BM25_DIR}/metadata.pkl",
    "rb"
) as f:
    bm25_metadata = pickle.load(f)

print("Loading model...")
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device="cuda"
)

chunk_lookup = {}

for file_name in os.listdir(CHUNK_DIR):

    if not file_name.endswith(".json"):
        continue

    with open(
        os.path.join(CHUNK_DIR, file_name),
        "r"
    ) as f:

        chunks = json.load(f)

    for chunk in chunks:

        chunk_lookup[
            (chunk["arxiv_id"], chunk["chunk_id"])
        ] = chunk["text"]

print("Ready.")

def hybrid_search(query,
                  dense_k=20,
                  sparse_k=20):

    q_emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    dense_scores, dense_ids = index.search(
        q_emb.astype(np.float32),
        dense_k
    )

    bm25_scores = bm25.get_scores(
        query.split()
    )

    sparse_ids = np.argsort(
        bm25_scores
    )[::-1][:sparse_k]

    results = {}

    for score, idx in zip(
        dense_scores[0],
        dense_ids[0]
    ):

        meta = faiss_metadata[idx]

        key = (
            meta["arxiv_id"],
            meta["chunk_id"]
        )

        results[key] = float(score)

    for idx in sparse_ids:

        meta = bm25_metadata[idx]

        key = (
            meta["arxiv_id"],
            meta["chunk_id"]
        )

        if key not in results:
            results[key] = 0

        results[key] += 1.0

    ranked = sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for key, score in ranked[:10]:

        final_results.append({
            "arxiv_id": key[0],
            "chunk_id": key[1],
            "score": score,
            "text": chunk_lookup[key]
        })

    return final_results

query = "How do agent memory systems improve long-horizon task performance?"

results = hybrid_search(query)

for i, r in enumerate(results):

    print("="*80)
    print("Rank:", i+1)
    print("Paper:", r["arxiv_id"])
    print("Score:", r["score"])
    print()
    print(r["text"][:800])
