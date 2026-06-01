
import os
import json
from tqdm import tqdm

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

PARSED_DIR = f"{PROJECT_PATH}/data/parsed"
CHUNK_DIR = f"{PROJECT_PATH}/data/chunks"

os.makedirs(CHUNK_DIR, exist_ok=True)

CHUNK_SIZE = 1200
OVERLAP = 200

json_files = [
    f for f in os.listdir(PARSED_DIR)
    if f.endswith(".json")
]

total_chunks = 0

for file_name in tqdm(json_files):

    file_path = os.path.join(
        PARSED_DIR,
        file_name
    )

    with open(file_path, "r") as f:
        paper = json.load(f)

    arxiv_id = paper["arxiv_id"]

    full_text = ""

    for page in paper["pages"]:
        full_text += page["text"] + "\n"

    chunks = []

    start = 0

    while start < len(full_text):

        end = start + CHUNK_SIZE

        chunk_text = full_text[start:end]

        chunks.append({
            "arxiv_id": arxiv_id,
            "chunk_id": len(chunks),
            "text": chunk_text
        })

        start += (CHUNK_SIZE - OVERLAP)

    total_chunks += len(chunks)

    output_file = os.path.join(
        CHUNK_DIR,
        f"{arxiv_id}.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            chunks,
            f
        )

print("Total Chunks:", total_chunks)
