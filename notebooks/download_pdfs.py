
import os
import time
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

metadata_file = (
    f"{PROJECT_PATH}/data/metadata/paper_metadata.csv"
)

pdf_dir = (
    f"{PROJECT_PATH}/data/raw_pdfs"
)

os.makedirs(pdf_dir, exist_ok=True)

df = pd.read_csv(metadata_file)

downloaded = 0
skipped = 0
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):

    arxiv_id = row["arxiv_id"]

    pdf_url = row["pdf_url"]

    filename = os.path.join(
        pdf_dir,
        f"{arxiv_id}.pdf"
    )

    if os.path.exists(filename):
        skipped += 1
        continue

    try:

        response = requests.get(
            pdf_url,
            timeout=30
        )

        with open(filename, "wb") as f:
            f.write(response.content)

        downloaded += 1

        time.sleep(0.5)

    except Exception:
        failed += 1

print("\\nFinished")
print("Downloaded:", downloaded)
print("Skipped:", skipped)
print("Failed:", failed)
