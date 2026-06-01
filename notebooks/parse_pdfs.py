
import os
import json
import fitz
from tqdm import tqdm

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

PDF_DIR = f"{PROJECT_PATH}/data/raw_pdfs"
PARSED_DIR = f"{PROJECT_PATH}/data/parsed"

os.makedirs(PARSED_DIR, exist_ok=True)

pdf_files = [
    f for f in os.listdir(PDF_DIR)
    if f.endswith(".pdf")
]

parsed_count = 0
failed_count = 0

for pdf_file in tqdm(pdf_files):

    arxiv_id = pdf_file.replace(".pdf", "")

    output_file = os.path.join(
        PARSED_DIR,
        f"{arxiv_id}.json"
    )

    if os.path.exists(output_file):
        continue

    try:

        pdf_path = os.path.join(
            PDF_DIR,
            pdf_file
        )

        doc = fitz.open(pdf_path)

        pages = []

        for page_num in range(len(doc)):

            text = doc[page_num].get_text("text")

            pages.append({
                "page": page_num + 1,
                "text": text
            })

        parsed_data = {
            "arxiv_id": arxiv_id,
            "num_pages": len(doc),
            "pages": pages
        }

        with open(output_file, "w") as f:
            json.dump(
                parsed_data,
                f
            )

        parsed_count += 1

    except Exception:
        failed_count += 1

print("\\nParsing Complete")
print("Parsed:", parsed_count)
print("Failed:", failed_count)
