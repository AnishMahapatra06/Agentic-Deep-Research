
import pandas as pd
import pickle

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

df = pd.read_csv(
    f"{PROJECT_PATH}/data/metadata/paper_metadata.csv"
)

lookup = {}

for _, row in df.iterrows():

    arxiv_id = row["arxiv_id"]

    lookup[arxiv_id] = {
        "title": row["title"],
        "summary": row["summary"],
        "published": row["published"]
    }

with open(
    f"{PROJECT_PATH}/data/metadata/paper_lookup.pkl",
    "wb"
) as f:

    pickle.dump(
        lookup,
        f
    )

print("Metadata cache saved.")
print("Papers:", len(lookup))
