
import arxiv
import pandas as pd

SEARCH_TERMS = [
    "LLM agent",
    "agentic RAG",
    "agent memory",
    "tool use",
    "multi-agent systems",
    "computer use agent",
    "GUI agent",
    "web agent",
    "software engineering agent",
    "agent benchmark",
    "reflection",
    "ReAct",
    "Self-RAG",
    "deep research"
]

client = arxiv.Client()

papers = []

for term in SEARCH_TERMS:

    print(f"Searching: {term}")

    search = arxiv.Search(
        query=term,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    try:

        for result in client.results(search):

            year = result.published.year

            if year < 2024:
                continue

            papers.append({
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "summary": result.summary,
                "published": str(result.published.date()),
                "pdf_url": result.pdf_url
            })

    except Exception as e:
        print(e)

df = pd.DataFrame(papers)

df.drop_duplicates(
    subset=["arxiv_id"],
    inplace=True
)

print("Unique papers:", len(df))

df.to_csv(
    "/content/drive/MyDrive/Agentic_Deep_Research/data/metadata/paper_metadata.csv",
    index=False
)

print("Metadata saved successfully.")
