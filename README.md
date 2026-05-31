# Agentic Deep Research

## Overview

This repository contains an Agentic Deep Research system developed for answering research-oriented questions over a large corpus of arXiv papers.

The system combines:

- Hybrid Retrieval (FAISS + BM25)
- Cross-Encoder Re-ranking
- Gemini 2.5 Flash Answer Generation

to generate evidence-grounded answers with supporting paper citations.

---

## Architecture

```text
Question
   │
   ▼
Hybrid Retrieval
(FAISS + BM25)
   │
   ▼
Cross-Encoder Re-ranking
   │
   ▼
Gemini 2.5 Flash
   │
   ▼
Answer + Citations
```

---

## Repository Structure

```text
Agentic-Deep-Research/
│
├── predictions.jsonl
├── predictions/
├── notebooks/
├── configs/
├── evaluation/
└── report/
```

---

## Models Used

| Component | Model |
|------------|------------|
| Embeddings | BAAI/bge-small-en-v1.5 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Gemini 2.5 Flash |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
python collect_papers.py
python download_pdfs.py
python parse_pdfs.py
python chunk_papers.py
python build_faiss.py
python build_bm25.py
python build_reranker.py
python pilot_answers.py
```

---

## Output Format

```json
{
  "id": "q01",
  "answer": "...",
  "cited_papers": ["2605.30102"]
}
```

---

## Deliverables

- `predictions.jsonl` — final submission file
- `predictions/full_agent.jsonl` — full-agent configuration
- `report/technical_report.pdf` — technical report

---

## Author

**Anish Mahapatra**  
