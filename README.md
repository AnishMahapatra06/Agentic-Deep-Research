# Agentic Deep Research

## Overview

This repository contains my submission for the Agentic Deep Research assignment.

The objective was to build a research assistant capable of answering questions from a collection of recent arXiv papers using a Retrieval-Augmented Generation (RAG) pipeline.

The system retrieves relevant scientific evidence from a research corpus and generates grounded answers with supporting paper citations.

---

## Methodology

### 1. Paper Collection

Research papers were collected from arXiv and stored locally for processing.

### 2. PDF Parsing

PDFs were parsed using PyMuPDF to extract page-level textual content.

### 3. Chunking

Documents were split into overlapping chunks to improve retrieval quality.

- Chunk Size: 1200 characters
- Overlap: 200 characters

### 4. Hybrid Retrieval

A hybrid retrieval pipeline was implemented using:

- Dense Retrieval with FAISS and BAAI/bge-small-en-v1.5 embeddings
- Sparse Retrieval with BM25
- Retrieval fusion for candidate selection

### 5. Reranking

Retrieved candidates were reranked using a cross-encoder-based reranking stage to improve relevance.

### 6. Answer Generation

The final answer generation stage uses retrieved evidence to produce grounded responses along with associated paper citations.

---

## Repository Structure

```text
Agentic_Deep_Research/

├── README.md
├── requirements.txt

├── evaluation/
│   └── questions.jsonl

├── predictions/
│   └── final_agent.jsonl

├── notebooks/
│   ├── collect_papers.py
│   ├── download_pdfs.py
│   ├── parse_pdfs.py
│   ├── chunk_papers.py
│   ├── build_metadata_cache.py
│   ├── build_bm25.py
│   ├── build_faiss.py
│   ├── build_faiss_gpu.py
│   ├── hybrid_retriever.py
│   ├── build_reranker.py
│   ├── reranked_retriever.py
│   ├── baseline_qa.py
│   ├── answer_question.py
│   └── pilot_answers.py

├── configs/
│   └── config.py

├── data/
├── index/
└── reports/
```

---

## Models and Libraries

### Embedding Model
- BAAI/bge-small-en-v1.5

### Retrieval Components
- FAISS
- BM25

### Core Libraries
- PyMuPDF
- SentenceTransformers
- Transformers
- PyTorch
- NumPy
- Pandas

---

## Pipeline

```text
ArXiv Papers
      │
      ▼
 PDF Downloading
      │
      ▼
 PDF Parsing
      │
      ▼
 Text Chunking
      │
      ▼
 Dense Retrieval (FAISS)
      │
      ├──────────┐
      │          │
      ▼          ▼
 Sparse Retrieval (BM25)
      │
      ▼
 Hybrid Retrieval
      │
      ▼
 Cross-Encoder Reranking
      │
      ▼
 Answer Generation
      │
      ▼
final_agent.jsonl
```

---

## Reproducing Results

Starting from a fresh clone:

```bash
pip install -r requirements.txt

python notebooks/collect_papers.py
python notebooks/download_pdfs.py
python notebooks/parse_pdfs.py
python notebooks/chunk_papers.py

python notebooks/build_metadata_cache.py
python notebooks/build_bm25.py
python notebooks/build_faiss.py

python notebooks/build_reranker.py

python notebooks/answer_question.py
```

The final predictions file is written to:

```text
predictions/final_agent.jsonl
```

---


## Final Output

The final submission file is available at:

```text
predictions/final_agent.jsonl
```

The evaluation questions are located at:

```text
evaluation/questions.jsonl
```

---

## Notes

Additional benchmark papers including AppWorld, OpenHands, UI-TARS, OSWorld, OS-MAP, Mem0, and A-MEM were incorporated during retrieval and answer refinement to improve coverage of benchmark-oriented questions.

---

## Author

Anish Mahapatra  
