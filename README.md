# RAG_GP: Agentic RAG for AI Research Papers

An agentic research assistant over **300 recent arXiv papers** on agents, retrieval, and reasoning. Built for the Neural AI RSS Panel project.

---

## Architecture & Team Division

The system tests the trade-offs of **Cost**, **Trust (Citation Grounding)**, and **Safety (Prompt Injection Resistance)** across 6 experimental setups (S0–S5):

- **Pair A (Knowledge)**:
  - **R1 (Corpus)**: Paper fetching, section-aware chunking, Anthropic contextual headers (`corpus_v0`, 6,079 chunks). *(Delivered by Aarna)*
  - **R2 (Retrieval Engine)**: Dual sparse/dense indexing, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, S1/S2 baselines, and Recall@10 benchmarking. *(Delivered by R2)*
- **Pair B (Agent)**:
  - **R3 (Router & Planner)**: Complexity routing and sub-question decomposition.
  - **R4 (Workers & Context)**: Tool-calling loops, parallel retrieval, and context budget management.
- **Pair C (Tools & Trust)**:
  - **R5 (MCP Tool Server)**: Standardization of search, read section, and keyword grep.
  - **R6 (Verification)**: Claim-level citation checking and prompt injection filtering.
- **Pair D (Eval & Product)**:
  - **R7 (Evaluation Harness)**: Automated benchmark across S0–S5 with LLM judge.
  - **R8 (Product & UI)**: FastAPI backend, Streamlit UI, live traces, and cost tracking.

---

## Role R2 (Retrieval Engine) Overview

Role R2 provides the retrieval backbone consumed by R4 (Workers), R5 (MCP Server), and R7 (Evaluation).

### Key Components Built
1. **Sparse Lexical Index (`bm25s`)**:
   - Tokenizes Anthropic contextual headers + chunk body.
   - Ultra-low latency retrieval (**8.1 ms**) for exact keyword matching, paper acronyms (*CRAG*, *TrustRAG*), and author names.
   - Persisted in `indices/bm25/`.
2. **Dense Vector Store (`LanceDB` + `BAAI/bge-small-en-v1.5`)**:
   - Embeds 6,079 chunks using contextual headers (`{title} > {section} \n {text}`).
   - Fast cosine vector search indexed in serverless LanceDB table at `indices/lancedb/`.
3. **Hybrid Reciprocal Rank Fusion (RRF)**:
   - Fuses top candidates from BM25 and Dense search:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. **Cross-Encoder Reranker (`BAAI/bge-reranker-base`)**:
   - Re-scores top-25 hybrid candidates via full cross-attention for high precision.
5. **Standardized Search API (`src/retriever.py`)**:
   - Unified interface with pluggable modes: `dense`, `bm25`, `hybrid`, and `reranked`.
6. **Benchmark Baselines (`src/baselines.py`)**:
   - **S1 (Vanilla RAG)**: Single-pass dense top-5 retrieval.
   - **S2 (Tuned RAG)**: Hybrid BM25 + dense search + cross-encoder reranker + context headers.

---

## Retrieval Benchmark (Recall@10 Report)

Benchmarked on multi-hop and single-hop research questions across the 300-paper corpus:

| Strategy | Recall@5 | Recall@10 | MRR | Avg Latency (ms) | Primary Advantage |
|:---|:---:|:---:|:---:|:---:|:---|
| **BM25 (Sparse)** | **87.5%** | **87.5%** | 0.7292 | **8.1 ms** | Catches exact paper acronyms & author queries |
| **Dense (S1 Vanilla)** | **87.5%** | **87.5%** | **0.8750** | 801.4 ms | Matches semantic and paraphrased intent |
| **Hybrid RRF** | **87.5%** | **87.5%** | **0.8750** | 169.6 ms | Optimal speed-accuracy trade-off for worker agents |
| **Reranked (S2 Tuned)**| **87.5%** | **87.5%** | 0.8125 | High Precision | Filters irrelevant context before generation |

*Full report available at [`reports/recall_at_10_report.md`](reports/recall_at_10_report.md).*

---

## Project Structure

```
.
├── corpus_v0/                      # Ingested chunks from R1 (Aarna)
│   ├── chunks.jsonl                # 6,079 contextualized chunks
│   └── manifest.json               # Corpus metadata (v0)
├── eval/
│   └── r2_eval_questions.json      # 10 custom evaluation Qs with gold evidence
├── indices/                        # Pre-built persistent indices
│   ├── bm25/                       # BM25 sparse index
│   └── lancedb/                    # LanceDB dense vector table
├── reports/
│   ├── recall_at_10_report.md      # Official R2 retrieval benchmark report
│   └── paper_presentation_notes.md # Notes for tagged paper (ACL 2026)
├── src/
│   ├── __init__.py
│   ├── baselines.py                # S1 (Vanilla RAG) and S2 (Tuned RAG)
│   ├── build_bm25.py               # Sparse index builder
│   ├── build_dense.py              # Dense vector builder
│   ├── evaluate_retrieval.py       # Benchmark evaluation script
│   └── retriever.py                # Core PaperRetriever class & Search API
├── tests/
│   └── test_search.py              # Retrieval test across all 4 modes
├── papers.csv                      # Metadata for all 300 arXiv papers
└── README.md
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install lancedb bm25s sentence-transformers torch pandas
```

### 2. Run Retrieval Tests
```bash
python tests/test_search.py
```

### 3. Use the Search API in Python (for R4, R5, R8)
```python
from src.retriever import PaperRetriever

retriever = PaperRetriever()

# Fast hybrid search for exploratory worker agents
results = retriever.search("What is Corrective RAG?", top_k=5, mode="hybrid")

# High-precision reranked search for final answer generation
results = retriever.search("How does Adaptive-RAG route queries?", top_k=5, mode="reranked")

for res in results:
    print(res["chunk_id"], res["header"], res["score"])
```

### 4. Run Retrieval Benchmark
```bash
python src/evaluate_retrieval.py
```

---

## Role R2 Deliverables Checklist
- [x] Sparse BM25 index built with `bm25s`
- [x] Dense vector index built with LanceDB + `BAAI/bge-small-en-v1.5`
- [x] Hybrid RRF fusion + Cross-Encoder reranking (`BAAI/bge-reranker-base`)
- [x] Unified `PaperRetriever` Search API for R4, R5, R8
- [x] S1 and S2 baseline implementations in `src/baselines.py`
- [x] Quantitative `Recall@10` report in `reports/recall_at_10_report.md`
- [x] 10 custom evaluation questions in `eval/r2_eval_questions.json`
- [x] Tagged paper presentation notes in `reports/paper_presentation_notes.md`
