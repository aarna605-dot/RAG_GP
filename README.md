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

# R3: Router and Planner

Decides how much agent a question deserves and splits hard questions into sub-questions for R4's workers.
Nothing is trained: heuristics, short LLM calls, and one threshold `tau` calibrated on the cost-accuracy curve.

```
question --> Router --> SINGLE  --> one tuned-RAG pass (trivial plan, no extra LLM cost)
                    +-> AGENTIC --> Planner --> Plan (DAG of sub-questions) --> R4 workers --> R6 verifier
```

## Status
* Router and planner: run on a live LLM (Gemini) over `eval/r3_probe_questions.json`.
* Stop rules: not yet used inside a real loop (needs R4).
* Cost-accuracy curve: not yet run on real S2/S4 results (needs R4 + R7).

## Files
| Path | What |
|---|---|
| `src/router/router.py` | `Router`: `heuristic`, `llm` or `hybrid` (LLM only when the heuristic is unsure) |
| `src/router/planner.py` | `Planner`, plan validation and fallback, `route_and_plan()` |
| `src/router/stop_rules.py` | `StopRules`, `LoopState`, `should_stop()` |
| `src/router/features.py` | `PaperIndex` (corpus papers named in a question) and difficulty features |
| `src/router/schemas.py` | `Route`, `RouteDecision`, `SubQuestion`, `Plan`, `Usage` (all have `to_trace()`) |
| `src/router/llm.py` | `OpenAICompatibleClient`, `CachedLLM`, `MockLLM` (for tests) |
| `src/router/curve.py` | cost-accuracy curve, CV threshold selection, bootstrap CIs |
| `scripts/r3_route.py` | routes questions, writes decisions |
| `scripts/r3_plan.py` | routes and plans questions, writes plans |
| `eval/r3_probe_questions.json` | 18 hand-made probe questions (a sanity set, not the eval set) |
| `tests/test_router.py` | unit tests |

## Setup
Set these in the same terminal you run the scripts from (any OpenAI-compatible endpoint works):
```powershell
$env:LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
$env:LLM_API_KEY  = "<key>"        # never commit
$env:LLM_MODEL    = "<model id>"
```
`src/__init__.py` imports R2's retriever, so R2's dependencies must be installed.

## Run (from the repo root)
```powershell
python scripts/r3_route.py --question "Compare CRAG and TrustRAG on noisy context."
python scripts/r3_route.py                        # probe questions -> reports/r3/route_decisions.jsonl
python scripts/r3_plan.py                         # probe questions -> reports/r3/plans.json
python scripts/r3_route.py --method heuristic     # no LLM needed
pytest tests/test_router.py
```
Flags: `--questions file.json`, `--question "..."`, `--method`, `--tau`, `--max-sub`, `--out`, `--no-cache`.
Replies are cached in `.cache/r3_llm_cache.jsonl`, so reruns are free and identical; a changed prompt
causes fresh calls automatically.

`plans.json` entries: `{id, question, route, difficulty, sub_questions:[{id, text, depends_on, paper_hints}],
waves, fallback, notes, cost}`. R4 can use it as fixtures.

## Using it from code
```python
from src.router import PaperIndex, Router, Planner, route_and_plan, StopRules, LoopState, should_stop

index   = PaperIndex.from_csv("papers.csv")
router  = Router(llm=llm, index=index, method="hybrid", tau=0.5)
planner = Planner(llm, index, max_subquestions=4)

decision, plan = route_and_plan(question, router, planner)
answers = {}
for wave in plan.waves():                 # one wave = parallel
    for sub in wave:
        answers[sub.id] = run_worker(sub.resolve(answers))   # fills {qN} placeholders (R4's worker)
```
Add `decision.usage` and `plan.usage` to the answer's cost. `llm` is any object with
`complete(system, user, max_tokens) -> LLMResponse`.

Loop control:
```python
rules, state = StopRules(), LoopState()
# after each step:
state.record_step(chunk_ids, tokens, sufficient=reflection_says_enough)
d = should_stop(rules, state, step_cap=rules.max_steps_per_worker)   # d.reason: budget|max_steps|sufficient|saturation
```

## Design decisions
* One difficulty score and one threshold: `AGENTIC if difficulty >= tau`.
* Hybrid: the LLM is asked only when the heuristic score is in 0.3 to 0.7; router overhead counts in the curve.
* No-retrieval route exists but is off by default (all questions concern specific papers).
* `tau` is chosen by k-fold CV with bootstrap CIs, not on one split.
* The planner never crashes the pipeline: bad output retries once, then falls back to one sub-question.
* Weights, the band and the planner prompt are tuned on the probe set only; do not tune on final eval questions.

## Known limitations
* **HotpotQA / MuSiQue questions are untested.** The strongest feature (papers named) is zero for them, and several
  two-hop examples scored <= 0.35 and went SINGLE, some outside the LLM band. Test on about 15 to 20 labelled
  questions before applying the router to those 120 questions; if needed, widen the band or always ask the LLM
  when no paper is named.
* Stop rules: `max_steps_per_worker` only applies if `step_cap=` is passed; the budget can overshoot by one step;
  `verifier_retries` is never incremented here; `token_budget=30000` is a placeholder until R7 fixes it.
* Planner: a comparison may still be planned as one sub-question; `{qN}` is replaced by the whole earlier answer
  (use a short form); sub-questions use short names, so use `paper_hints` for full titles in dense search.

## Open points
* R4: one shared LLM client (the loop needs tool calling); who sets `sufficient`; who synthesizes the final answer.
  S4 router-off = always plan; S3 = no planner, `max_steps=8`.
* R7: results file format for the curve, the real token budget, whether the router covers all 200 questions.
* R7 / R8: answer-object schema that `to_trace()` and `Usage.to_dict()` should map onto.

## Cost-accuracy curve
```powershell
python -m src.router.curve --questions eval/<questions>.json --results results/results.jsonl `
    --cheap S2 --strong S4 --metric judge --cost tokens --method hybrid --out reports/r3
```
`results.jsonl`: one row per question per setup: `{"qid","setup","<metric>","tokens","usd","latency_s"}`.
Target: cost per query down 30% or more with under 2 points of accuracy lost.
