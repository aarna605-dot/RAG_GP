# Ask the Papers: Agentic RAG web app (R8 Product)

A single-page chat app served by the FastAPI backend: streaming chat, live agent console,
clickable citations, cost per answer, history, feedback, and a results view for the evaluation table.
Runs on a demo agent today; real agents plug in through `app/agents.py`.

## Run and test
This app lives in the `r8_app/` folder of the merged repo (R1 to R6 code sits next to it, untouched).

    cd r8_app
    pip install -r requirements-rag.txt     # once; large (torch). First S1/S2/S3/S4 question downloads the models
    cp .env.example .env                    # then open .env and paste your keys (never commit it)
    bash run.sh local                       # app at http://localhost:8000
    bash run.sh test                        # unit tests
    # API docs: http://localhost:8000/docs

`run.sh` reads `.env` for you. Docker (`bash run.sh docker`) builds from the repo root; it is untested.

## How each setup is wired (all teammates' code is used as-is)
| Setup | Built from | Answer step |
|---|---|---|
| S1 Vanilla RAG | R2 `s1_vanilla_retrieval` (dense top-5) | this app |
| S2 Tuned RAG | R2 `s2_tuned_retrieval` (hybrid + reranker) | this app |
| S3 Single agent | R4 `RetrievalWorker`, search/reflect loop, max 8 steps (hybrid search) | this app |
| S4 Multi-agent | R3 `Router` then `Planner` (hard questions only), R4 workers in parallel waves, R6 injection filter, answer, R6 `ClaimVerifier` with one repair round | this app |
| S5 Index-free | R5 `search` / `keyword_grep` / `read`, driven by the LLM in a loop, no embeddings | this app |
| S0 Closed book | R7 (not here yet) | |

R2 to R6 only retrieve, plan or check; none of them writes the final answer, so this app does, with one
prompt and one model for every setup (temperature 0). Router, planner, worker and verifier LLM calls are
all counted in the cost shown for the answer. Easy questions routed `single` in S4 take one reranked search and no planner.

## LLM, keys and tracing
Everything reads the same variables: `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` (Gemini's OpenAI-compatible
endpoint is the default in `.env.example`). S3, S4 and S5 need them; S1/S2 fall back to showing the best passages.
Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) to send each question to LangSmith as one run
with a child run per step. Export is best effort and never slows or breaks an answer.

## Results view
The evaluation harness (R7) writes `runs/results.json`, a JSON list with one row per setup. Known keys:
`setup, em, f1, judge_acc, recall10, cite_precision, cite_recall, unsupported_rate, latency_s, tokens, usd`.
Ratios between 0 and 1 are shown as percentages. The Results tab fills in on refresh.

## Endpoints
- `POST /ask` blocking, returns the `AnswerObject` (eval harness); 502 if an agent fails
- `POST /ask/stream` server-sent events (web app): `step` events, then `final` (or `error`)
- `GET /api/setups`, `/api/history`, `/api/runs/{id}`, `/api/stats`, `/api/results`; `POST /api/feedback`
- `GET /health`

## Data and tracing
Runs go to `runs/runs.jsonl`, feedback to `runs/feedback.jsonl`.
