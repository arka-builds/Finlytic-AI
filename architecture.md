# Architecture

## Request flow

1. `app.py` authenticates the user (`src/auth.py`), then calls `set_current_user(user_id)`. A `ContextVar` holds the active user so every data-access call resolves to that user's SQLite file — modules never take a user id in SQL helpers, which makes cross-user leakage hard to write by accident.
2. Pages read through `src/repository.py` (and write through it or `budget_manager.py`).
3. ML modules read expense rows from the same per-user database.
4. Finance Chat (agent mode) runs `agent.answer`: with a key it drives an OpenAI function-calling loop, otherwise a deterministic planner. Both call the same validated tools, so numbers always come from code. Classic mode (toggle off) uses keyword intents + RAG.

## Data isolation

```
data/auth.db               users + password hashes
data/users/user_<id>.db    transactions, budgets, app_meta (one per user)
```
Set `FINLYTIC_DATA_DIR` to relocate these (e.g. a Docker volume). Seed CSVs stay in `data/`.

Demo seeding runs once per new database (marker row in `app_meta`) and re-bases dates so the newest record is today; it never touches user-entered data.

## RAG

- **Ingest:** Markdown files in `data/knowledge_base/` → section-aware chunks (≤180 words, 30-word overlap, headings kept).
- **Embed:** `all-MiniLM-L6-v2` (384-d, normalised) → dot product = cosine similarity.
- **Fallback:** TF-IDF (1–2-grams) if the model can't load. Thresholds differ per mode (semantic 0.40, TF-IDF 0.15) because the score distributions differ.
- **Generate:** filtered chunks + a numeric financial summary are passed to the LLM, which is told to treat them as reference material, not instructions. Mock mode answers deterministically and says so.

## Agent

`ToolContext` injects data sources (transactions, budgets, forecast, anomalies, search), so tools are unit-testable without a database or model. `execute()` validates arguments against each tool's JSON schema, times the call and converts failures into error results the model can read. `run_openai` allows at most 4 tool rounds, then forces a final answer. Offline, `plan_offline` maps keywords and relative dates ("last month") to tool calls and `compose_offline` renders results with templates and citations.

## Retrieval details

Rankers: dense cosine (MiniLM, or TF-IDF offline) and BM25 (own implementation, k1 1.5, b 0.75, score normalised by the ideal query score). Hybrid = reciprocal rank fusion (k = 60). A chunk is returned only if some ranker clears its absolute threshold (dense 0.40, TF-IDF 0.15, BM25 0.30 normalised). Evaluation lives in `evals/` and `docs/RAG_EVAL.md`.

## ML

- **Forecast:** transactions → daily totals on a continuous calendar → lag/rolling features (leakage-safe `shift`) → RandomForest; chronological 80/20 split; MAE/RMSE/R² plus a naive 7-day-average baseline. Output is the next calendar day's total.
- **Anomalies:** Isolation Forest on log-amount, amount÷category-median, calendar features and one-hot category/payment method; contamination 5 %; reason string cites the multiple of the category median when ≥ 2×.

## Trade-offs

| Choice | Why | Cost |
|---|---|---|
| SQLite per user | Zero-ops isolation, easy to inspect | No concurrency at scale |
| Rule-based chat routing | Deterministic, testable, no hallucinated numbers | Brittle to phrasing |
| Local embeddings | No data leaves the machine, no API cost | Model download, CPU latency |
| Streamlit | Fast to build and demo | Whole-script reruns; large single file |
