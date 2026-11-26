# Interview Guide — Finlytic AI

## 30-second pitch

"Finlytic AI is an applied-AI project: a tool-calling agent over a user's finance data. The model chooses from eight validated tools — SQL/pandas analytics, an ML forecast, anomaly detection and a hybrid BM25-plus-embeddings RAG search with citations — and every number comes from a tool, not the model. I built a labelled retrieval eval, so I can show hybrid fusion lifting MRR from 0.80 to 0.90, and I know exactly which questions still fail and why. It degrades gracefully offline."

## 2-minute walkthrough

1. **Login → demo data.** Register; each user gets their own SQLite file, seeded once.
2. **Dashboard.** KPIs, cash flow, forecast. Open "ML model performance": show the baseline comparison.
3. **Budget.** Pick a month; explain budgets are monthly limits.
4. **Anomaly Detection.** Show a flagged row and its reason text.
5. **Finance Chat (agent mode).** Ask "Compare my spending last month with this month" → expand **Tool calls** to show the chosen tool, validated args and latency. Then "How do I build an emergency fund?" → cited RAG answer.
6. **Evals.** Run `python evals/run_rag_eval.py` and walk through the table and the missed questions.

## Decisions you should be able to defend

| Question | Answer |
|---|---|
| Why per-user SQLite? | Simple, strong logical isolation, trivial to inspect. A `ContextVar` selects the user's DB. I'd move to Postgres with row-level user ids for real scale. |
| Why not let the LLM answer numeric questions? | LLMs hallucinate arithmetic. SQL/pandas produce the numbers; the LLM only explains reference material. |
| Why a TF-IDF fallback? | Embedding model download can fail or be unavailable offline; the app should degrade, not break. Thresholds differ per mode because score distributions differ. |
| How do you know the forecast is any good? | Chronological split (no leakage), lag features shifted by one day, and comparison to a naive 7-day-average baseline. The UI warns if it doesn't beat it. |
| Why Isolation Forest? | No labels exist. It handles mixed features and is fast. Limit: it flags rare-but-legitimate items (rent) — hence reason strings and "signal, not proof". |
| How are passwords stored? | PBKDF2-HMAC-SHA256, 600k iterations, random salt, versioned format, constant-time compare, auto-upgrade of old hashes. |
| How is password reset protected? | 6-digit code, 10-minute expiry, hashed at rest, 5 attempts then invalidated. Honest gap: no email delivery. |

## AI-engineering questions you will get

| Question | Answer |
|---|---|
| Why tool calling instead of stuffing data into the prompt? | Exact numbers, small context, auditable calls, and the model can't do arithmetic reliably. The trace makes every answer inspectable. |
| How do you stop the model misusing tools? | Schema validation before execution (types, ranges, month format, no extra args), unknown tools rejected, max 4 steps then a forced final answer, bounded outputs, errors returned to the model so it can recover. Tests cover each. |
| Prompt injection? | Tool results are labelled data in the system prompt, tools are read-only and scoped to the signed-in user, arguments are validated. Gap: no dedicated injection test suite yet. |
| Why hybrid retrieval? | Lexical and dense rankers fail differently. RRF needs no score calibration. Eval: hybrid 0.90 MRR vs 0.80 for each lexical ranker alone. |
| Why gate by absolute score and not just top-k? | Top-k always returns something; off-topic questions must return nothing (6/6 rejected in eval). |
| Weaknesses of your eval? | 36 hand-written questions, one author, a 6-section corpus, retrieval only — no answer-faithfulness metric yet. MiniLM numbers not yet reported. |
| How would you test the LLM loop without the API? | Injected client; scripted fake model that issues a valid call, an invalid call and then answers (see `tests/test_agent.py`). |
| What would you add for production? | Tracing and token/cost logging, answer-faithfulness evals, a FastAPI service, rate limits, caching, Postgres. |

## Bugs you found and fixed (great STAR stories)

1. **Budget bug:** monthly limits were compared to lifetime spend, so everything looked over budget. Found by running an end-to-end test over every page and noticing a permanent error banner. Fixed with a month selector.
2. **Hidden threshold:** the UI re-filtered RAG results at 0.40 by regex-parsing a formatted string, silently disabling the TF-IDF fallback. Fixed by returning structured results and per-mode thresholds.
3. **README drift:** docs described features that didn't exist. Rewrote to match the code. (Lesson: docs are part of the product.)
4. **Security hardening:** raised hash cost, hashed reset codes, attempt limits, HTML-escaping.

## Likely questions

- **How would you scale it?** Postgres, background jobs for model training, cache forecasts, proper auth provider, email service.
- **What are the privacy implications of live mode?** Only an aggregate summary and retrieved text go to the API, never raw transactions.
- **What would you refactor next?** Split `app.py` into one module per page; the classic chat intent logic exists in both `app.py` and `finance_chat.py` and could be removed now that agent mode is the default.

## Be upfront about

Offline mode uses a keyword planner and templates, not a real LLM; the live OpenAI loop is tested only against a fake client, not the real API; MiniLM retrieval metrics have not been run; the demo data is synthetic; the app has no investment-planning or CSV-import features.
