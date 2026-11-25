# Interview Guide — Finlytic AI

## 30-second pitch

"Finlytic AI is a multi-user personal finance app I built in Python and Streamlit. It tracks transactions and monthly budgets, forecasts next-day spending with a Random Forest, flags unusual expenses with an Isolation Forest and explains why, and has a chat assistant that answers questions about your numbers deterministically and uses RAG over a finance knowledge base for educational questions. It works fully offline, and I kept the LLM away from the arithmetic on purpose so it can't invent numbers."

## 2-minute walkthrough

1. **Login → demo data.** Register; each user gets their own SQLite file, seeded once.
2. **Dashboard.** KPIs, cash flow, forecast. Open "ML model performance": show the baseline comparison.
3. **Budget.** Pick a month; explain budgets are monthly limits.
4. **Anomaly Detection.** Show a flagged row and its reason text.
5. **Finance Chat.** Ask "How much did I spend?" (SQL answer) then "How big should my emergency fund be?" (RAG + mock LLM; label shows mock mode).

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

## Bugs you found and fixed (great STAR stories)

1. **Budget bug:** monthly limits were compared to lifetime spend, so everything looked over budget. Found by running an end-to-end test over every page and noticing a permanent error banner. Fixed with a month selector.
2. **Hidden threshold:** the UI re-filtered RAG results at 0.40 by regex-parsing a formatted string, silently disabling the TF-IDF fallback. Fixed by returning structured results and per-mode thresholds.
3. **README drift:** docs described features that didn't exist. Rewrote to match the code. (Lesson: docs are part of the product.)
4. **Security hardening:** raised hash cost, hashed reset codes, attempt limits, HTML-escaping.

## Likely questions

- **How would you evaluate the RAG?** Build a small question → expected-section set, measure hit@k and MRR for semantic vs TF-IDF, then tune thresholds. (Not done yet — say so.)
- **How would you add citations?** `retrieve_chunks` already returns source and section; pass them to the prompt and require the model to cite them.
- **How would you scale it?** Postgres, background jobs for model training, cache forecasts, proper auth provider, email service.
- **What are the privacy implications of live mode?** Only an aggregate summary and retrieved text go to the API, never raw transactions.
- **What would you refactor next?** Split `app.py` into one module per page; the chat intent logic exists in both `app.py` and `finance_chat.py` and should be unified.

## Be upfront about

Mock mode isn't a real LLM; the demo data is synthetic; no live-API run was tested here; the app has no investment-planning or CSV-import features.
