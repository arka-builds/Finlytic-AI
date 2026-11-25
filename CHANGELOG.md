# Changelog

## 2026-11-25 — Portfolio polish

### Fixed
- **Budget page compared monthly limits with lifetime spending.** It now has a month selector and uses that month's spending.
- Mock LLM replies for needs-vs-wants and savings rate showed literal `\n` sequences.
- Chat applied a hidden 0.40 relevance filter by regex-parsing a formatted string, which starved the TF-IDF fallback. Retrieval now returns structured results with per-mode thresholds.
- Logged-in name/email are HTML-escaped in sidebar markup.
- Branding unified to "Finlytic AI".

### Security
- PBKDF2 raised to 600,000 iterations with a versioned format; old hashes verify and upgrade on login.
- Password-reset codes are hashed, with a 5-attempt limit; on-screen display is configurable.

### Added
- Naive-baseline comparison for the expense forecast.
- Regenerated, seeded demo dataset; demo dates re-based to today on seeding.
- `FINLYTIC_DATA_DIR`, Dockerfile, GitHub Actions CI, MIT license, screenshots, interview guide.
- End-to-end Streamlit smoke test and tests for the above (38 total).

### Changed
- CSS moved to `assets/styles.css`; data access moved to `src/repository.py`.
- Dark theme set in `.streamlit/config.toml`; deprecated `use_container_width` replaced (Streamlit ≥ 1.50).
- README rewritten to describe only features that exist.

### Removed
- `app_final_polished.py` (stale UI copy), the broken terminal CLI and its helpers, legacy `finance.db` code, scratch scripts, committed caches, `AUDIT_REPORT.md` (folded into this log).

## Earlier
Reliability and audit work: user-scoped databases, budget CRUD, daily forecast aggregation, multi-feature anomaly detection, TF-IDF retrieval fallback, mock-mode disclosure.
