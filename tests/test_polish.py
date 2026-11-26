"""Tests for hardening and correctness fixes."""
from datetime import date

import pandas as pd

from src import auth, database, rag_engine
from src.llm_client import _mock_response


def _auth_env(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(auth, "AUTH_DATABASE_PATH", str(tmp_path / "auth.db"))


def test_new_hashes_are_versioned_and_verify():
    stored = auth.hash_password("s3cret-pass")
    assert stored.startswith("pbkdf2_sha256$600000$")
    assert auth.verify_password("s3cret-pass", stored)
    assert not auth.verify_password("wrong", stored)
    assert not auth.needs_rehash(stored)


def test_legacy_hash_still_works_and_is_upgraded_on_login(tmp_path, monkeypatch):
    import hashlib, secrets
    _auth_env(tmp_path, monkeypatch)
    ok, _, uid = auth.register_user("Old User", "old@example.com", "legacy-pass-1")
    salt = secrets.token_bytes(16)
    legacy = salt.hex() + ":" + hashlib.pbkdf2_hmac("sha256", b"legacy-pass-1", salt, 100_000).hex()
    conn = auth.get_auth_connection()
    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (legacy, uid)); conn.commit(); conn.close()
    assert auth.authenticate_user("old@example.com", "legacy-pass-1")["success"]
    conn = auth.get_auth_connection()
    upgraded = conn.execute("SELECT password_hash FROM users WHERE id=?", (uid,)).fetchone()[0]
    conn.close()
    assert upgraded.startswith("pbkdf2_sha256$")


def test_reset_code_is_not_stored_in_plaintext_and_locks_after_failures(tmp_path, monkeypatch):
    _auth_env(tmp_path, monkeypatch)
    auth.register_user("U", "u@example.com", "password-123")
    _, _, code = auth.create_password_reset_code("u@example.com")
    conn = auth.get_auth_connection()
    stored = conn.execute("SELECT reset_code FROM users").fetchone()[0]
    conn.close()
    assert stored != code
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(auth.MAX_RESET_ATTEMPTS):
        assert auth.reset_password("u@example.com", wrong, "new-password-1")[0] is False
    ok, message = auth.reset_password("u@example.com", code, "new-password-1")
    assert not ok and "Too many" in message


def test_demo_dates_are_rebased_to_today():
    rows = [("2026-01-01", "Expense", "Food", "a", 1.0, "UPI"),
            ("2026-01-11", "Expense", "Food", "b", 2.0, "UPI")]
    out = database._rebase_demo_dates(rows, today=date(2026, 11, 25))
    assert [r[0] for r in out] == ["2026-11-15", "2026-11-25"]


def test_mock_responses_contain_no_literal_backslash_n():
    reply = _mock_response("needs vs wants", pd.DataFrame(), "needs and wants are different")
    assert "\\n" not in reply


def test_retrieval_thresholds_are_per_mode_and_context_is_formatted():
    assert rag_engine.MIN_RELEVANCE_SCORES["tfidf_fallback"] < rag_engine.MIN_RELEVANCE_SCORES["semantic"]
    assert rag_engine.format_context([]) == rag_engine.NO_CONTEXT_MESSAGE
    text = rag_engine.format_context([{"section": "S", "source": "f.md", "score": 0.5, "text": "body"}])
    assert "### [1] S" in text and "body" in text


def test_forecast_reports_naive_baseline(tmp_path, monkeypatch):
    from src import ml_expense_prediction as ml
    days = pd.date_range("2026-01-01", periods=60)
    raw = pd.DataFrame({"date": days.astype(str), "amount": [100 + (i % 7) * 20 for i in range(60)]})
    monkeypatch.setattr(ml, "get_expense_data", lambda: raw)
    result = ml.get_prediction()
    assert result["baseline_mae"] >= 0
    assert result["beats_baseline"] == (result["mae"] < result["baseline_mae"])
