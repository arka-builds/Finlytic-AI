"""Regression tests for database isolation, authentication, and RAG fallbacks."""

import pandas as pd

from src import auth, database
from src.llm_client import _mock_response, answer_with_llm
from src.rag_engine import load_markdown_documents


def test_user_databases_are_isolated(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    users_dir = data_dir / "users"
    monkeypatch.setattr(database, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(database, "USERS_DIR", str(users_dir))

    database.initialize_user_database(101, load_demo=False)
    database.initialize_user_database(202, load_demo=False)

    first = database.get_user_connection(101)
    second = database.get_user_connection(202)
    try:
        first.execute(
            "INSERT INTO transactions (date, type, category, description, amount, payment_method) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-01-01", "Expense", "Food", "lunch", 100, "Cash"),
        )
        first.commit()
        first_count = first.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        second_count = second.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    finally:
        first.close()
        second.close()

    assert first_count == 1
    assert second_count == 0


def test_auth_registration_login_and_password_reset(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(auth, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(auth, "AUTH_DATABASE_PATH", str(data_dir / "auth.db"))

    success, _, user_id = auth.register_user("Test User", "TEST@example.com", "secure-pass-123")
    assert success is True
    assert user_id is not None
    assert auth.authenticate_user("test@example.com", "secure-pass-123")["success"] is True
    assert auth.authenticate_user("test@example.com", "wrong-password")["success"] is False

    reset_ok, _, code = auth.create_password_reset_code("test@example.com")
    assert reset_ok is True
    changed, _ = auth.reset_password("test@example.com", code, "new-secure-pass-123")
    assert changed is True
    assert auth.authenticate_user("test@example.com", "new-secure-pass-123")["success"] is True


def test_mock_llm_accepts_legacy_summary_string():
    summary = "Total income: ₹50,000.00\nTotal expenses: ₹15,000.00\nSavings: ₹35,000.00"
    response = _mock_response(summary)
    assert "₹50,000.00" in response
    assert "₹15,000.00" in response
    assert "₹35,000.00" in response


def test_mock_llm_includes_retrieved_knowledge_context():
    df = pd.DataFrame(
        {"type": ["Income", "Expense"], "category": ["Salary", "Food"], "amount": [1000, 200]}
    )
    response = answer_with_llm(
        "How can I build an emergency fund?",
        df,
        retrieved_context="Keep a cash reserve for essential expenses.",
    )
    assert "essential expenses" in response


def test_knowledge_base_loader_returns_chunks():
    chunks = load_markdown_documents()
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


def test_live_llm_receives_retrieved_context(monkeypatch):
    import src.llm_client as client_module

    captured = {}

    class FakeResponses:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return type("Response", (), {"output_text": "A grounded test answer."})()

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(client_module, "MOCK_LLM", False)
    monkeypatch.setattr(client_module, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(client_module, "OPENAI_MODEL", "test-model")
    monkeypatch.setattr(client_module, "OpenAI", FakeOpenAI)

    df = pd.DataFrame(
        {"type": ["Income", "Expense"], "category": ["Salary", "Food"], "amount": [1000, 200]}
    )
    answer = client_module.answer_with_llm(
        "How do I create an emergency fund?",
        df,
        retrieved_context="Keep a reserve for essential expenses.",
    )

    assert answer == "A grounded test answer."
    assert "Keep a reserve for essential expenses." in captured["input"]
    assert captured["model"] == "test-model"


def test_expense_feature_preparation_drops_invalid_rows_and_sorts_dates():
    from src.ml_expense_prediction import prepare_dataset

    frame = pd.DataFrame(
        {
            "date": ["2026-01-09", "not-a-date", "2026-01-01"] + [
                f"2026-01-{day:02d}" for day in range(2, 9)
            ],
            "amount": [900, 50, 100] + [100 + day for day in range(2, 9)],
        }
    )
    prepared = prepare_dataset(frame)
    assert not prepared.empty
    assert prepared["date"].is_monotonic_increasing
    assert prepared["date"].notna().all()
    assert prepared["amount"].notna().all()


def test_mock_llm_includes_retrieved_context_for_advice_questions():
    df = pd.DataFrame(
        {"type": ["Income", "Expense"], "category": ["Salary", "Food"], "amount": [1000, 200]}
    )
    response = answer_with_llm(
        "Please give me financial advice.",
        df,
        retrieved_context="Build a cash reserve before taking additional risk.",
    )
    assert "Build a cash reserve" in response


def test_user_context_is_isolated_between_threads():
    from concurrent.futures import ThreadPoolExecutor

    database.set_current_user(303)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            other_context_user = pool.submit(database.get_current_user_id).result()
        assert database.get_current_user_id() == 303
        assert other_context_user is None
    finally:
        database.clear_current_user()
    assert database.get_current_user_id() is None


def test_demo_data_is_not_reseeded_after_user_deletes_all_records(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    users_dir = data_dir / "users"
    transactions_csv = tmp_path / "transactions.csv"
    budgets_csv = tmp_path / "budgets.csv"
    transactions_csv.write_text(
        "date,type,category,description,amount,payment_method\n"
        "2026-01-01,Expense,Food,Sample lunch,100,Cash\n",
        encoding="utf-8",
    )
    budgets_csv.write_text(
        "category,budget\nFood,5000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(database, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(database, "USERS_DIR", str(users_dir))
    monkeypatch.setattr(database, "TRANSACTIONS_CSV", str(transactions_csv))
    monkeypatch.setattr(database, "BUDGETS_CSV", str(budgets_csv))

    database.initialize_user_database(404, load_demo=True)
    connection = database.get_user_connection(404)
    try:
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 1
        connection.execute("DELETE FROM transactions")
        connection.commit()
    finally:
        connection.close()

    database.initialize_user_database(404, load_demo=True)
    connection = database.get_user_connection(404)
    try:
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    finally:
        connection.close()
