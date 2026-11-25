"""Regression tests for truthful budget reporting, daily forecasting and RAG fallback."""
from pathlib import Path

import pandas as pd


def test_daily_forecast_dataset_aggregates_transactions_and_fills_zero_days():
    from src.ml_expense_prediction import prepare_dataset

    raw = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18"],
        "amount": [100, 50, 200, 0, 100, 120, 80, 90, 110, 100, 120, 80, 90, 110, 100, 120, 80, 90],
    })
    prepared = prepare_dataset(raw)
    # Jan 1 is aggregated (150), Jan 2 is retained as a zero-spend day.
    assert prepared.loc[prepared["date"] == pd.Timestamp("2026-01-08"), "amount"].iloc[0] == 90
    assert prepared["date"].is_monotonic_increasing
    assert prepared["amount"].ge(0).all()
    assert len(prepared) == 11  # 18 calendar days minus seven warm-up days


def test_anomaly_detection_adds_reason_and_category_relative_features(monkeypatch):
    from src import anomaly_detection

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    amounts = [100 + (i % 4) * 5 for i in range(29)] + [5000]
    frame = pd.DataFrame({
        "id": range(1, 31), "date": dates,
        "category": ["Food"] * 30,
        "description": ["test"] * 30,
        "amount": amounts,
        "payment_method": ["Cash"] * 30,
    })
    monkeypatch.setattr(anomaly_detection, "get_expense_data", lambda: frame)
    result = anomaly_detection.detect_anomalies()
    assert result is not None
    assert "anomaly_reason" in result.columns
    assert "category_amount_ratio" in result.columns
    flagged = result[result["anomaly_prediction"] == -1]
    assert not flagged.empty
    assert flagged["anomaly_reason"].str.len().gt(0).all()


def test_rag_uses_tfidf_fallback_when_semantic_model_is_unavailable(tmp_path, monkeypatch):
    from src import rag_engine

    (tmp_path / "finance.md").write_text(
        "# Emergency Fund\nAn emergency fund is cash reserved for essential unexpected expenses.\n\n"
        "# Budgeting\nTrack income and expenses and set realistic category limits.",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_engine, "KNOWLEDGE_BASE_DIR", tmp_path)
    monkeypatch.setattr(rag_engine, "get_embedding_model", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    retriever = rag_engine.LocalRAGRetriever()
    assert retriever.retrieval_mode == "tfidf_fallback"
    results = retriever.retrieve("cash reserve for unexpected expenses")
    assert results
    assert results[0]["section"] == "Emergency Fund"


def test_mock_mode_is_explicit_in_chat_ui_source():
    app_source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert "RULE-BASED + LOCAL MOCK RESPONSES" in app_source
    assert "LIVE LLM CONFIGURED" in app_source


def test_budget_create_update_delete_is_persisted_per_user(tmp_path, monkeypatch):
    from src import database
    from src.budget_manager import save_budget, delete_budget

    data_dir = tmp_path / "data"
    monkeypatch.setattr(database, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(database, "USERS_DIR", str(data_dir / "users"))
    database.initialize_user_database(909, load_demo=False)
    database.set_current_user(909)
    try:
        save_budget("Travel", 4000)
        save_budget("Travel", 6500)
        connection = database.get_connection()
        try:
            row = connection.execute(
                "SELECT budget FROM budgets WHERE category = ?", ("Travel",)
            ).fetchone()
        finally:
            connection.close()
        assert row[0] == 6500
        assert delete_budget("Travel") is True
        assert delete_budget("Travel") is False
    finally:
        database.clear_current_user()


def test_budget_rejects_non_positive_amount():
    from src.budget_manager import save_budget
    import pytest
    with pytest.raises(ValueError):
        save_budget("Food", 0)


def test_budget_records_are_isolated_between_users(tmp_path, monkeypatch):
    from src import database
    from src.budget_manager import save_budget

    data_dir = tmp_path / "data"
    monkeypatch.setattr(database, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(database, "USERS_DIR", str(data_dir / "users"))
    database.initialize_user_database(910, load_demo=False)
    database.initialize_user_database(911, load_demo=False)
    try:
        database.set_current_user(910)
        save_budget("Travel", 3000)
        database.clear_current_user()
        database.set_current_user(911)
        connection = database.get_connection()
        try:
            count = connection.execute("SELECT COUNT(*) FROM budgets").fetchone()[0]
        finally:
            connection.close()
        assert count == 0
    finally:
        database.clear_current_user()


def test_forecast_is_for_next_calendar_day_and_reports_daily_baseline(monkeypatch):
    from src import ml_expense_prediction
    dates = pd.date_range("2026-02-01", periods=50, freq="D")
    frame = pd.DataFrame({"date": dates, "amount": [float(100 + (i % 5) * 20) for i in range(50)]})
    monkeypatch.setattr(ml_expense_prediction, "get_expense_data", lambda: frame)
    result = ml_expense_prediction.get_prediction()
    assert result is not None
    assert result["date"] == pd.Timestamp("2026-03-23")
    assert result["latest_record_date"] == pd.Timestamp("2026-03-22")
    assert result["forecast_type"] == "next_calendar_day_total"
    assert result["historical_daily_average"] > 0


def test_budget_page_does_not_fabricate_budget_scenarios():
    app_source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    assert "ACTUAL MONTHLY SPENDING HISTORY" in app_source
    assert "Monthly Budget Demo Scenarios" not in app_source
    assert "spending_value / 0.70" not in app_source
    assert "Manage Category Budgets" in app_source



def test_finance_chat_routes_knowledge_questions_to_rag_before_expense_keywords():
    import ast

    app_source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    module = ast.parse(app_source)
    function_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "detect_chat_intent"
    )
    isolated_module = ast.Module(body=[function_node], type_ignores=[])
    namespace = {}
    exec(compile(isolated_module, "app.py", "exec"), namespace)
    detect = namespace["detect_chat_intent"]

    assert detect(
        "How much money should I keep in an emergency fund, and how should I calculate the amount based on my monthly essential expenses?"
    ) == "general"
    assert detect("How much did I spend on food?") == "expenses"
    assert detect("What are my total expenses?") == "expenses"
    assert detect("Which category has the highest spending?") == "category"
    assert detect("How do I calculate my savings rate?") == "savings"



def test_mock_rag_generates_an_explanation_instead_of_dumping_passage(monkeypatch):
    import src.llm_client as client_module

    monkeypatch.setattr(client_module, "MOCK_LLM", True)
    transactions = pd.DataFrame({
        "type": ["Income", "Expense"],
        "category": ["Salary", "Food"],
        "amount": [30000, 12000],
    })
    retrieved = (
        "### Emergency Fund\\n"
        "**Source:** finance_basics.md\\n"
        "**Relevance:** 0.715\\n\\n"
        "An emergency fund is money kept aside for unexpected and necessary "
        "expenses. A common budgeting guideline is to build several months "
        "of essential living expenses as an emergency reserve. The "
        "appropriate amount depends on income stability, essential "
        "expenses, dependents, insurance coverage, and other circumstances."
    )

    response = client_module.answer_with_llm(
        "How much money should I keep in an emergency fund, and how should "
        "I calculate the amount based on my monthly essential expenses?",
        transactions,
        retrieved_context=retrieved,
    )

    assert "Emergency Fund" in response
    assert "Monthly essential expenses" in response
    assert "Months of coverage" in response
    assert "cannot safely give you a personal target" in response
    assert "Retrieved knowledge-base context" not in response
    assert "live LLM is not generating" in response



def test_finance_chat_uses_editable_form_composer_instead_of_pinned_chat_input():
    app_source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")
    chat_section = app_source.split('elif page == "💬 Finance Chat":', 1)[1].split(
        "# ============================================================", 1
    )[0]
    assert 'st.form("finance_chat_form"' in chat_section
    assert 'st.text_input(' in chat_section
    assert 'st.form_submit_button(' in chat_section
    assert 'st.chat_input(' not in chat_section



def test_mock_rag_answers_needs_and_wants_in_own_words(monkeypatch):
    import src.llm_client as client_module

    monkeypatch.setattr(client_module, "MOCK_LLM", True)
    transactions = pd.DataFrame({
        "type": ["Income", "Expense"],
        "category": ["Salary", "Food"],
        "amount": [30000, 12000],
    })
    passage = (
        "### Needs and Wants\\n**Source:** finance_basics.md\\n"
        "**Relevance:** 0.798\\n\\n"
        "Needs are expenses required for basic living or essential "
        "obligations. Examples can include basic housing, food, utilities, "
        "transportation, and required healthcare. Wants are discretionary "
        "expenses that may improve comfort or enjoyment but are not "
        "essential for basic needs. The distinction is personal and can "
        "vary by situation."
    )
    answer = client_module.answer_with_llm(
        "What is the difference between needs and wants?",
        transactions,
        retrieved_context=passage,
    )
    assert "Needs vs. Wants" in answer
    assert "How to tell them apart" in answer
    assert "cover essential needs first" in answer
    assert "Retrieved knowledge-base context" not in answer
