"""Tests for the tool-calling agent, hybrid retrieval and evaluation harness."""
import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import agent, rag_engine

TODAY = date(2026, 9, 30)


def make_ctx(**overrides):
    rows = [
        ("2026-08-10", "Expense", "Food", "Lunch", 1000.0),
        ("2026-08-12", "Expense", "Transport", "Cab", 500.0),
        ("2026-09-05", "Expense", "Food", "Dinner", 2000.0),
        ("2026-09-06", "Expense", "Shopping", "Shoes", 3000.0),
        ("2026-09-15", "Income", "Salary", "Salary", 40000.0),
    ]
    df = pd.DataFrame(rows, columns=["date", "type", "category", "description", "amount"])
    base = dict(
        df=df,
        budgets=lambda: pd.DataFrame({"category": ["Food", "Shopping"], "budget": [2500.0, 2000.0]}),
        today=TODAY,
    )
    base.update(overrides)
    return agent.ToolContext(**base)


# ---------- tools ----------

def test_summary_and_category_tools_use_data_not_guesses():
    ctx = make_ctx()
    s = agent.tool_get_summary(ctx)
    assert s["total_income"] == 40000.0 and s["total_expenses"] == 6500.0 and s["savings_rate_pct"] == 83.8
    c = agent.tool_spending_by_category(ctx, month="2026-09")
    assert c["categories"][0] == {"category": "Shopping", "amount": 3000.0, "share_pct": 60.0}


def test_compare_months_and_budget_status():
    ctx = make_ctx()
    cmp_ = agent.tool_compare_months(ctx, "2026-08", "2026-09")
    assert cmp_["expenses_a"] == 1500.0 and cmp_["expenses_b"] == 5000.0 and cmp_["change_pct"] == 233.3
    status = {r["category"]: r["status"] for r in agent.tool_budget_status(ctx, "2026-09")["categories"]}
    assert status == {"Food": "near_limit", "Shopping": "over"}


# ---------- validation / guardrails ----------

@pytest.mark.parametrize("name,args,fragment", [
    ("compare_months", {"month_a": "2026-08"}, "missing"),
    ("spending_by_category", {"month": "September"}, "YYYY-MM"),
    ("spending_by_category", {"top_n": 500}, "out of range"),
    ("spending_by_category", {"top_n": "5"}, "integer"),
    ("get_summary", {"drop_table": 1}, "unexpected"),
    ("delete_everything", {}, "unknown tool"),
])
def test_invalid_tool_calls_are_rejected_not_executed(name, args, fragment):
    call = agent.execute(make_ctx(), name, args)
    assert call.result is None and fragment in call.error


def test_failing_tool_is_contained():
    def boom():
        raise RuntimeError("db down")
    call = agent.execute(make_ctx(forecast=boom), "forecast_next_day", {})
    assert "Tool failed" in call.error


def test_tool_output_is_truncated():
    big = agent.ToolCall("x", {}, result={"blob": "a" * 10_000})
    assert len(agent.serialise(big)) < agent.MAX_TOOL_OUTPUT_CHARS + 50


# ---------- offline planner ----------

@pytest.mark.parametrize("question,expected", [
    ("Compare august and september", ["compare_months"]),
    ("How much did I spend last month?", ["spending_by_category"]),
    ("Am I over budget this month?", ["budget_status"]),
    ("What will I spend tomorrow?", ["forecast_next_day"]),
    ("Any unusual transactions?", ["list_anomalies"]),
    ("How do I build an emergency fund?", ["search_knowledge_base"]),
])
def test_offline_planner_routes_to_the_right_tool(question, expected):
    assert [name for name, _ in agent.plan_offline(question, TODAY)] == expected


def test_last_month_resolves_relative_to_today():
    name, args = agent.plan_offline("How much did I spend last month?", TODAY)[0]
    assert args == {"month": "2026-08"}


def test_offline_answer_cites_knowledge_base_sources():
    hit = {"section": "Emergency Fund", "source": "finance_basics.md", "text": "Keep several months aside."}
    ctx = make_ctx(search=lambda q: [hit])
    result = agent.run_offline("What is an emergency fund?", ctx)
    assert "[1]" in result.answer and "Sources" in result.answer and result.sources


# ---------- OpenAI loop with a fake client ----------

class FakeClient:
    """Scripts a model that calls one valid tool, one invalid tool, then answers."""
    def __init__(self):
        self.seen_outputs = []
        self.responses = SimpleNamespace(create=self._create)
        self.n = 0

    def _create(self, **kw):
        self.n += 1
        if self.n == 1:
            items = [
                SimpleNamespace(type="function_call", name="get_summary", arguments="{}", call_id="c1"),
                SimpleNamespace(type="function_call", name="spending_by_category",
                                arguments=json.dumps({"month": "last month"}), call_id="c2"),
            ]
            return SimpleNamespace(id="r1", output=items, output_text="")
        self.seen_outputs = kw["input"]
        return SimpleNamespace(id="r2", output=[], output_text="Final answer.")


def test_openai_loop_executes_validated_tools_and_feeds_errors_back():
    client = FakeClient()
    result = agent.run_openai("How am I doing?", make_ctx(), client, "test-model")
    assert result.answer == "Final answer." and result.mode == "openai"
    assert [c.name for c in result.calls] == ["get_summary", "spending_by_category"]
    assert result.calls[0].error is None and "YYYY-MM" in result.calls[1].error
    outputs = {o["call_id"]: json.loads(o["output"]) for o in client.seen_outputs}
    assert "total_income" in outputs["c1"] and "error" in outputs["c2"]


def test_openai_step_limit_forces_a_final_answer():
    class Looping:
        def __init__(self):
            self.responses = SimpleNamespace(create=self.create)
            self.n = 0
        def create(self, **kw):
            self.n += 1
            if "Provide the final answer" in str(kw.get("input")):
                return SimpleNamespace(id="end", output=[], output_text="Stopped.")
            call = SimpleNamespace(type="function_call", name="get_summary", arguments="{}", call_id=f"c{self.n}")
            return SimpleNamespace(id=f"r{self.n}", output=[call], output_text="")
    client = Looping()
    result = agent.run_openai("loop", make_ctx(), client, "m", max_steps=3)
    assert result.answer == "Stopped." and len(result.calls) == 3


# ---------- hybrid retrieval ----------

def test_bm25_ranks_exact_term_matches_first_and_rrf_fuses_rankings():
    docs = [rag_engine.tokenize(t) for t in ["emergency fund reserve", "budget plan monthly", "debt interest minimum payment"]]
    bm = rag_engine.BM25(docs)
    raw, ceiling = bm.scores(rag_engine.tokenize("minimum payment on debt"))
    assert int(np.argmax(raw)) == 2 and 0 < raw[2] / ceiling <= 1
    fused = rag_engine.reciprocal_rank_fusion([[2, 0, 1], [0, 2, 1]])
    assert fused[2] == fused[0] > fused[1]


def test_hybrid_retrieval_gates_off_topic_and_exposes_per_ranker_scores():
    orig = rag_engine.get_embedding_model
    rag_engine.get_embedding_model = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    try:
        retriever = rag_engine.LocalRAGRetriever()
    finally:
        rag_engine.get_embedding_model = orig
    hits = retriever.retrieve("What happens if I only pay the minimum on my credit card?", method="hybrid")
    assert hits and hits[0]["section"] == "Debt Management"
    assert {"dense_score", "bm25_score", "fused_score"} <= hits[0].keys()
    assert retriever.retrieve("Who won the football world cup?", method="hybrid") == []
    with pytest.raises(ValueError):
        retriever.retrieve("x y z", method="nope")


def test_retrieval_eval_harness_meets_a_quality_floor():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
    import run_rag_eval
    retrievers = run_rag_eval.build_retrievers()
    retriever, method = retrievers["Hybrid (TF-IDF + BM25, RRF)"]
    m = run_rag_eval.evaluate(retriever, method, run_rag_eval.load_set())
    assert m["mrr"] >= 0.85 and m["rejection"] == 1.0


def test_combined_question_uses_current_month_for_budget_and_category():
    today = date(2026, 11, 26)
    question = (
        "Compare my spending this month with last month, identify my highest "
        "spending category this month, and check whether I exceeded my monthly budget."
    )
    plan = agent.plan_offline(question, today)
    planned = {name: args for name, args in plan}
    assert planned["compare_months"] == {"month_a": "2026-10", "month_b": "2026-11"}
    assert planned["budget_status"] == {"month": "2026-11"}
    assert planned["spending_by_category"]["month"] == "2026-11"


def test_combined_question_composes_comparison_highest_category_and_budget():
    today = date(2026, 11, 26)
    df = pd.DataFrame([
        ("2026-10-05", "Expense", "Bills", "Electricity", 10000.0),
        ("2026-11-05", "Expense", "Food", "Groceries", 4000.0),
        ("2026-11-06", "Expense", "Transport", "Taxi", 5000.0),
    ], columns=["date", "type", "category", "description", "amount"])
    ctx = agent.ToolContext(
        df=df,
        budgets=lambda: pd.DataFrame({
            "category": ["Food", "Transport"],
            "budget": [3000.0, 6000.0],
        }),
        today=today,
    )
    result = agent.run_offline(
        "Compare my spending this month with last month, identify my highest spending "
        "category this month, and check whether I exceeded my monthly budget.",
        ctx,
    )
    assert "Highest spending category (2026-11)" in result.answer
    assert "Transport" in result.answer
    assert "You exceeded 1 category budget" in result.answer
    assert "Food" in result.answer
    assert "Budget status (2026-11)" in result.answer
    assert "Offline mode" not in result.answer


def test_combined_financial_answer_does_not_append_unrequested_monthly_trend():
    comparison = agent.ToolCall(
        name="compare_months", args={},
        result={
            "month_a": "2026-10", "month_b": "2026-11",
            "expenses_a": 10000.0, "expenses_b": 8000.0,
            "change": -2000.0, "change_pct": -20.0,
            "largest_category_changes": [],
        },
    )
    category = agent.ToolCall(
        name="spending_by_category", args={},
        result={"month": "2026-11", "categories": [
            {"category": "Food", "amount": 3000.0, "share_pct": 37.5}
        ]},
    )
    budget = agent.ToolCall(
        name="budget_status", args={},
        result={"month": "2026-11", "categories": [
            {"category": "Food", "spent": 3000.0, "budget": 2500.0,
             "used_pct": 120.0, "status": "over"}
        ]},
    )
    trend = agent.ToolCall(
        name="monthly_trend", args={},
        result={"months": [
            {"month": "2026-10", "income": 10000.0, "expenses": 10000.0, "net": 0.0},
            {"month": "2026-11", "income": 10000.0, "expenses": 8000.0, "net": 2000.0},
        ]},
    )
    answer, _ = agent.compose_offline([comparison, category, budget, trend])
    assert "Monthly spending comparison" in answer
    assert "Highest spending category" in answer
    assert "Budget status" in answer
    assert "Monthly trend" not in answer
    assert "| Month | Income | Expenses | Net |" not in answer


def test_answer_logs_offline_reason_when_mock_mode_enabled(monkeypatch, caplog):
    import logging
    import src.config as config

    monkeypatch.setattr(config, "MOCK_LLM", True)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    df = pd.DataFrame(columns=["date", "type", "category", "description", "amount"])
    with caplog.at_level(logging.INFO):
        result = agent.answer("What is my summary?", df, agent.ToolContext(df=df))
    assert result.mode == "offline"
    assert "MOCK_LLM is enabled" in caplog.text
    assert "OFFLINE execution completed" in caplog.text


def test_answer_reports_live_openai_failure_and_fallback(monkeypatch, caplog):
    import logging
    import src.config as config

    class BrokenOpenAI:
        def __init__(self, api_key):
            raise RuntimeError("simulated API connection failure")

    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(config, "OPENAI_MODEL", "test-model")
    import sys
    import types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = BrokenOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    df = pd.DataFrame(columns=["date", "type", "category", "description", "amount"])
    with caplog.at_level(logging.INFO):
        result = agent.answer("What is my summary?", df, agent.ToolContext(df=df))
    assert result.mode == "offline"
    assert "LIVE OPENAI failed" in caplog.text
    assert "simulated API connection failure" in caplog.text
