"""End-to-end smoke test: drive the real Streamlit app through every page."""
import importlib
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "🏠 Dashboard", "💳 Transactions", "📊 Analysis",
    "💰 Budget", "🤖 AI Advisor", "💬 Finance Chat", "🚨 Anomaly Detection",
]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FINLYTIC_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MOCK_LLM", "true")
    import src.auth, src.database, src.config, src.llm_client
    for module in (src.config, src.database, src.auth, src.llm_client):
        importlib.reload(module)
    return AppTest.from_file(APP, default_timeout=90).run()


def _sign_up_and_log_in(at):
    from src import auth
    ok, msg, _ = auth.register_user("Smoke Tester", "smoke@example.com", "correct-horse-battery")
    assert ok, msg
    at.text_input[0].set_value("smoke@example.com")
    at.text_input[1].set_value("correct-horse-battery")
    at.button[0].click()
    return at.run()


def test_login_screen_renders_without_errors(app):
    assert not app.exception
    assert [t.label for t in app.tabs][:3] == ["🔐 Login", "📝 Register", "🔑 Forgot Password"]


def test_every_page_renders_after_login(app):
    at = _sign_up_and_log_in(app)
    assert not at.exception, at.exception
    assert at.session_state["authenticated"] is True
    for page in PAGES:
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception, f"{page}: {at.exception}"
