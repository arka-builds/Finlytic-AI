import sqlite3
import csv
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from contextvars import ContextVar


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Seed files shipped with the repository.
SEED_DIR = PROJECT_ROOT / "data"

# Writable runtime data (auth.db, per-user databases). Override with
# FINLYTIC_DATA_DIR to point at a persistent volume when deploying.
DATA_DIR = os.getenv("FINLYTIC_DATA_DIR") or str(SEED_DIR)
USERS_DIR = os.path.join(DATA_DIR, "users")

TRANSACTIONS_CSV = str(SEED_DIR / "transactions.csv")
BUDGETS_CSV = str(SEED_DIR / "budgets.csv")


# ============================================================
# CURRENT USER CONTEXT
# ============================================================

_CURRENT_USER_ID = ContextVar("finlytic_current_user_id", default=None)


def set_current_user(user_id):
    """
    Set the authenticated user for the current application session.
    """
    if user_id is None:
        raise ValueError("user_id cannot be None.")

    _CURRENT_USER_ID.set(int(user_id))


def clear_current_user():
    """Clear the authenticated user in the current execution context."""
    _CURRENT_USER_ID.set(None)


def get_current_user_id():
    """
    Return the currently authenticated user ID.
    """
    return _CURRENT_USER_ID.get()


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_data_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(USERS_DIR, exist_ok=True)


# ============================================================
# USER DATABASE
# ============================================================

def get_user_db_path(user_id):
    if user_id is None:
        raise ValueError("user_id is required for a user database.")

    ensure_data_directories()

    return os.path.join(
        USERS_DIR,
        f"user_{int(user_id)}.db"
    )


def get_user_connection(user_id):
    """
    Return a SQLite connection for a specific user.
    """
    db_path = get_user_db_path(user_id)

    connection = sqlite3.connect(db_path, timeout=30)

    return connection


# ============================================================
# MAIN APPLICATION CONNECTION
# ============================================================

def get_connection():
    """
    Return the database connection for the authenticated user.

    All application modules should use this function.
    """

    user_id = _CURRENT_USER_ID.get()
    if user_id is None:
        raise RuntimeError(
            "No authenticated user is set. "
            "Call set_current_user(user_id) after login."
        )

    return get_user_connection(user_id)


# ============================================================
# USER DATABASE TABLES
# ============================================================

def create_tables(user_id):
    connection = get_user_connection(user_id)

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            amount REAL NOT NULL,
            payment_method TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS budgets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT UNIQUE NOT NULL,
            budget REAL NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# DEMO DATA
# ============================================================

def _rebase_demo_dates(records, today=None):
    """Shift bundled demo transactions so the newest one is dated today.

    The demo CSV is a fixed snapshot. Without this, every new demo account
    would open to a stale dataset (empty "current month", forecast warnings).
    The shift is applied once, when a brand-new account is seeded, and is
    never applied to data a user entered themselves.
    """
    if not records:
        return records
    try:
        parsed = [datetime.strptime(r[0], "%Y-%m-%d").date() for r in records]
    except ValueError:
        return records  # unexpected format: keep the data exactly as provided
    shift = timedelta(days=((today or date.today()) - max(parsed)).days)
    return [
        ((d + shift).isoformat(),) + tuple(r[1:])
        for d, r in zip(parsed, records)
    ]



def load_demo_data(user_id):
    """Import bundled demo CSV data at most once for a new user database.

    The marker is stored even when no CSV data is present. This prevents an
    intentionally emptied account from being silently repopulated on a later
    application rerun.
    """
    connection = get_user_connection(user_id)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT 1 FROM app_meta WHERE key = ?",
            ("demo_data_initialized_v1",),
        )
        if cursor.fetchone():
            return

        cursor.execute("SELECT COUNT(*) FROM transactions")
        transaction_count = cursor.fetchone()[0]
        if transaction_count == 0 and os.path.exists(TRANSACTIONS_CSV):
            with open(TRANSACTIONS_CSV, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                records = []
                for row in reader:
                    try:
                        amount = float(row["amount"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if amount < 0 or not row.get("date") or not row.get("type") or not row.get("category"):
                        continue
                    records.append((
                        row["date"].strip(),
                        row["type"],
                        row["category"],
                        row.get("description", ""),
                        amount,
                        row.get("payment_method", ""),
                    ))
                records = _rebase_demo_dates(records)
                if records:
                    cursor.executemany(
                        """
                        INSERT INTO transactions
                        (date, type, category, description, amount, payment_method)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        records,
                    )

        cursor.execute("SELECT COUNT(*) FROM budgets")
        budget_count = cursor.fetchone()[0]
        if budget_count == 0 and os.path.exists(BUDGETS_CSV):
            with open(BUDGETS_CSV, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                records = []
                for row in reader:
                    try:
                        budget = float(row["budget"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if budget < 0 or not row.get("category"):
                        continue
                    records.append((row["category"], budget))
                if records:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO budgets (category, budget) VALUES (?, ?)",
                        records,
                    )

        cursor.execute(
            "INSERT INTO app_meta (key, value) VALUES (?, ?)",
            ("demo_data_initialized_v1", "1"),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ============================================================
# INITIALIZE USER DATABASE
# ============================================================

def initialize_user_database(
    user_id,
    load_demo=True
):
    """
    Create and optionally populate a user's database.
    """

    if user_id is None:
        raise ValueError(
            "Cannot initialize database without user_id."
        )

    create_tables(user_id)

    if load_demo:
        load_demo_data(user_id)


# ============================================================
# USER DATABASE CHECK
# ============================================================

def user_database_exists(user_id):

    if user_id is None:
        return False

    return os.path.exists(
        get_user_db_path(user_id)
    )


# ============================================================
# DELETE USER DATABASE
# ============================================================

def delete_user_database(user_id):

    db_path = get_user_db_path(user_id)

    if os.path.exists(db_path):
        os.remove(db_path)
