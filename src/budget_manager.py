"""Database-backed category budget management for the Streamlit app."""
from src.database import get_connection


def save_budget(category: str, amount: float) -> None:
    """Create a category budget or update its existing amount."""
    category = str(category or "").strip().title()
    try:
        amount = float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter a valid budget amount.") from exc
    if not category:
        raise ValueError("Budget category is required.")
    if amount <= 0:
        raise ValueError("Budget amount must be greater than zero.")
    if len(category) > 60:
        raise ValueError("Category name must be 60 characters or fewer.")

    connection = get_connection()
    try:
        connection.execute(
            """INSERT INTO budgets (category, budget) VALUES (?, ?)
               ON CONFLICT(category) DO UPDATE SET budget = excluded.budget""",
            (category, amount),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def delete_budget(category: str) -> bool:
    """Delete a category budget. Return False if it did not exist."""
    category = str(category or "").strip()
    if not category:
        raise ValueError("Select a budget category to delete.")
    connection = get_connection()
    try:
        cursor = connection.execute(
            "DELETE FROM budgets WHERE category = ?", (category,)
        )
        connection.commit()
        return cursor.rowcount > 0
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
