"""Data-access functions for the signed-in user's transactions and budgets."""
import pandas as pd

from src.database import get_connection


def get_transactions():
    connection = get_connection()

    query = """
        SELECT
            id,
            date,
            type,
            category,
            description,
            amount,
            payment_method
        FROM transactions
        ORDER BY date DESC, id DESC
    """

    try:
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


def get_budgets():
    connection = get_connection()

    query = """
        SELECT
            category,
            budget
        FROM budgets
        ORDER BY category
    """

    try:
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()



def add_transaction(
    transaction_date,
    transaction_type,
    category,
    description,
    amount,
    payment_method,
):
    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO transactions
            (
                date,
                type,
                category,
                description,
                amount,
                payment_method
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                transaction_date.isoformat(),
                transaction_type,
                category,
                description.strip(),
                float(amount),
                payment_method,
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def update_transaction(
    transaction_id,
    transaction_date,
    transaction_type,
    category,
    description,
    amount,
    payment_method,
):
    connection = get_connection()

    try:
        connection.execute(
            """
            UPDATE transactions
            SET
                date = ?,
                type = ?,
                category = ?,
                description = ?,
                amount = ?,
                payment_method = ?
            WHERE id = ?
            """,
            (
                transaction_date.isoformat(),
                transaction_type,
                category,
                description.strip(),
                float(amount),
                payment_method,
                int(transaction_id),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def delete_transaction(transaction_id):
    connection = get_connection()

    try:
        connection.execute(
            "DELETE FROM transactions WHERE id = ?",
            (int(transaction_id),),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
