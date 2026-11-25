"""Explainable, multi-feature Isolation Forest expense anomaly detection."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.database import get_connection


def get_expense_data():
    connection = get_connection()
    try:
        return pd.read_sql_query(
            """SELECT id, date, category, description, amount, payment_method
               FROM transactions WHERE LOWER(TRIM(type)) = 'expense' ORDER BY date""",
            connection,
        )
    finally:
        connection.close()


def _build_features(df):
    amounts = pd.to_numeric(df["amount"], errors="coerce").clip(lower=0)
    log_amount = np.log1p(amounts)
    category_median = amounts.groupby(df["category"].fillna("Uncategorised")).transform("median").clip(lower=0.01)
    dates = pd.to_datetime(df["date"], errors="coerce")
    features = pd.DataFrame(index=df.index)
    features["log_amount"] = log_amount
    features["category_relative_amount"] = np.log1p(amounts / category_median)
    features["day_of_week"] = dates.dt.dayofweek.fillna(-1)
    features["day_of_month"] = dates.dt.day.fillna(0)
    features["month"] = dates.dt.month.fillna(0)
    category_dummies = pd.get_dummies(df["category"].fillna("Uncategorised").astype(str), prefix="category", dtype=float)
    method_dummies = pd.get_dummies(df["payment_method"].fillna("Unknown").astype(str), prefix="method", dtype=float)
    return pd.concat([features, category_dummies, method_dummies], axis=1).replace([np.inf, -np.inf], 0).fillna(0), category_median


def detect_anomalies(contamination=0.05):
    df = get_expense_data()
    if df is None or df.empty:
        return None
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"]).copy()
    df = df[df["amount"] >= 0].copy()
    if len(df) < 10:
        return None
    df["category"] = df["category"].fillna("Uncategorised").astype(str)
    features, category_median = _build_features(df)
    model = IsolationForest(
        n_estimators=250,
        contamination=min(max(float(contamination), 0.01), 0.20),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features)
    df["anomaly_prediction"] = model.predict(features)
    df["anomaly_score"] = model.decision_function(features)
    df["category_median_amount"] = category_median.astype(float)
    df["category_amount_ratio"] = (
        df["amount"] / df["category_median_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    def explain(row):
        if row["anomaly_prediction"] != -1:
            return "No strong unusual-spending signal detected."
        ratio = float(row["category_amount_ratio"])
        if ratio >= 2:
            return f"Amount is {ratio:.1f}× the median transaction in {row['category']}."
        return "Combination of amount, category, date and payment-method patterns is unusual."
    df["anomaly_reason"] = df.apply(explain, axis=1)
    df["status"] = df["anomaly_prediction"].map({-1: "🚨 Anomaly", 1: "✅ Normal"})
    return df


def get_anomalies():
    result = detect_anomalies()
    if result is None:
        return None
    return result[result["anomaly_prediction"] == -1].copy()


if __name__ == "__main__":
    result = detect_anomalies()
    if result is None:
        print("Not enough valid expense data for anomaly detection.")
    else:
        flagged = result[result["anomaly_prediction"] == -1]
        print(f"Expense transactions: {len(result)} | Anomalies: {len(flagged)}")
        columns = ["date", "category", "description", "amount", "anomaly_score", "anomaly_reason"]
        print(flagged[columns].to_string(index=False) if not flagged.empty else "No unusual transactions detected.")
