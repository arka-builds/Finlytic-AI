"""Daily expense forecasting using a chronological Random Forest baseline."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.database import get_connection

FEATURES = [
    "day_of_week", "day_of_month", "month",
    "previous_day_expense", "previous_7_day_expense", "rolling_7_day_avg",
]


def get_expense_data():
    connection = get_connection()
    try:
        return pd.read_sql_query(
            """SELECT date, amount FROM transactions
               WHERE LOWER(TRIM(type)) = 'expense' ORDER BY date""",
            connection,
        )
    finally:
        connection.close()


def prepare_dataset(df):
    """Aggregate transactions to daily totals and create leakage-safe lags.

    Calendar days without expenses are retained as zero-spend days. The model
    therefore predicts the next day's total recorded expense, not an arbitrary
    next transaction amount.
    """
    if df is None or df.empty or not {"date", "amount"}.issubset(df.columns):
        return pd.DataFrame()
    work = df[["date", "amount"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["date", "amount"])
    work = work[work["amount"] >= 0]
    if work.empty:
        return pd.DataFrame()
    work["date"] = work["date"].dt.normalize()
    daily = work.groupby("date")["amount"].sum().sort_index()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0.0)
    result = daily.rename("amount").rename_axis("date").reset_index()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_month"] = result["date"].dt.day
    result["month"] = result["date"].dt.month
    result["previous_day_expense"] = result["amount"].shift(1)
    result["previous_7_day_expense"] = result["amount"].rolling(7).sum().shift(1)
    result["rolling_7_day_avg"] = result["amount"].rolling(7).mean().shift(1)
    return result.dropna().reset_index(drop=True)


def train_model(df):
    if df is None or len(df) < 10:
        raise ValueError("At least 10 prepared daily records are required to train the model.")
    X, y = df[FEATURES], df["amount"]
    split_index = min(max(int(len(df) * 0.8), 2), len(df) - 2)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    model = RandomForestRegressor(
        n_estimators=250, random_state=42, max_depth=8,
        min_samples_leaf=2, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0
    return model, float(mae), float(rmse), float(r2)


def baseline_mae(df):
    """MAE of the naive "tomorrow = trailing 7-day average" forecast.

    Evaluated on the same chronological holdout as ``train_model``. A model
    is only worth shipping if it beats this simple baseline.
    """
    split_index = min(max(int(len(df) * 0.8), 2), len(df) - 2)
    test = df.iloc[split_index:]
    return float(mean_absolute_error(test["amount"], test["rolling_7_day_avg"]))


def predict_next_day(model, df):
    if df is None or df.empty:
        raise ValueError("No prepared daily expense history is available.")
    last_date = pd.to_datetime(df["date"]).max()
    next_date = last_date + pd.Timedelta(days=1)
    recent = df.sort_values("date").tail(7)["amount"].astype(float)
    if recent.empty:
        raise ValueError("Not enough recent daily history to forecast.")
    features = pd.DataFrame([{
        "day_of_week": next_date.dayofweek,
        "day_of_month": next_date.day,
        "month": next_date.month,
        "previous_day_expense": float(recent.iloc[-1]),
        "previous_7_day_expense": float(recent.sum()),
        "rolling_7_day_avg": float(recent.mean()),
    }])
    prediction = max(0.0, float(model.predict(features)[0]))
    return next_date, prediction


def get_prediction():
    raw = get_expense_data()
    prepared = prepare_dataset(raw)
    if len(prepared) < 10:
        return None
    model, mae, rmse, r2 = train_model(prepared)
    naive_mae = baseline_mae(prepared)
    next_date, prediction = predict_next_day(model, prepared)
    latest_record_date = pd.to_datetime(raw["date"], errors="coerce").max()
    return {
        "date": next_date,
        "prediction": prediction,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "baseline_mae": naive_mae,
        "beats_baseline": mae < naive_mae,
        "latest_record_date": latest_record_date,
        "historical_daily_average": float(prepared["amount"].mean()),
        "forecast_type": "next_calendar_day_total",
        "training_days": int(len(prepared)),
    }


if __name__ == "__main__":
    result = get_prediction()
    if result:
        print("Daily expense forecast")
        print(f"Latest recorded date: {result['latest_record_date'].date()}")
        print(f"Forecast date: {result['date'].date()}")
        print(f"Predicted daily expenses: ₹{result['prediction']:,.2f}")
        print(f"Baseline (7-day average) MAE: ₹{result['baseline_mae']:,.2f} | beats baseline: {result['beats_baseline']}")
        print(f"MAE: ₹{result['mae']:,.2f} | RMSE: ₹{result['rmse']:,.2f} | R²: {result['r2']:.4f}")
        print(f"Training calendar days: {result['training_days']}")
    else:
        print("Not enough daily expense history to produce a forecast.")
