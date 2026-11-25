"""Generate the synthetic demo dataset bundled with Finlytic AI.

Deterministic (seeded) so the demo is reproducible. All records are
synthetic. Usage:

    python scripts/generate_demo_data.py [--out data/transactions.csv]

New accounts are seeded from this file and its dates are re-based so the
newest record is dated "today" (see ``src.database._rebase_demo_dates``).
"""
import argparse
import csv
import random
from datetime import date, timedelta

CATEGORIES = {  # category: (min, max, weight, descriptions)
    "Food": (80, 450, 6, ["Lunch", "Groceries", "Dinner", "Coffee", "Snacks"]),
    "Transport": (40, 350, 4, ["Metro", "Cab", "Fuel", "Auto"]),
    "Shopping": (300, 2200, 1.2, ["Clothes", "Online order", "Accessories"]),
    "Entertainment": (150, 900, 1.5, ["Movie", "Streaming", "Outing"]),
    "Education": (200, 1500, 0.6, ["Course", "Books", "Exam fee"]),
    "Healthcare": (100, 900, 0.5, ["Pharmacy", "Clinic"]),
}
METHODS = ["UPI", "Card", "Cash", "Bank"]
OUTLIERS = [(37, "Shopping", "Laptop bag + accessories", 9800), (84, "Entertainment", "Concert tickets", 5200),
            (121, "Healthcare", "Dental procedure", 7400)]


def build(days=150, end=date(2026, 9, 30), seed=42):
    rng = random.Random(seed)
    start = end - timedelta(days=days - 1)
    names = list(CATEGORIES)
    weights = [CATEGORIES[n][2] for n in names]
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
            cat = rng.choices(names, weights)[0]
            lo, hi, _, descs = CATEGORIES[cat]
            rows.append((d, "Expense", cat, rng.choice(descs), round(rng.uniform(lo, hi), 2), rng.choice(METHODS)))
        if d.day == 1:
            rows.append((d, "Expense", "Bills", "Rent", 9000.0, "Bank"))
        if d.day == 5:
            rows.append((d, "Expense", "Bills", "Utilities & internet", round(rng.uniform(1100, 1600), 2), "UPI"))
        if d.day == 15:
            rows.append((d, "Income", "Salary", d.strftime("%B Salary"), 45000.0, "Bank Transfer"))
    for offset, cat, desc, amount in OUTLIERS:
        rows.append((start + timedelta(days=offset), "Expense", cat, desc, float(amount), "Card"))
    return sorted(rows, key=lambda r: (r[0], r[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/transactions.csv")
    args = parser.parse_args()
    rows = build()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "type", "category", "description", "amount", "payment_method"])
        for r in rows:
            w.writerow([r[0].isoformat(), *r[1:]])
    print(f"Wrote {len(rows)} synthetic transactions to {args.out}")
