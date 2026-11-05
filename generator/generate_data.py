"""
Synthetic data generator for the income reconciliation pipeline.

Creates four "source system" files, the way a custodian would receive them:
    securities.csv          security master (reference data)
    positions.csv           shares each account held on each record date
    dividend_events.csv     announced dividends
    custodian_payments.csv  cash actually received from the depository

It also writes an answer key (break_manifest.csv) listing every break it
planted on purpose, so the pipeline can later be tested against it.

All data is fake. Run from the project root:
    python generator/generate_data.py
"""

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NUM_SECURITIES = 25
NUM_ACCOUNTS = 200
HOLDINGS_PER_ACCOUNT = (3, 8)        # min and max securities per account
START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 10, 31)
US_WITHHOLDING_RATE = 0.15           # tax withheld on USD dividends
MISMATCH_TOLERANCE = 0.01            # differences up to 1 cent are not breaks

# How many of each break type to plant
BREAK_COUNTS = {
    "MISSING_PAYMENT": 30,     # entitlement expected, no payment received
    "UNEXPECTED_PAYMENT": 15,  # payment received, no entitlement exists
    "AMOUNT_MISMATCH": 40,     # paid, but gross amount is wrong
    "DUPLICATE_PAYMENT": 12,   # same payment received twice
    "LATE_PAYMENT": 25,        # received after the scheduled pay date
}
DIRTY_ISIN_RATE = 0.05   # share of payment rows with messy ISIN formatting
DIRTY_DATE_RATE = 0.05   # share of payment rows with a non-standard date format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def business_days_after(start: date, n: int) -> date:
    """Return the date n business days (Mon to Fri) after start."""
    d = start
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def to_weekday(d: date) -> date:
    """Move a weekend date forward to Monday."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def make_isin(country: str, rng: random.Random) -> str:
    """Build an ISIN shaped identifier (2 letter country, 9 digits, check digit).
    The check digit is random, so these are clearly synthetic."""
    return country + "".join(str(rng.randint(0, 9)) for _ in range(10))


# ---------------------------------------------------------------------------
# Generators for each table
# ---------------------------------------------------------------------------
def generate_securities(rng: random.Random, fake: Faker) -> pd.DataFrame:
    rows, used_tickers = [], set()
    for i in range(1, NUM_SECURITIES + 1):
        currency = "CAD" if rng.random() < 0.6 else "USD"
        name = fake.unique.company()
        ticker = "".join(c for c in name.upper() if c.isalpha())[:4]
        while ticker in used_tickers:
            ticker = ticker[:3] + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        used_tickers.add(ticker)
        rows.append({
            "security_id": f"SEC{i:03d}",
            "isin": make_isin("CA" if currency == "CAD" else "US", rng),
            "ticker": ticker,
            "security_name": name,
            "currency": currency,
            # Most pay quarterly, some monthly
            "frequency": "MONTHLY" if rng.random() < 0.2 else "QUARTERLY",
        })
    return pd.DataFrame(rows)


def generate_dividend_events(securities: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    rows, event_num = [], 1
    for sec in securities.itertuples():
        step_months = 1 if sec.frequency == "MONTHLY" else 3
        base_rate = round(rng.uniform(0.05, 1.20), 4)
        month = rng.randint(1, step_months)  # stagger first payment month
        while month <= END_DATE.month:
            record_date = to_weekday(date(2026, month, rng.randint(5, 22)))
            pay_date = business_days_after(record_date, rng.randint(8, 15))
            if pay_date <= END_DATE:
                # Small rate changes over time, like real dividend raises
                rate = round(base_rate * rng.uniform(0.98, 1.05), 4)
                rows.append({
                    "event_id": f"EVT{event_num:04d}",
                    "security_id": sec.security_id,
                    "record_date": record_date,
                    "pay_date": pay_date,
                    "rate_per_share": rate,
                    "currency": sec.currency,
                })
                event_num += 1
            month += step_months
    return pd.DataFrame(rows)


def generate_positions(securities: pd.DataFrame, events: pd.DataFrame,
                       rng: random.Random) -> pd.DataFrame:
    """Each account holds a handful of securities. A position snapshot is
    written on every record date of a security the account holds."""
    accounts = [f"ACC{i:05d}" for i in range(1, NUM_ACCOUNTS + 1)]
    sec_ids = securities["security_id"].tolist()

    holdings = {}  # (account, security) -> starting quantity
    for acct in accounts:
        for sec in rng.sample(sec_ids, rng.randint(*HOLDINGS_PER_ACCOUNT)):
            holdings[(acct, sec)] = rng.choice([50, 100, 200, 250, 500, 1000, 1500, 2500, 5000])

    rows = []
    for (acct, sec), qty in holdings.items():
        sec_events = events[events["security_id"] == sec].sort_values("record_date")
        for ev in sec_events.itertuples():
            # Occasionally the client buys or sells between record dates
            if rng.random() < 0.15:
                qty = max(10, qty + rng.choice([-1, 1]) * rng.choice([25, 50, 100, 250]))
            rows.append({
                "account_id": acct,
                "security_id": sec,
                "as_of_date": ev.record_date,
                "quantity": qty,
            })
    return pd.DataFrame(rows)


def calculate_entitlements(positions, events, securities) -> pd.DataFrame:
    """What SHOULD be paid. The generator uses this to build clean payments.
    The pipeline will recalculate it independently in SQL."""
    df = positions.merge(
        events, left_on=["security_id", "as_of_date"],
        right_on=["security_id", "record_date"],
    ).merge(securities[["security_id", "isin"]], on="security_id")
    df["gross_amount"] = (df["quantity"] * df["rate_per_share"]).round(2)
    df["tax_withheld"] = df.apply(
        lambda r: round(r.gross_amount * US_WITHHOLDING_RATE, 2) if r.currency == "USD" else 0.0,
        axis=1,
    )
    df["net_amount"] = (df["gross_amount"] - df["tax_withheld"]).round(2)
    return df


def generate_payments(entitlements: pd.DataFrame, securities: pd.DataFrame,
                      rng: random.Random):
    """Start from perfect payments, then plant breaks. Returns the payments
    and a manifest recording every planted break."""
    manifest = []
    idx = list(entitlements.index)
    rng.shuffle(idx)

    # Pick a separate, non overlapping group of entitlements for each break
    picks, cursor = {}, 0
    for brk in ["MISSING_PAYMENT", "AMOUNT_MISMATCH", "DUPLICATE_PAYMENT", "LATE_PAYMENT"]:
        picks[brk] = set(idx[cursor:cursor + BREAK_COUNTS[brk]])
        cursor += BREAK_COUNTS[brk]

    payments = []
    for i, ent in entitlements.iterrows():
        key = {"account_id": ent.account_id, "isin": ent["isin"], "event_id": ent.event_id}

        if i in picks["MISSING_PAYMENT"]:
            manifest.append({**key, "break_type": "MISSING_PAYMENT"})
            continue

        pay = {
            "account_id": ent.account_id,
            "isin": ent["isin"],
            "pay_date": ent.pay_date,
            "gross_amount": ent.gross_amount,
            "tax_withheld": ent.tax_withheld,
            "net_amount": ent.net_amount,
            "currency": ent.currency,
        }

        if i in picks["AMOUNT_MISMATCH"]:
            # Off by more than the tolerance, either direction
            diff = round(rng.uniform(0.50, 75.00), 2) * rng.choice([-1, 1])
            pay["gross_amount"] = round(max(0.01, ent.gross_amount + diff), 2)
            rate = US_WITHHOLDING_RATE if ent.currency == "USD" else 0.0
            pay["tax_withheld"] = round(pay["gross_amount"] * rate, 2)
            pay["net_amount"] = round(pay["gross_amount"] - pay["tax_withheld"], 2)
            manifest.append({**key, "break_type": "AMOUNT_MISMATCH"})

        if i in picks["LATE_PAYMENT"]:
            pay["pay_date"] = ent.pay_date + timedelta(days=rng.randint(3, 20))
            manifest.append({**key, "break_type": "LATE_PAYMENT"})

        payments.append(pay)

        if i in picks["DUPLICATE_PAYMENT"]:
            payments.append(dict(pay))  # same payment arrives a second time
            manifest.append({**key, "break_type": "DUPLICATE_PAYMENT"})

    # Unexpected payments: money for an account and security with no position
    held = set(zip(entitlements["account_id"], entitlements["isin"]))
    accounts = sorted(entitlements["account_id"].unique())
    payable = entitlements.drop_duplicates("event_id")
    while sum(m["break_type"] == "UNEXPECTED_PAYMENT" for m in manifest) < BREAK_COUNTS["UNEXPECTED_PAYMENT"]:
        ev = payable.sample(1, random_state=rng.randint(0, 10**6)).iloc[0]
        acct = rng.choice(accounts)
        if (acct, ev["isin"]) in held:
            continue
        held.add((acct, ev["isin"]))
        gross = round(rng.choice([100, 200, 500, 1000]) * ev.rate_per_share, 2)
        tax = round(gross * US_WITHHOLDING_RATE, 2) if ev.currency == "USD" else 0.0
        payments.append({
            "account_id": acct, "isin": ev["isin"], "pay_date": ev.pay_date,
            "gross_amount": gross, "tax_withheld": tax,
            "net_amount": round(gross - tax, 2), "currency": ev.currency,
        })
        manifest.append({"account_id": acct, "isin": ev["isin"],
                         "event_id": ev.event_id, "break_type": "UNEXPECTED_PAYMENT"})

    # Shuffle so breaks aren't grouped together, then assign payment IDs
    rng.shuffle(payments)
    pay_df = pd.DataFrame(payments)
    pay_df.insert(0, "payment_id", [f"PAY{n:06d}" for n in range(1, len(pay_df) + 1)])
    pay_df["pay_date"] = pay_df["pay_date"].astype(str)  # ISO YYYY-MM-DD

    # Dirty data: messy ISINs and inconsistent date formats, like real feeds
    for row in pay_df.sample(frac=DIRTY_ISIN_RATE, random_state=1).index:
        isin = pay_df.at[row, "isin"]
        pay_df.at[row, "isin"] = rng.choice([isin.lower(), f" {isin} ", f"{isin} "])
    for row in pay_df.sample(frac=DIRTY_DATE_RATE, random_state=2).index:
        d = date.fromisoformat(pay_df.at[row, "pay_date"])
        pay_df.at[row, "pay_date"] = rng.choice([d.strftime("%d/%m/%Y"), d.strftime("%Y%m%d")])

    return pay_df, pd.DataFrame(manifest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reconciliation data")
    parser.add_argument("--out", default="data/landing", help="output folder for source files")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    securities = generate_securities(rng, fake)
    events = generate_dividend_events(securities, rng)
    positions = generate_positions(securities, events, rng)
    entitlements = calculate_entitlements(positions, events, securities)
    payments, manifest = generate_payments(entitlements, securities, rng)

    securities.drop(columns="frequency").to_csv(out / "securities.csv", index=False)
    events.to_csv(out / "dividend_events.csv", index=False)
    positions.to_csv(out / "positions.csv", index=False)
    payments.to_csv(out / "custodian_payments.csv", index=False)

    # The answer key is not a source file, so it goes beside the landing folder
    manifest.to_csv(out.parent / "break_manifest.csv", index=False)

    print(f"securities:          {len(securities):>6} rows")
    print(f"dividend_events:     {len(events):>6} rows")
    print(f"positions:           {len(positions):>6} rows")
    print(f"expected payments:   {len(entitlements):>6} (entitlements)")
    print(f"custodian_payments:  {len(payments):>6} rows")
    print("\nPlanted breaks:")
    for brk, n in manifest["break_type"].value_counts().items():
        print(f"  {brk:<20} {n:>4}")
    print(f"\nFiles written to {out.resolve()}")


if __name__ == "__main__":
    main()
