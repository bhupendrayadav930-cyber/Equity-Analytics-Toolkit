"""
ingest.py — normalises three trade-history sources (a broker FIFO P&L export
and two manual trade journals) into ONE table with a "Bucket" column.

Buckets:
  ETF-Momentum        - ETF trades (SILVERBEES, GOLDBEES, ITBEES, ...)
  Stock-Setup         - stock swing trades from the setup journal, tagged by
                        setup (S1-S10, DARVAS, ...)
  Stock-Discretionary - broker-export stock trades without a setup tag
  T1T2T3-System       - partial-exit (T1/T2/T3) system, flagged RETIRED

Usage:
  python ingest.py --swing SWING.csv --t1t2t3 T1T2T3.csv --broker Trading_Insights.xlsx

Writes trades_master.csv next to this script, which app.py reads.
Raw exports contain personal financial data and are git-ignored.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Setup tags with negative historical expectancy -- flagged, not deleted,
# so they stay visible in the dashboard but are visually called out.
RETIRED_STOCK_TAGS = {"S1", "S2", "S4", "S6", "S7", "S10", "DARVAS"}
ETF_SYMBOLS = {"SILVERBEES", "GOLDBEES", "ITBEES", "NIFTYBEES", "GOLDIETF"}


def clean_num(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return s
    s = str(s).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def ingest_swing_journal(path):
    """Setup journal - S1-S10 tagged individual stock trades."""
    df = pd.read_csv(path, header=1)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Share Name"]).copy()
    df["PnL"] = df["Profit / loss (In rupees)"].apply(clean_num)
    df["HoldDays"] = df["Trade_Time (Days)"].apply(clean_num)
    df = df.dropna(subset=["PnL"])

    df["first_tag"] = df["Strategy"].fillna("Untagged").str.split(",").str[0].str.strip()

    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Selling Date"], errors="coerce", dayfirst=True),
        "Symbol": df["Share Name"],
        "Bucket": "Stock-Setup",
        "SubTag": df["first_tag"],
        "PnL": df["PnL"],
        "HoldDays": df["HoldDays"],
        "Source": "SWING journal",
        "Retired": df["first_tag"].isin(RETIRED_STOCK_TAGS),
        "Notes": df.get("Details", ""),
    })
    return out


def ingest_t1t2t3(path):
    """T1/T2/T3 partial-exit journal - whole system flagged retired."""
    df = pd.read_csv(path, header=1)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["SYMBOL"])
    df = df[df["SYMBOL"] != "FINACIAL YEAR 2025-26"]
    df["PnL"] = df["PNL"].apply(clean_num)
    df = df.dropna(subset=["PnL"])
    df = df[df["PnL"] != 0]

    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], errors="coerce"),
        "Symbol": df["SYMBOL"],
        "Bucket": "T1T2T3-System",
        "SubTag": df.get("CHART", "").fillna("Untagged"),
        "PnL": df["PnL"],
        "HoldDays": np.nan,
        "Source": "T1T2T3 journal",
        "Retired": True,  # whole system flagged
        "Notes": "",
    })
    return out


def ingest_broker_trading_insights(path):
    """Broker-generated FIFO realized trades - Equity-Delivery segment."""
    df = pd.read_excel(path, sheet_name="Trade", header=7)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["Symbol"])
    df["PnL"] = pd.to_numeric(df["PnL"], errors="coerce")
    df["Entry Date"] = pd.to_datetime(df["Entry Date"], errors="coerce")
    df["Exit Date"] = pd.to_datetime(df["Exit Date"], errors="coerce")
    df["HoldDays"] = (df["Exit Date"] - df["Entry Date"]).dt.days
    df = df.dropna(subset=["PnL"])

    def bucket_for(sym):
        return "ETF-Momentum" if str(sym).upper() in ETF_SYMBOLS else "Stock-Discretionary"

    out = pd.DataFrame({
        "Date": df["Exit Date"],
        "Symbol": df["Symbol"],
        "Bucket": df["Symbol"].apply(bucket_for),
        "SubTag": df["Symbol"],
        "PnL": df["PnL"],
        "HoldDays": df["HoldDays"],
        "Source": "Broker Trading Insights",
        "Retired": False,
        "Notes": "",
    })
    return out


def build_master(swing_path, t1t2t3_path, broker_path,
                 out_path=Path(__file__).resolve().parent / "trades_master.csv"):
    frames = [
        ingest_swing_journal(swing_path),
        ingest_t1t2t3(t1t2t3_path),
        ingest_broker_trading_insights(broker_path),
    ]
    master = pd.concat(frames, ignore_index=True)
    master = master.sort_values("Date")
    master.to_csv(out_path, index=False)
    print(f"Wrote {len(master)} trades to {out_path}")
    print(master.groupby("Bucket")["PnL"].agg(["count", "sum", "mean"]))
    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build trades_master.csv from raw exports")
    parser.add_argument("--swing", required=True, help="Setup journal CSV")
    parser.add_argument("--t1t2t3", required=True, help="T1/T2/T3 journal CSV")
    parser.add_argument("--broker", required=True, help="Broker 'Trading Insights' XLSX")
    args = parser.parse_args()
    build_master(args.swing, args.t1t2t3, args.broker)
