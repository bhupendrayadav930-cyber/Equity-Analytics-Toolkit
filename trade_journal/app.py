"""
app.py — Unified Trade Tracker (Streamlit)

Splits one brokerage account's realised trades into strategy "buckets"
(ETF momentum, tagged stock setups, discretionary trades, a partial-exit
system) and scores each bucket on win rate, average win/loss and
expectancy, so capital can be allocated to what actually works.

Run:  streamlit run app.py
Data: reads trades_master.csv (built by ingest.py from real exports) if it
      exists, otherwise falls back to the synthetic sample in sample_data/.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path

from ingest import RETIRED_STOCK_TAGS

HERE = Path(__file__).resolve().parent
MASTER_PATH = HERE / "trades_master.csv"
SAMPLE_PATH = HERE / "sample_data" / "trades_sample.csv"

st.set_page_config(page_title="Unified Trade Tracker", layout="wide")
st.title("📒 Unified Trade Tracker")
st.caption("Strategy-level attribution of realised P&L: which bucket of trades is actually adding value?")


def load_master() -> pd.DataFrame:
    if MASTER_PATH.exists():
        path = MASTER_PATH
    else:
        path = SAMPLE_PATH
        st.info("Showing SYNTHETIC sample data. Run `python ingest.py ...` to build "
                "trades_master.csv from your own broker/journal exports.")
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Retired"] = df["Retired"].astype(str).str.lower().eq("true")
    return df


df = load_master()

# ---------------------------------------------------------------
# TOP-LEVEL BUCKET SCORECARD
# ---------------------------------------------------------------
st.subheader("Performance by Bucket — who's actually making the money?")

def bucket_stats(g):
    wins = g[g["PnL"] > 0]
    losses = g[g["PnL"] < 0]
    win_rate = len(wins) / len(g) * 100 if len(g) else 0
    avg_win = wins["PnL"].mean() if len(wins) else 0
    avg_loss = losses["PnL"].mean() if len(losses) else 0
    expectancy = (len(wins) / len(g)) * avg_win + (len(losses) / len(g)) * avg_loss if len(g) else 0
    return pd.Series({
        "Trades": len(g),
        "Total PnL": g["PnL"].sum(),
        "Win Rate %": round(win_rate, 1),
        "Avg Win": round(avg_win, 0),
        "Avg Loss": round(avg_loss, 0),
        "Expectancy/Trade": round(expectancy, 0),
    })

summary = df.groupby("Bucket")[["PnL"]].apply(bucket_stats).sort_values("Total PnL", ascending=False)
st.dataframe(summary.style.format({"Total PnL": "₹{:,.0f}", "Avg Win": "₹{:,.0f}", "Avg Loss": "₹{:,.0f}", "Expectancy/Trade": "₹{:,.0f}"}), width="stretch")

total_pnl = df["PnL"].sum()
retired_pnl = df[df["Retired"]]["PnL"].sum()
col1, col2, col3 = st.columns(3)
col1.metric("Total Realized P&L (all buckets)", f"₹{total_pnl:,.0f}")
col2.metric("P&L from RETIRED setups/systems", f"₹{retired_pnl:,.0f}", delta=None)
col3.metric("P&L if retired setups were skipped", f"₹{total_pnl - retired_pnl:,.0f}")

st.markdown("---")

# ---------------------------------------------------------------
# DRILL-DOWN: Stock setup tags (S1-S10 etc.)
# ---------------------------------------------------------------
st.subheader("Stock Setup Tags — which setups have positive expectancy")
stock_setups = df[df["Bucket"] == "Stock-Setup"]
if not stock_setups.empty:
    tag_stats = stock_setups.groupby("SubTag")[["PnL"]].apply(bucket_stats).sort_values("Total PnL", ascending=False)
    tag_stats["Status"] = tag_stats.index.map(lambda t: "🔴 RETIRED" if t in RETIRED_STOCK_TAGS else "🟢 Active")
    st.dataframe(tag_stats, width="stretch")
else:
    st.info("No Stock-Setup trades loaded yet.")

st.markdown("---")

# ---------------------------------------------------------------
# LOG A NEW TRADE (manual entry, going forward)
# ---------------------------------------------------------------
st.subheader("Log a New Trade")
st.caption("Log each trade at exit, with a short written review of the thesis and outcome.")

with st.form("new_trade_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        trade_date = st.date_input("Exit Date", value=date.today())
        symbol = st.text_input("Symbol")
        bucket = st.selectbox("Bucket", ["ETF-Momentum", "Stock-Setup", "Stock-Discretionary", "T1T2T3-System", "BankNifty-Supertrend"])
    with c2:
        sub_tag = st.text_input("Setup Tag (e.g. S8, or ETF name, or 'Sell-side')")
        pnl = st.number_input("Realized P&L (₹)", step=100.0, format="%.2f")
        hold_days = st.number_input("Holding Days", min_value=0, step=1)
    with c3:
        is_retired_tag = sub_tag.strip().upper() in RETIRED_STOCK_TAGS
        if is_retired_tag:
            st.error(f"'{sub_tag}' is a RETIRED setup (negative historical expectancy).")
        notes = st.text_area("Trade Review", height=120,
                              placeholder="e.g. Entered on breakout above resistance with volume confirmation. Exited at target 1. Setup: clean trend continuation.")

    submitted = st.form_submit_button("Save Trade")
    if submitted:
        if not symbol:
            st.warning("Symbol is required.")
        else:
            new_row = pd.DataFrame([{
                "Date": pd.Timestamp(trade_date),
                "Symbol": symbol.upper(),
                "Bucket": bucket,
                "SubTag": sub_tag if sub_tag else "Untagged",
                "PnL": pnl,
                "HoldDays": hold_days,
                "Source": "Manual entry",
                "Retired": is_retired_tag or bucket == "T1T2T3-System",
                "Notes": notes,
            }])
            df_updated = pd.concat([df, new_row], ignore_index=True)
            df_updated.to_csv(MASTER_PATH, index=False)
            st.success(f"Saved {symbol} — ₹{pnl:,.0f}. Refresh to see it in the stats above.")

st.markdown("---")

# ---------------------------------------------------------------
# RAW LOG
# ---------------------------------------------------------------
st.subheader("Full Trade Log")
show_retired = st.checkbox("Include retired setups/systems", value=True)
display_df = df if show_retired else df[~df["Retired"]]
st.dataframe(display_df.sort_values("Date", ascending=False), width="stretch")
