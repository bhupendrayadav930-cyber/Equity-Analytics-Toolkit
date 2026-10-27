"""High-water-mark billing dashboard (Streamlit).

    streamlit run app.py

Ships with DEMO clients only. Point it at your own CSVs by placing
clients.csv / bills.csv in a local `data/` folder (git-ignored).
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import Bill, Client, compute_ledger, firm_summary

HERE = Path(__file__).resolve().parent
LOCAL, SAMPLE = HERE / "data", HERE / "sample_data"

st.set_page_config(page_title="HWM Billing Engine", layout="wide")


def _src() -> Path:
    return LOCAL if (LOCAL / "clients.csv").exists() else SAMPLE


@st.cache_data
def load(src: str):
    c = pd.read_csv(Path(src) / "clients.csv")
    b = pd.read_csv(Path(src) / "bills.csv", parse_dates=["from_date", "to_date"])
    return c, b


def to_client(row) -> Client:
    return Client(client_id=row.client_id, name=row.name, risk_bearer=row.risk_bearer,
                  client_capital=float(row.client_capital), client_commission_pct=float(row.client_commission_pct),
                  owner_capital=float(row.owner_capital), owner_commission_pct=float(row.owner_commission_pct),
                  max_loss_pct=float(row.max_loss_pct), status=row.status)


def to_bills(df: pd.DataFrame):
    out = []
    for r in df.itertuples():
        share = getattr(r, "client_share_of_capital_change", None)
        out.append(Bill(r.from_date.date(), r.to_date.date(), float(r.gross_pnl), float(r.total_charges),
                        float(r.brokerage), float(r.fixed_charge), float(r.capital_change),
                        float(r.payment_received), None if pd.isna(share) else float(share)))
    return out


def rs(x: float) -> str:
    return f"₹{x:,.0f}"


clients_df, bills_df = load(str(_src()))
if "extra_bills" not in st.session_state:
    st.session_state.extra_bills = []
all_bills = pd.concat([bills_df, pd.DataFrame(st.session_state.extra_bills)], ignore_index=True) \
    if st.session_state.extra_bills else bills_df

clients = [to_client(r) for r in clients_df.itertuples()]
ledgers = {c.client_id: compute_ledger(c, to_bills(all_bills[all_bills.client_id == c.client_id]))
           for c in clients}

st.title("High-Water-Mark Performance Fee & Billing Engine")
if _src() == SAMPLE:
    st.caption("Showing DEMO data (fictional clients). No real client information is included in this repository.")

page = st.sidebar.radio("View", ["Firm dashboard", "Client ledger", "Invoice", "New bill"])

# ---------------------------------------------------------------- firm view
if page == "Firm dashboard":
    s = firm_summary(clients, ledgers)
    k = st.columns(4)
    k[0].metric("Active accounts", s["clients"])
    k[1].metric("Capital under management", rs(s["capital_under_management"]))
    k[2].metric("Performance fees earned", rs(s["performance_fees"]))
    k[3].metric("Fees receivable", rs(s["fees_receivable"]))
    k = st.columns(4)
    k[0].metric("Gross trading P&L", rs(s["gross_pnl"]))
    k[1].metric("Brokerage", rs(s["brokerage"]))
    k[2].metric("Fixed charges recovered", rs(s["fixed_charges"]))
    k[3].metric("Owner capital at risk (guarantees)", rs(s["owner_capital_at_risk"]))

    rows = []
    for c in clients:
        led = ledgers[c.client_id]
        last = led.iloc[-1]
        rows.append({"Client": c.name, "Type": c.risk_bearer, "Capital": c.capital if not len(led) else last.capital + last.capital_change,
                     "Commission": f"{last.commission_rate:.1%}", "Cum. client return": last.cum_client_return,
                     "Cum. ROI %": last.cum_roi_pct, "HWM deficit": last.hwm_deficit, "Fees earned": led.performance_fee.sum(),
                     "Receivable": last.fee_receivable, "Current risk bearer": last.risk_bearer})
    st.dataframe(pd.DataFrame(rows).set_index("Client").style.format(
        {"Capital": "₹{:,.0f}", "Cum. client return": "₹{:,.0f}", "Cum. ROI %": "{:.2f}",
         "HWM deficit": "₹{:,.0f}", "Fees earned": "₹{:,.0f}", "Receivable": "₹{:,.0f}"}), width="stretch")
    fees = pd.concat({c.name: ledgers[c.client_id].set_index("to_date")["performance_fee"] for c in clients}, axis=1).fillna(0)
    st.subheader("Monthly performance fees by client")
    st.bar_chart(fees)

# ---------------------------------------------------------------- client view
elif page == "Client ledger":
    name = st.sidebar.selectbox("Client", [c.name for c in clients])
    c = next(x for x in clients if x.name == name)
    led = ledgers[c.client_id]
    last = led.iloc[-1]
    k = st.columns(5)
    k[0].metric("Cum. client return", rs(last.cum_client_return), f"{last.cum_roi_pct:.2f}% cum. ROI")
    k[1].metric("HWM deficit to recover", rs(last.hwm_deficit))
    k[2].metric("Fees earned", rs(led.performance_fee.sum()))
    k[3].metric("Fixed charges carried fwd", rs(last.fixed_carried_fwd))
    k[4].metric("Risk bearer", last.risk_bearer)
    st.subheader("Cumulative client return vs. high-water-mark deficit")
    st.line_chart(led.set_index("to_date")[["cum_client_return", "hwm_deficit"]])
    show = ["bill_no", "from_date", "to_date", "capital", "gross_pnl", "total_charges", "fixed_applied",
            "fixed_carried_fwd", "net_pnl", "hwm_deficit", "commission_rate", "performance_fee",
            "client_return", "cum_client_return", "payment_received", "fee_receivable", "cycle_roi_pct",
            "cum_roi_pct", "risk_bearer", "remarks"]
    money = {k: "₹{:,.0f}" for k in show if k not in ("bill_no", "from_date", "to_date", "commission_rate",
                                                       "cycle_roi_pct", "cum_roi_pct", "risk_bearer", "remarks")}
    st.dataframe(led[show].set_index("bill_no").style.format({**money, "commission_rate": "{:.1%}"}), width="stretch")
    st.download_button("Download ledger (CSV)", led.to_csv(index=False), f"{c.client_id}_ledger.csv")

# ---------------------------------------------------------------- invoice
elif page == "Invoice":
    name = st.sidebar.selectbox("Client", [c.name for c in clients])
    c = next(x for x in clients if x.name == name)
    led = ledgers[c.client_id]
    n = st.sidebar.selectbox("Bill no.", led.bill_no.tolist()[::-1])
    r = led[led.bill_no == n].iloc[0]
    st.subheader(f"P&L report & invoice — {c.name} ({c.client_id}), bill #{n}")
    st.caption(f"Period {r.from_date:%d-%b-%Y} to {r.to_date:%d-%b-%Y} · account size {rs(r.capital)}")
    prev_hwm = led[led.bill_no == n - 1].hwm_deficit.iloc[0] if n > 1 else 0.0
    inv = pd.DataFrame([
        ("A  Realised P&L (gross)", r.gross_pnl),
        ("B  Brokerage, taxes & charges", -r.total_charges),
        ("C  Fixed / service charges applied", -r.fixed_applied),
        ("D  Net P&L = A − B − C", r.net_pnl),
        ("E  Carried-forward loss (HWM) before this cycle", -prev_hwm),
        ("F  Performance fee @ " + f"{r.commission_rate:.1%} on profit above HWM", -r.performance_fee),
        ("G  Net return to client = D − F", r.client_return),
        ("H  Cumulative client return", r.cum_client_return),
        ("I  Fee receivable after payments", r.fee_receivable),
    ], columns=["Component", "Amount (₹)"]).set_index("Component")
    st.table(inv.style.format("{:,.0f}"))
    if r.remarks:
        st.info(r.remarks)
    st.download_button("Download invoice (CSV)", inv.to_csv(), f"{c.client_id}_bill_{n}.csv")

# ---------------------------------------------------------------- new bill
else:
    name = st.sidebar.selectbox("Client", [c.name for c in clients])
    c = next(x for x in clients if x.name == name)
    led = ledgers[c.client_id]
    start = (led.to_date.iloc[-1] + timedelta(days=1)) if len(led) else date.today().replace(day=1)
    with st.form("bill"):
        a, b_ = st.columns(2)
        f = a.date_input("From", start)
        t = a.date_input("To", start + timedelta(days=29))
        gross = a.number_input("Gross realised P&L (₹)", value=0.0, step=1000.0)
        charges = b_.number_input("Total charges (₹)", min_value=0.0, step=100.0)
        brok = b_.number_input("of which brokerage (₹)", min_value=0.0, step=100.0)
        fixed = b_.number_input("Fixed / service charge (₹)", value=950.0, step=50.0)
        capchg = a.number_input("Capital added (+) / withdrawn (−) after this cycle", value=0.0, step=10000.0)
        pay = b_.number_input("Fee payment received (₹)", min_value=0.0, step=100.0)
        if st.form_submit_button("Preview bill"):
            try:
                new = dict(client_id=c.client_id, from_date=pd.Timestamp(f), to_date=pd.Timestamp(t), gross_pnl=gross,
                           total_charges=charges, brokerage=brok, fixed_charge=fixed, capital_change=capchg,
                           payment_received=pay, client_share_of_capital_change=None)
                Bill(f, t, gross, charges, brok, fixed, capchg, pay)        # validation
                st.session_state.extra_bills.append(new)
                st.success("Bill added for this session — see Client ledger / Invoice.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.caption("Bills added here live in this browser session only; the ledger is recomputed from scratch each time.")
