"""High-water-mark performance-fee and client-billing engine.

A pure-Python port of a production Excel/VBA billing ledger. Each client
account is billed per period ("cycle"). For every bill:

1. Net trading P&L = gross P&L - total charges (brokerage + STT/other).
2. HWM first: losses carried forward from earlier cycles (the high-water
   mark deficit) must be recovered before any fee is earned.
3. Fixed / service charges are deferred, not waived. They are only taken
   when the cycle's profit, after recovering the HWM deficit, exceeds a
   minimum client-retention threshold (default INR 1,000); unpaid charges
   carry forward.
4. Performance fee = commission % x profit ABOVE the high-water mark.
5. Client return, cumulative return, cycle / monthly / cumulative ROI and
   receivables (fees billed - payments received) are rolled forward.
6. Risk-bearer switch: on "Owner"-guaranteed accounts, downside risk moves
   to the client once cumulative client return reaches max-loss % x capital.

For "Both" accounts (capital split between a client tranche and an owner
tranche) the commission is the capital-weighted average of the two rates
and is re-weighted whenever capital is added or withdrawn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

MIN_CLIENT_RETENTION = 1000.0      # INR the client must keep before fixed charges are recovered
DEFAULT_FIXED_CHARGE = 950.0       # default per-cycle service charge


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Client:
    client_id: str
    name: str
    risk_bearer: str                  # "Client" | "Owner" | "Both"
    client_capital: float
    client_commission_pct: float      # e.g. 20.0 for 20%
    owner_capital: float = 0.0        # only for "Both"
    owner_commission_pct: float = 0.0
    max_loss_pct: float = 0.0         # e.g. 10.0 -> owner absorbs losses up to 10% of capital
    join_date: Optional[date] = None
    status: str = "Active"            # Active | Paused | Closed

    def __post_init__(self):
        if self.risk_bearer not in ("Client", "Owner", "Both"):
            raise ValueError(f"risk_bearer must be Client/Owner/Both, got {self.risk_bearer!r}")
        if self.risk_bearer != "Both":
            self.owner_capital, self.owner_commission_pct = 0.0, 0.0

    @property
    def capital(self) -> float:
        return self.client_capital + self.owner_capital

    @property
    def commission_rate(self) -> float:
        """Capital-weighted commission as a fraction (0.20 = 20%)."""
        if self.capital <= 0:
            return 0.0
        return (self.client_capital * self.client_commission_pct
                + self.owner_capital * self.owner_commission_pct) / self.capital / 100

    def apply_capital_change(self, total_change: float, client_share: Optional[float] = None) -> None:
        """Add (+) or withdraw (-) capital. For 'Both' accounts, `client_share`
        is the portion belonging to the client tranche; the rest is the owner's."""
        if self.risk_bearer == "Both":
            cs = total_change if client_share is None else client_share
            self.client_capital += cs
            self.owner_capital += total_change - cs
        else:
            self.client_capital += total_change


@dataclass
class Bill:
    from_date: date
    to_date: date
    gross_pnl: float
    total_charges: float
    brokerage: float
    fixed_charge: float = DEFAULT_FIXED_CHARGE
    capital_change: float = 0.0        # applied AFTER this cycle (affects the next bill's capital)
    payment_received: float = 0.0
    client_share_of_capital_change: Optional[float] = None

    def __post_init__(self):
        if self.brokerage > self.total_charges:
            raise ValueError("Brokerage cannot exceed total charges")
        if self.to_date < self.from_date:
            raise ValueError("to_date is before from_date")


# --------------------------------------------------------------------------- #
# Ledger computation
# --------------------------------------------------------------------------- #
def compute_ledger(client: Client, bills: List[Bill],
                   min_retention: float = MIN_CLIENT_RETENTION) -> pd.DataFrame:
    """Recompute the full ledger from scratch (pure function: the client
    object passed in is not mutated)."""
    c = Client(**{k: getattr(client, k) for k in client.__dataclass_fields__})
    rows = []
    prev_hwm = prev_unpaid_fix = prev_cum = prev_due = 0.0
    risk_bearer = c.risk_bearer
    inv_history: List[float] = []
    last_to: Optional[date] = None

    for n, b in enumerate(sorted(bills, key=lambda x: x.from_date), start=1):
        if last_to is not None and b.from_date <= last_to:
            raise ValueError(f"Bill {n}: period overlaps the previous bill (ends {last_to})")
        last_to = b.to_date

        inv_amt = c.capital
        comm_rate = c.commission_rate
        stt_other = b.total_charges - b.brokerage
        net_trade = b.gross_pnl - b.total_charges

        # --- fixed charges: HWM is cleared first, then the client must keep > min_retention
        owed_fix = prev_unpaid_fix + b.fixed_charge
        surplus = net_trade - prev_hwm
        remark = ""
        if surplus > min_retention:
            if surplus >= owed_fix:
                applied_fix, unpaid_fix = owed_fix, 0.0
                if applied_fix > b.fixed_charge:
                    remark = "Previous fixed charges recovered."
            else:
                applied_fix, unpaid_fix = surplus, owed_fix - surplus
                remark = "Partial fixed charge recovered."
        else:
            applied_fix, unpaid_fix = 0.0, owed_fix
            if net_trade > 0:
                remark = "Profit used for HWM recovery; fixed charge on hold."

        net_pnl = net_trade - applied_fix

        # --- high-water mark and performance fee
        if net_pnl < 0:
            new_hwm, fee = prev_hwm + abs(net_pnl), 0.0
        elif prev_hwm > 0:
            if net_pnl > prev_hwm:
                fee, new_hwm = (net_pnl - prev_hwm) * comm_rate, 0.0
            else:
                fee, new_hwm = 0.0, prev_hwm - net_pnl
        else:
            fee, new_hwm = net_pnl * comm_rate, 0.0

        client_ret = net_pnl - fee
        cum_ret = prev_cum + client_ret

        # --- risk-bearer switch for owner-guaranteed accounts
        if (c.risk_bearer == "Owner" and risk_bearer == "Owner"
                and cum_ret >= inv_amt * c.max_loss_pct / 100):
            risk_bearer = "Client"

        due = prev_due + fee - b.payment_received

        # --- return metrics
        days = max((b.to_date - b.from_date).days, 1)
        inv_history.append(inv_amt)
        avg_inv = sum(inv_history) / len(inv_history)
        cycle_roi = client_ret / inv_amt * 100 if inv_amt else 0.0

        rows.append({
            "bill_no": n, "from_date": b.from_date, "to_date": b.to_date, "capital": inv_amt,
            "capital_change": b.capital_change, "gross_pnl": b.gross_pnl,
            "total_charges": b.total_charges, "brokerage": b.brokerage, "stt_other": stt_other,
            "fixed_billed": b.fixed_charge, "fixed_applied": applied_fix, "fixed_carried_fwd": unpaid_fix,
            "net_pnl": net_pnl, "hwm_deficit": new_hwm, "risk_bearer": risk_bearer,
            "commission_rate": comm_rate, "performance_fee": fee, "client_return": client_ret,
            "cum_client_return": cum_ret, "payment_received": b.payment_received, "fee_receivable": due,
            "days": days, "cycle_roi_pct": round(cycle_roi, 2), "avg_capital": round(avg_inv, 0),
            "est_monthly_roi_pct": round(cycle_roi / days * 30, 2),
            "cum_roi_pct": round(cum_ret / avg_inv * 100, 2) if avg_inv else 0.0,
            "remarks": remark,
        })

        prev_hwm, prev_unpaid_fix, prev_cum, prev_due = new_hwm, unpaid_fix, cum_ret, due
        if b.capital_change:
            c.apply_capital_change(b.capital_change, b.client_share_of_capital_change)

    return pd.DataFrame(rows)


def service_charge_for_cycle(to_date: date, prior_bills: List[Bill], global_cycle_start: Optional[date],
                             global_amount: float, default: float = DEFAULT_FIXED_CHARGE) -> float:
    """Firm-wide service-charge cycle: charge `global_amount` once per client
    in the first bill ending on/after the cycle start, then 0 until the next
    cycle. Falls back to `default` when no global cycle is configured."""
    if not global_cycle_start or global_amount <= 0:
        return default
    if to_date < global_cycle_start:
        return default
    already = any(b.to_date >= global_cycle_start and b.fixed_charge > 0 for b in prior_bills)
    return 0.0 if already else global_amount


# --------------------------------------------------------------------------- #
# Firm-level roll-up (owner dashboard)
# --------------------------------------------------------------------------- #
def firm_summary(clients: List[Client], ledgers: dict) -> dict:
    active = [c for c in clients if c.status != "Closed"]
    owner_risk = cum = 0.0
    for c in active:
        led = ledgers.get(c.client_id)
        has = led is not None and len(led)
        bearer = led["risk_bearer"].iloc[-1] if has else c.risk_bearer
        capital = float(led["capital"].iloc[-1] + led["capital_change"].iloc[-1]) if has else c.capital
        cum += capital
        owner_share = (c.owner_capital / c.capital) if c.capital else 0.0   # opening tranche mix
        if bearer == "Owner":
            owner_risk += capital * c.max_loss_pct / 100
        elif bearer == "Both":
            owner_risk += capital * owner_share * c.max_loss_pct / 100
    all_led = pd.concat([l for l in ledgers.values() if len(l)], ignore_index=True) if ledgers else pd.DataFrame()
    def s(col):
        return float(all_led[col].sum()) if len(all_led) else 0.0
    return {
        "clients": len(active),
        "capital_under_management": cum,
        "gross_pnl": s("gross_pnl"), "brokerage": s("brokerage"), "stt_other": s("stt_other"),
        "performance_fees": s("performance_fee"), "fixed_charges": s("fixed_applied"),
        "fees_receivable": float(sum(l["fee_receivable"].iloc[-1] for l in ledgers.values() if len(l))),
        "owner_capital_at_risk": owner_risk,
    }
