"""Tests encoding the billing rules of the original VBA ledger."""
from datetime import date

import pytest

from engine import Bill, Client, compute_ledger, firm_summary, service_charge_for_cycle


def bill(m, gross, charges=0.0, brok=0.0, fixed=0.0, pay=0.0, cap=0.0, share=None):
    return Bill(date(2026, m, 1), date(2026, m, 28), gross, charges, brok, fixed, cap, pay, share)


def client(**kw):
    base = dict(client_id="C1", name="Demo", risk_bearer="Client", client_capital=500_000,
                client_commission_pct=20.0)
    base.update(kw)
    return Client(**base)


def test_simple_profit_pays_fee_on_net():
    led = compute_ledger(client(), [bill(1, 10_000, charges=1_000, brok=600)])
    r = led.iloc[0]
    assert r.net_pnl == 9_000 and r.performance_fee == pytest.approx(1_800)
    assert r.client_return == pytest.approx(7_200) and r.stt_other == 400


def test_no_fee_until_high_water_mark_recovered():
    led = compute_ledger(client(), [bill(1, -20_000), bill(2, 15_000), bill(3, 10_000)])
    assert list(led.hwm_deficit) == [20_000, 5_000, 0]
    assert list(led.performance_fee) == [0, 0, pytest.approx(1_000)]   # 20% of (10k - 5k)


def test_fixed_charge_deferred_while_recovering_hwm():
    led = compute_ledger(client(), [bill(1, -5_000, fixed=950), bill(2, 5_500, fixed=950),
                                    bill(3, 10_000, fixed=950)])
    assert led.fixed_applied.tolist()[:2] == [0, 0]
    assert led.fixed_carried_fwd.tolist()[:2] == [950, 1_900]
    assert led.fixed_applied.iloc[2] == 2_850
    assert led.remarks.iloc[1] == "Profit used for HWM recovery; fixed charge on hold."
    assert led.remarks.iloc[2] == "Previous fixed charges recovered."


def test_partial_fixed_charge_recovery():
    led = compute_ledger(client(), [bill(1, 1_500, fixed=950), bill(2, 1_200, fixed=950)])
    # bill 1: surplus 1,500 > 1,000 -> full 950 applied; bill 2: 1,200 >= 950 -> applied
    assert led.fixed_applied.tolist() == [950, 950]
    led2 = compute_ledger(client(), [bill(1, 1_100, fixed=2_000)])
    assert led2.fixed_applied.iloc[0] == 1_100 and led2.fixed_carried_fwd.iloc[0] == 900


def test_both_account_weighted_commission_and_reweighting():
    c = client(risk_bearer="Both", client_capital=300_000, client_commission_pct=30,
               owner_capital=100_000, owner_commission_pct=10)
    assert c.commission_rate == pytest.approx(0.25)
    led = compute_ledger(c, [bill(1, 10_000, cap=200_000, share=0), bill(2, 10_000)])
    # owner tranche grows to 300k -> (300k*30 + 300k*10)/600k = 20%
    assert led.commission_rate.tolist() == [pytest.approx(0.25), pytest.approx(0.20)]
    assert led.capital.tolist() == [400_000, 600_000]


def test_risk_bearer_switches_after_threshold():
    c = client(risk_bearer="Owner", max_loss_pct=2)     # threshold 2% of 500k = 10k
    led = compute_ledger(c, [bill(1, 5_000), bill(2, 10_000)])
    assert led.risk_bearer.tolist() == ["Owner", "Client"]


def test_receivable_rolls_forward():
    led = compute_ledger(client(), [bill(1, 10_000), bill(2, 5_000, pay=1_500)])
    assert led.fee_receivable.tolist() == [pytest.approx(2_000), pytest.approx(1_500)]


def test_overlapping_periods_rejected():
    with pytest.raises(ValueError):
        compute_ledger(client(), [bill(1, 1), Bill(date(2026, 1, 15), date(2026, 2, 10), 1, 0, 0)])


def test_brokerage_above_charges_rejected():
    with pytest.raises(ValueError):
        Bill(date(2026, 1, 1), date(2026, 1, 31), 1_000, 100, 200)


def test_service_charge_cycle():
    start = date(2026, 4, 1)
    prior = [Bill(date(2026, 3, 1), date(2026, 3, 31), 0, 0, 0, 950)]
    assert service_charge_for_cycle(date(2026, 4, 30), prior, start, 1_200) == 1_200
    prior.append(Bill(date(2026, 4, 1), date(2026, 4, 30), 0, 0, 0, 1_200))
    assert service_charge_for_cycle(date(2026, 5, 31), prior, start, 1_200) == 0


def test_firm_summary_owner_risk():
    a = client(client_id="A", risk_bearer="Owner", max_loss_pct=10)
    b = client(client_id="B", risk_bearer="Both", client_capital=100_000, owner_capital=200_000,
               owner_commission_pct=10, max_loss_pct=10)
    ledgers = {"A": compute_ledger(a, [bill(1, -1_000)]), "B": compute_ledger(b, [bill(1, 2_000)])}
    s = firm_summary([a, b], ledgers)
    assert s["owner_capital_at_risk"] == pytest.approx(50_000 + 20_000)
    assert s["capital_under_management"] == 800_000
