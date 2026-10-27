# Equity Analytics Toolkit

Python tools for Indian equities and portfolio operations: a **fundamental quality screen plus a multi-strategy backtesting engine** for the Nifty 500, a **high-water-mark performance-fee and client-billing engine**, and a **Streamlit dashboard** that breaks realised P&L down by strategy.

All three projects follow the same idea: test an investment process against data with the biases stated openly, rather than judging it by feel.

| Project | What it answers | Stack |
|---|---|---|
| [`backtest_engine/`](backtest_engine/) | Which quality stocks pass a fundamental screen, and which entry/exit rules have historically worked on them, on which timeframe? | Python, pandas, NumPy, SciPy, yfinance |
| [`hwm_billing_engine/`](hwm_billing_engine/) | How much performance fee is due on each account, once loss carry-forwards, deferred charges and split-capital commission are applied? | Python, pandas, Streamlit, pytest |
| [`trade_journal/`](trade_journal/) | Across a real account, which strategy bucket is adding value and which is destroying it? | Python, pandas, Streamlit |

---

## 1. Backtesting engine: fundamental screen + multi-strategy backtest

### Objective
Start from the Nifty 500, keep only financially sound businesses, then compare 7 rule-based entry strategies across 5 timeframes on the same capital, cost and exit assumptions. The output is a Strategy × Timeframe performance matrix.

### How it works

```
Nifty 500 list ──► Phase 1: Fundamental screen ──► Phase 2: Price data ──► Phase 3: Signals ──► Phase 4: Backtest + analytics
 (NSE CSV)          ROE ≥ 15%, D/E ≤ 1.0,           1H bars via yfinance,     7 strategies          shared capital pool,
                    EPS growth ≥ 0, MCap ≥ ₹500 Cr   resampled to 2H/4H/1D/1W  per timeframe         costs, CAGR/Sharpe/MaxDD
```

1. **Universe & fundamental filter** (`universe.py`): loads Nifty 500 constituents and pulls ROE, debt/equity, EPS growth and market cap from yfinance. Stocks with missing data are excluded, not passed. A pluggable provider interface allows other data sources (a screener.in provider is included).
2. **Multi-timeframe data** (`data_providers.py`): fetches one base interval (1H) per stock, caches it to Parquet, and resamples locally to 2H/4H/1D/1W with bins anchored to the NSE 09:15 open. Adapters for the Dhan and Upstox broker APIs share the same interface (credentials come from environment variables).
3. **Signal generation** (`strategies/`): pivot-based pattern detection (`scipy.signal.argrelextrema`) for Head & Shoulders, Double Top/Bottom, Trendline Breakout, Triangles, Flags and Cup & Handle, plus a 50/200 EMA crossover baseline.
4. **Portfolio simulation** (`backtest.py`): event-driven, long-only by default. Each Strategy × Timeframe run starts with ₹10,00,000, shared across all stocks in the universe. It sizes 5% of current equity per position with a 20-position cap, charges a 0.15% transaction cost on each side, earns the risk-free rate on idle cash, and exits on a structural stop or a trailing 20-EMA.
5. **Performance analytics** (`analytics.py`): total return, CAGR, maximum drawdown, Sharpe ratio (7% risk-free rate, annualised per timeframe), win rate, average win/loss and profit factor. Results are written to `output/comparison_matrix.csv`, and current setups go to `output/active_setups.csv` for manual review.

### Methodology safeguards and known limitations
- **Look-ahead bias is controlled.** A pivot at bar *i* can be used only from bar *i + order*, once enough later bars exist to confirm it.
- **Capital is realistic.** One portfolio is shared across all stocks in a run, so the backtest cannot deploy more capital than it has.
- **Costs are included.** Brokerage, STT and other charges are approximated at 0.15% per side and can be changed in `config.py`.
- **Survivorship bias.** The universe is *today's* Nifty 500, so stocks that left the index are missing from history.
- **Fundamentals are not point-in-time.** The screen uses current ratios. The result answers "how would these rules have worked on today's quality names", not what an investor could have known at each date.
- **Data depth.** yfinance limits 1H history to about 730 days, so intraday conclusions rest on a short sample.

### Run it
```bash
cd backtest_engine
python smoke_test.py                                   # offline check on synthetic data (no network)
python -m techno_funda_engine.main --limit 15          # live run on the first 15 Nifty 500 names
```
All thresholds (screen criteria, costs, capital, pattern tolerances) live in `techno_funda_engine/config.py`.

---

## 2. High-water-mark performance-fee & billing engine

### Objective
Calculate performance fees and client statements for managed accounts the way PMS and fund back offices do. Fees are earned **only on profits above the high-water mark**, and every rupee of a client's earlier losses must be recovered first.

### Billing rules (per cycle)
1. **Net P&L** = gross realised P&L − brokerage, STT and other charges.
2. **High-water mark first.** Losses carry forward, and no fee is charged until the account is back above its previous peak. The fee is charged only on the excess.
3. **Deferred fixed charges.** Service charges are recovered only once the HWM deficit is cleared and the client keeps at least ₹1,000 of the cycle's profit. Unrecovered charges carry forward rather than being waived.
4. **Split-capital accounts.** When capital is split between a client tranche and an owner tranche, the commission is the capital-weighted average of the two rates, re-weighted whenever capital is added or withdrawn.
5. **Risk-bearer switch.** On owner-guaranteed accounts, downside risk passes to the client once cumulative returns cover the guaranteed max-loss.
6. **Return metrics and receivables.** Each cycle reports cycle ROI, estimated monthly ROI, cumulative ROI on average capital, and fees billed less payments received.

### Origin and validation
This engine started as a production **Excel/VBA** tool and was ported to pure Python so it is testable and portable. On the original ledgers, the Python engine matched the VBA output field for field: 9/9 on a guaranteed account and 24/24 on a split-capital account.

The port also fixes one issue in the VBA. Whenever a ledger was recalculated, the VBA re-applied the *current* weighted commission to every *historical* bill. The Python engine applies the rate that was in force during each period.

The repo contains **fictional demo clients only**. Credentials and personal data from the original tool are not included.

### Run it
```bash
cd hwm_billing_engine
pytest test_engine.py            # 11 tests encoding each billing rule
streamlit run app.py             # firm dashboard, client ledger, invoice, new-bill form
```

---

## 3. Trade journal: strategy-level P&L attribution (Streamlit)

### Objective
A single account often holds several "strategies" at once. The dashboard separates realised trades into buckets and scores each one, which shows where returns actually come from.

### How it works
- `ingest.py` normalises three differently formatted sources (a broker FIFO P&L export and two manual journals) into one tidy table.
- `app.py` (Streamlit) shows, for each bucket and for each setup tag: trade count, total P&L, win rate, average win/loss and **expectancy per trade**. It also shows the P&L effect of dropping retired setups, has a form to log new trades with a written review, and a filterable trade log.

The repo ships with **synthetic sample data** (`sample_data/trades_sample.csv`), so the app runs straight after cloning. No real account data is included.

### Run it
```bash
cd trade_journal
streamlit run app.py
# Optional: build from your own exports
python ingest.py --swing SWING.csv --t1t2t3 T1T2T3.csv --broker Trading_Insights.xlsx
```

---

## Installation
Requires Python 3.10+.
```bash
git clone bhupendrayadav930-cyber
cd equity-analytics-toolkit
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Repository structure
```
equity-analytics-toolkit/
├── backtest_engine/
│   ├── data/ind_nifty500list.csv        # NSE Nifty 500 constituents (public)
│   ├── smoke_test.py                    # offline end-to-end test
│   └── techno_funda_engine/
│       ├── config.py                    # every tunable parameter
│       ├── universe.py                  # Phase 1: fundamental screen
│       ├── data_providers.py            # Phase 2: data adapters, cache, resampler
│       ├── pivots.py                    # pivot detection (look-ahead-safe)
│       ├── strategies/                  # Phase 3: 7 strategies
│       ├── backtest.py                  # Phase 4: portfolio simulation
│       ├── analytics.py                 # Phase 4: performance metrics
│       ├── scanner.py                   # current-setup watchlist
│       └── main.py                      # pipeline entry point
├── hwm_billing_engine/
│   ├── engine.py                        # HWM fee / billing ledger (pure functions)
│   ├── app.py                           # Streamlit: firm dashboard, ledgers, invoices
│   ├── test_engine.py                   # 11 rule tests
│   └── sample_data/                     # fictional demo clients & bills
├── trade_journal/
│   ├── app.py                           # Streamlit dashboard
│   ├── ingest.py                        # multi-source trade normaliser
│   └── sample_data/trades_sample.csv    # synthetic demo data
├── requirements.txt
└── LICENSE
```

## Disclaimer
This is for research and education only and is not investment advice. Backtested results are hypothetical and do not guarantee future performance.

## Author
**Bhupendra Yadav**: [LinkedIn](https://www.linkedin.com/in/bhupendra-yadav07/) · Bhupendrayadav930@gmail.com
