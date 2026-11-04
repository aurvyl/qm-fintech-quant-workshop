# QM FinTech Quant Workshop: Algorithmic Trading

A beginner-friendly starter kit for **backtesting trading strategies in Python** — real market data, a small engine you can read in one sitting, worked notebooks, an interactive app, and resources to keep going.

Led by **Adnan Salehin** (Lead SWE & VP, JPMorgan) · Organised by **QM FinTech Society**

> Educational use only — not financial advice. Price data from Yahoo Finance; no ownership is claimed.

---

## Start here — everything runs in your browser

**Nothing to install.** All you need is a Google account for [Google Colab](https://colab.research.google.com), a free notebook service that runs Python in the cloud.

👉 **Workshop page: [kolyamkl.github.io/qm-fintech-quant-workshop](https://kolyamkl.github.io/qm-fintech-quant-workshop/)**: every notebook opens in Colab in a **new tab**. (Buttons on this GitHub page open in the same tab; Cmd/Ctrl-click to open a new one.)

1. Open the first notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/00_why_test_a_trading_idea.ipynb)
2. If Colab warns that the notebook wasn't written by Google, click **Run anyway**.
3. Click **File → Save a copy in Drive**, so your changes are kept.
4. Run the first cell (**▶ Setup**). It downloads the workshop code and data into your Colab session.
5. Work top to bottom: click a cell and press **Shift + Enter** to run it.

**Just want to play, no code?** Try the 🌐 interactive app: **[qmfintech-workshop.streamlit.app](https://qmfintech-workshop.streamlit.app)**

### All notebooks

| Notebook | What it covers | Colab |
|---|---|---|
| `00_why_test_a_trading_idea` | **Session 1.** What a trading strategy is, and why we test it on old prices first | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/00_why_test_a_trading_idea.ipynb) |
| `01_returns_and_risk` | **Session 2.** How much did it make, and how bumpy was the ride? Returns, volatility, worst fall, Sharpe ratio | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/01_returns_and_risk.ipynb) |
| `02_backtest_honesty` | **Session 3.** How a backtest can lie: peeking at the future, costs, luck, train and test, walk-forward | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/02_backtest_honesty.ipynb) |
| `03_risk_limits_one_asset` | **Session 4.** Managing risk: limits in plain words, warning lines, value at risk, three reactions to a crash | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/03_risk_limits_one_asset.ipynb) |
| `04_first_strategy_ma_crossover` | **Session 5.** Your first strategy: moving-average crossover, tested with the honest tools | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/04_first_strategy_ma_crossover.ipynb) |
| `05_second_strategy_rsi` | **Session 5.** A second strategy: buy the dip, sell the rip, and whether it beats buy and hold | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/05_second_strategy_rsi.ipynb) |
| `06_portfolios` | **Session 6.** Portfolios: weights, sector rotation, dual momentum, volatility targeting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/06_portfolios.ipynb) |
| `07_portfolio_risk_rules` | **Session 6.** Risk rules for a portfolio: limits, value at risk, four reactions through 2020 and 2022, what if | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/07_portfolio_risk_rules.ipynb) |
| `08_alpha_beta_competition` | **Session 7.** Is the result real? Alpha, beta, t-stat and the alpha competition ([interactive explainer](https://kolyamkl.github.io/qm-fintech-quant-workshop/alpha-beta.html)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/08_alpha_beta_competition.ipynb) |
| `09_ai_risk_agent` | **Session 8.** An AI agent that reacts to a crash, checked by maths. Runs offline without a key | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/09_ai_risk_agent.ipynb) |
| `take_home_dashboard` | **Take-home project:** build your own trading dashboard and run it on your computer ([details](projects/trading_dashboard/)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/take_home_dashboard.ipynb) |
| `appendix_library_comparison` | **Appendix:** the same strategy in backtesting.py, and why the numbers differ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/appendix_library_comparison.ipynb) |

## Workshop agenda: eight sessions of about 60 minutes

Each session adds one idea to the last: why we test, how to measure, how a test can lie, how to limit losses, then strategies, portfolios, proving a result is real, and finally an AI agent.

| Session | Topic | Notebook |
|---|---|---|
| 1 | What a trading strategy is, and why we test it on old prices first | `00` |
| 2 | How much did it make, and how bumpy was the ride? | `01` |
| 3 | How a backtest can lie | `02` |
| 4 | Managing risk on one asset | `03` |
| 5 | Your first strategies | `04`, `05` |
| 6 | Portfolios and their risk rules | `06`, `07` |
| 7 | Is the result real? Alpha, beta and the competition (the competition then runs until the deadline) | `08` |
| 8 | An AI agent that reacts to a crash | `09` |

The app demo and the take-home dashboard fit anywhere after session 2.

## Write your own strategy

A strategy is just a function: prices in, positions out (`1` = hold, `0` = cash, `-1` = short). In Colab, write it straight into a notebook cell (after running **▶ Setup**) — no files to edit:

```python
from backtester import load_prices, run_backtest

def my_strategy(prices):
    close = prices["Close"]
    return (close > close.shift(20)).astype(int)   # replace with your idea

spy = load_prices("SPY")
result = run_backtest(spy, my_strategy(spy), cost_bps=5)
result.summary()   # strategy vs buy-and-hold
result.plot()      # portfolio value and drawdown
```

Notebook `04` ends with a cell like this, ready to edit. To keep a strategy as a reusable file, start from `strategies/template.py`.

## What's in the repo

```
data/prices.csv        daily prices, 24 US ETFs & stocks, 2016 → 2026 (see data/README.md)
backtester/            the engine: data loading, run_backtest, metrics, charts
  portfolio.py         run_portfolio_backtest for several assets in chosen weights
  risk.py              portfolio limits, Value at Risk, and a day-by-day replay that obeys them
  agent.py             an AI risk manager whose every proposal passes the same checks
  alpha.py             alpha and beta by regression, and the alpha competition scorer
  honesty.py           toy strategies with a known answer, best-of-many luck, and walk-forward
competition/           house strategies and entries/ (one file per team)
scripts/leaderboard.py scores every entry the same way on the held-back period (the organiser keeps the result private)
strategies/            ma_crossover, rsi_mean_reversion, and template.py for your own
notebooks/             the workshop, step by step
app/streamlit_app.py   interactive backtester
projects/trading_dashboard/   take-home project: starter app, self-checks, solution
docs/index.html        the workshop page (GitHub Pages)
docs/glossary.md       every term, explained in a sentence
docs/resources.md      exercises, reading list, careers and interview prep
scripts/download_data.py   rebuilds data/prices.csv from Yahoo Finance
tests/                 checks that the engine computes correctly (run: pytest)
```

## How the engine works (in one paragraph)

`run_backtest` takes your daily positions, **shifts them by one day** (a decision made at today's close can only earn tomorrow's return — skipping this is the classic *look-ahead bias*), multiplies by the asset's daily returns, subtracts a trading cost whenever the position changes, and compounds the result from $10,000. Everything is compared with simply buying and holding. It deliberately leaves out slippage, borrowing fees for shorts, position sizing and multi-asset portfolios — those are exercises in [`docs/resources.md`](docs/resources.md).

For several assets at once, `run_portfolio_backtest(closes, weights)` follows the same rules with **weights** (the share of money in each asset, the rest in cash), costs on every change of weight, and holdings that drift with prices between trades. `backtester/risk.py` turns limits such as *"never fall more than 15% from my peak"* into warning lines and replays a portfolio day by day under them; `backtester/agent.py` lets a language model propose the trades, but only the ones that pass the same fixed checks are made. Notebooks `03`, `07` and `09` walk through both.

## Optional: run it on your own laptop

Not needed for the workshop — everything above runs in the cloud. Use this if you want to keep building after it.

You need **Python 3.11 or newer** ([python.org/downloads](https://www.python.org/downloads/)) and **git**.

```bash
git clone https://github.com/kolyamkl/qm-fintech-quant-workshop.git
cd qm-fintech-quant-workshop
python -m venv .venv
```

Activate the environment:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`

Then install and run:

```bash
pip install -r requirements.txt
jupyter lab                         # opens the notebooks in your browser
streamlit run app/streamlit_app.py  # opens the interactive app
```

(On macOS you may need `python3` instead of `python`.)

## Deploy the app (organisers)

The hosted app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) (free). With the repo pushed to GitHub:

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns the repo.
2. Click **Create app** and choose repository `kolyamkl/qm-fintech-quant-workshop`, branch `main`, main file path `app/streamlit_app.py`.
3. Click **Deploy**. The live app is at <https://qmfintech-workshop.streamlit.app> and redeploys automatically on every push to `main`.

After a push that changes `backtester/` or `strategies/`, click **Manage app → Reboot app** on the live site. Streamlit reloads `app/` by itself but keeps the old versions of those two folders in memory, so without a reboot the app can crash with an `ImportError`.

## Licence

Code: [MIT](LICENSE). Data: see [`data/README.md`](data/README.md).
