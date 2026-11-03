# QM FinTech Quant Workshop: Algorithmic Trading

A beginner-friendly starter kit for **backtesting trading strategies in Python** — real market data, a small engine you can read in one sitting, worked notebooks, an interactive app, and resources to keep going.

Led by **Adnan Salehin** (Lead SWE & VP, JPMorgan) · Organised by **QM FinTech Society**

> Educational use only — not financial advice. Price data from Yahoo Finance; no ownership is claimed.

---

## Start here — everything runs in your browser

**Nothing to install.** All you need is a Google account for [Google Colab](https://colab.research.google.com), a free notebook service that runs Python in the cloud.

👉 **Workshop page: [kolyamkl.github.io/qm-fintech-quant-workshop](https://kolyamkl.github.io/qm-fintech-quant-workshop/)**: every notebook opens in Colab in a **new tab**. (Buttons on this GitHub page open in the same tab; Cmd/Ctrl-click to open a new one.)

1. Open the main workshop notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/01_workshop_ma_crossover.ipynb)
2. If Colab warns that the notebook wasn't written by Google, click **Run anyway**.
3. Click **File → Save a copy in Drive**, so your changes are kept.
4. Run the first cell (**▶ Setup**). It downloads the workshop code and data into your Colab session.
5. Work top to bottom: click a cell and press **Shift + Enter** to run it.

**Just want to play, no code?** Try the 🌐 interactive app: **[qmfintech-workshop.streamlit.app](https://qmfintech-workshop.streamlit.app)**

### All notebooks

| Notebook | What it covers | Colab |
|---|---|---|
| `00_intro_to_algo_trading` | What algo trading is + key terms | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/00_intro_to_algo_trading.ipynb) |
| `01_workshop_ma_crossover` | **The main workshop**: data → signal → backtest → pitfalls | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/01_workshop_ma_crossover.ipynb) |
| `02_rsi_mean_reversion` | A second strategy: buy the dip, sell the rip | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/02_rsi_mean_reversion.ipynb) |
| `03_compare_with_library` | The same strategy in backtesting.py | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/03_compare_with_library.ipynb) |
| `04_project_trading_dashboard` | **Take-home project:** build your own trading dashboard and run it on your computer ([details](projects/trading_dashboard/)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/04_project_trading_dashboard.ipynb) |
| `05_portfolio_risk_rules` | Portfolios with weights, sector rotation, dual momentum, and risk rules in plain words: drawdown limits, Value at Risk, hold vs sell everything vs rotate through 2020 and 2022 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/05_portfolio_risk_rules.ipynb) |
| `06_ai_risk_agent` | **An AI agent that trades, checked by maths:** a language model proposes trades, fixed code checks every one, and the backtest scores it. Runs offline without a key | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/06_ai_risk_agent.ipynb) |
| `07_alpha_competition` | **Alpha, beta and the alpha competition:** how much of a return is just the market, why a big alpha can be luck, then write your own strategy and compete for the best alpha on data it was not built on ([interactive explainer](https://kolyamkl.github.io/qm-fintech-quant-workshop/alpha-beta.html)) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/07_alpha_competition.ipynb) |

## Workshop agenda (~1.5–2 hours)

| Part | Time |
|---|---|
| Intro: what is algorithmic trading? (`00`) | 10–15 min |
| Build and backtest a moving-average crossover (`01`) | ~60 min |
| Demo: the interactive app | ~10 min |
| If time allows: RSI mean reversion (`02`) | remaining |
| Follow-up session: risk rules (`05`) and the AI agent (`06`) | ~45 min each |
| Alpha, beta and the alpha competition (`07`) | ~45 min, then the competition runs until the deadline |
| Q&A | ~10 min |

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

Notebook `01` ends with a cell like this, ready to edit. To keep a strategy as a reusable file, start from `strategies/template.py`.

## What's in the repo

```
data/prices.csv        daily prices, 24 US ETFs & stocks, 2016 → 2026 (see data/README.md)
backtester/            the engine: data loading, run_backtest, metrics, charts
  portfolio.py         run_portfolio_backtest for several assets in chosen weights
  risk.py              portfolio limits, Value at Risk, and a day-by-day replay that obeys them
  agent.py             an AI risk manager whose every proposal passes the same checks
  alpha.py             alpha and beta by regression, and the alpha competition scorer
competition/           house strategies, entries/ (one file per team), leaderboard.md
scripts/leaderboard.py scores every entry the same way and writes competition/leaderboard.md
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

For several assets at once, `run_portfolio_backtest(closes, weights)` follows the same rules with **weights** (the share of money in each asset, the rest in cash), costs on every change of weight, and holdings that drift with prices between trades. `backtester/risk.py` turns limits such as *"never fall more than 15% from my peak"* into warning lines and replays a portfolio day by day under them; `backtester/agent.py` lets a language model propose the trades, but only the ones that pass the same fixed checks are made. Notebooks `05` and `06` walk through both.

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
