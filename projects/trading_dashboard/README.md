# Take-home project: build your own trading dashboard

A web app that shows how an investment performed, how risky it was, and how much of its movement is just the market (CAPM). You build every number yourself, then run the app on your own computer from a terminal.

**Start with the notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/take_home_dashboard.ipynb)

## What the finished app shows

- **Six headline numbers:** annual return, annual volatility, hit rate, Sharpe ratio, Sortino ratio and max drawdown.
- **Performance tab:** the growth of $1 and a rolling 63-day Sharpe ratio.
- **Risk tab:** drawdown and rolling 30-day volatility.
- **CAPM tab:** alpha, beta and R squared against a benchmark such as SPY.
- **Data tab:** the summary table and the prices.

## The files

```
analytics.py           ✏️ your calculations: copy your functions here from the take-home notebook
app.py                 ✏️ the web page, with five TODOs to fill in
checks.py              the ✅ / ❌ checks used by the notebook
solution/analytics.py  reference answers for every function
solution/app.py        the finished app
```

## Steps

1. **Write the functions** in notebook `take_home_dashboard` (Colab or `jupyter lab`). Each step has a hint and a check.
2. **Get the code on your computer.** You need Python 3.11 or newer and git:

   ```bash
   git clone https://github.com/kolyamkl/qm-fintech-quant-workshop.git
   cd qm-fintech-quant-workshop
   python -m venv .venv
   source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Add your functions:** paste them into `analytics.py`, replacing each `raise NotImplementedError(...)` line.
4. **Run the app** from the repo folder. It opens in your browser at `http://localhost:8501`:

   ```bash
   streamlit run projects/trading_dashboard/app.py
   ```

5. **Fill in the TODOs** in `app.py`. Save, then refresh the browser page. Stop the app with **Ctrl + C**.

Want to see the finished version first? `streamlit run projects/trading_dashboard/solution/app.py`

The notebook's Part 2 has the full walkthrough and ideas to take it further.
