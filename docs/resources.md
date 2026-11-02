# Resources: keep going after the workshop

1. [Exercises](#1-exercises)
2. [Reading list](#2-reading-list)
3. [How to get good at quant](#3-how-to-get-good-at-quant)
4. [Interviews and the job](#4-interviews-and-the-job)

---

## 1. Exercises

Roughly easy → hard. Each builds on the code in this repo.

**Looking for a bigger project?** Build and publish your own trading dashboard, step by step: notebook [`04_project_trading_dashboard`](https://colab.research.google.com/github/kolyamkl/qm-fintech-quant-workshop/blob/main/notebooks/04_project_trading_dashboard.ipynb).

**Warm-up**
1. Run the MA crossover on every ticker in the dataset (`backtester.available_tickers()`). Which does it work best on? Worst?
2. Run the 5/20 crossover with 0 and 20 bps costs on QQQ. How much of the return do costs eat?
3. Plot the RSI of `TLT` and mark every day it went below 30.

**Build**
4. **Momentum strategy:** in `strategies/template.py`, hold when the price is higher than it was 120 days ago. Compare with the MA crossover.
5. **Stop-loss:** modify a strategy so it exits if the price falls 10% below the entry price.
6. **Combine signals:** only buy RSI dips when the 200-day trend is up (price above its 200-day MA).
7. **Go short:** let the MA crossover return `-1` instead of `0` in downtrends. Does shorting help?

**Research**
8. **Robustness:** instead of the single best MA settings, find settings that are *good on most tickers*. Why is that a better sign than one great result?
9. **Walk-forward test:** tune MA settings on 2016–2018, test on 2019; tune on 2016–2019, test on 2020; and so on. Stitch the test periods together.

**Extend the engine**
10. Add **slippage** (e.g. a fixed extra cost per trade) to `run_backtest`.
11. Add a **borrowing fee** for short positions (e.g. 1% a year, charged for every day you hold `-1`). Short sellers pay this in real life.
12. Add **position sizing**: allow positions between 0 and 1 (e.g. 0.5 = half invested).
13. Make a **multi-asset** backtest: hold the 3 sector ETFs with the best 6-month return, rebalanced monthly. Notebook `05` builds this with `run_portfolio_backtest`; try other lookbacks and compare with `recent_period`.

**Risk rules and AI agents** (notebooks `05` and `06`)
14. **Your own risk manager:** write a function that takes the day's `context` and returns weights, and beat `rotate_manager` on Sharpe without more days over the daily limit.
15. **A cheaper agent:** only ask the AI model when `risk.unusual_moves` shows a z-score below -3. How many calls do you save, and does the result change?
16. **The look-ahead test:** run the Gemini agent with `anonymous=True` and without. If the named run is much better, what does that tell you?
17. **Audit the explanations:** collect every `unsupported_numbers` flag across a replay. Which kinds of numbers does the model make up?

---

## 2. Reading list

**Start here**
- Ernest P. Chan — *Quantitative Trading: How to Build Your Own Algorithmic Trading Business* (2nd ed., Wiley, 2021). Practical, beginner-friendly.
- Rishi K. Narang — *Inside the Black Box: A Simple Guide to Systematic Investing* (Wiley). How quant firms actually work, no maths needed.
- Larry Harris — *Trading and Exchanges: Market Microstructure for Practitioners* (Oxford University Press, 2003). How markets work under the hood.

**Next steps**
- Ernest P. Chan — *Algorithmic Trading: Winning Strategies and Their Rationale* (Wiley, 2013). Mean reversion and momentum in depth.
- Yves Hilpisch — *Python for Finance* (2nd ed., O'Reilly, 2018).
- Stefan Jansen — *Machine Learning for Algorithmic Trading* (2nd ed., Packt, 2020) — with a free code repository: <https://github.com/stefan-jansen/machine-learning-for-trading>
- John C. Hull — *Options, Futures, and Other Derivatives* (Pearson). The standard derivatives textbook.

**Advanced**
- Marcos López de Prado — *Advances in Financial Machine Learning* (Wiley, 2018). Rigorous on backtest overfitting.

**Free online**
- MIT OpenCourseWare — *Topics in Mathematics with Applications in Finance* (18.S096): <https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/>
- Jake VanderPlas — *Python Data Science Handbook* (free online): <https://jakevdp.github.io/PythonDataScienceHandbook/>
- pandas — *10 minutes to pandas*: <https://pandas.pydata.org/docs/user_guide/10min.html>
- Khan Academy — Statistics and probability: <https://www.khanacademy.org/math/statistics-probability>
- QuantConnect documentation (a free backtesting platform): <https://www.quantconnect.com/docs>
- backtesting.py documentation: <https://kernc.github.io/backtesting.py/>

---

## 3. How to get good at quant

**The four skills that matter**
1. **Programming** — Python (pandas, numpy) for research; C++ or Java are common for production trading systems. Write code every week.
2. **Probability and statistics** — expected value, distributions, hypothesis testing, regression. This is the language of quant interviews and research.
3. **Maths** — linear algebra and calculus; stochastic calculus if you're aiming at derivatives pricing.
4. **Markets** — how exchanges, orders and the main asset classes work. Read financial news with a "why did that move?" mindset.

**Build things people can see**
- Turn one of the exercises above into a clean GitHub project with a README explaining your idea, method and honest results — including what *didn't* work.
- Try a competition: Kaggle (<https://www.kaggle.com/competitions>) regularly hosts finance competitions from trading firms.
- Solve puzzles: Jane Street publishes a monthly puzzle (<https://www.janestreet.com/puzzles/>).

**Habits**
- Be sceptical of your own results — try to break your backtest before you believe it.
- Keep a research log: what you tried, why, and what happened.
- Join societies (like this one!), attend talks, and ask practitioners questions.

---

## 4. Interviews and the job

> *Draft — to be reviewed and extended by Adnan Salehin.*

**Common quant roles**
| Role | What they do | Emphasis |
|---|---|---|
| **Quant researcher** | Finds and tests trading ideas and models | Statistics, maths, research |
| **Quant trader** | Runs strategies, manages risk, makes real-time decisions | Probability, quick thinking, markets |
| **Quant developer / SWE** | Builds the trading systems, data pipelines and tools | Software engineering, performance |

**What interviews usually test**
- **Mental maths** — fast arithmetic under time pressure. Practise: <https://arithmetic.zetamac.com/>
- **Probability and brainteasers** — dice, cards, expected value, conditional probability.
- **Coding** — data structures and algorithms. Practise: <https://leetcode.com/>
- **Statistics / ML** — for research roles: regression, overfitting, model evaluation.
- **Markets and your projects** — be ready to explain a project end to end, including its weaknesses.

**Classic interview prep books**
- Xinfeng Zhou — *A Practical Guide to Quantitative Finance Interviews*.
- Timothy Falcon Crack — *Heard on the Street: Quantitative Questions from Wall Street Job Interviews*.
- Mark Joshi, Nick Denson, Andrew Downes — *Quant Job Interview Questions and Answers*.

**Tips**
- Think out loud — interviewers care about your reasoning as much as the answer.
- When unsure, estimate first, then refine.
- Apply early: many firms run spring weeks and summer internships with deadlines well before the summer.
