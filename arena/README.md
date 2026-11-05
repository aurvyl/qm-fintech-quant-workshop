# Agent arena

A web page where teams paste guidance for the AI risk manager and compete on one leaderboard. Every entry is replayed on the same prices, in parallel, from 2023 to the last day of data, and scored by alpha against the market.

## What teams see and do

1. Open the arena link, enter the team code you were given and a team name.
2. Write up to 1,000 characters of guidance, for example "Prefer gold over bonds when you rotate. Keep tech in small steps."
3. Press **Run my agent**. The entry joins a queue, runs with the others, and appears on the leaderboard, ranked by its overall score, with its growth line.
4. Each team has 100 runs and one run at a time. The latest finished run is the one that counts.

## How it is kept fair

- The model sees dates and tickers hidden (`ASSET_A`, `ASSET_B`, ...). Tickers in a team's guidance are hidden the same way.
- Every proposal still passes `check_plan`, the same fixed gate as the rule-based manager. Guidance cannot change the rules.
- Each entry has a cap of 120 model calls and 20 minutes. Past the cap the portfolio simply holds.
- Guidance and the model's written explanations stay private to the team until the organiser reveals them.
- Once a run finishes, every team can see its growth line, its buy and sell markers and its holdings history. Live progress is private until the run finishes.
- The arena reaches the model through the workshop model server, so the key, the model and its limits stay there.
- The leaderboard shows the bars to beat: buy and hold, the rule-based rotation and the market.

Teams are ranked by an overall score from 0 to 100. It weighs five measures: alpha a year (40%), total return (25%), max drawdown (15%, where a smaller fall scores higher), Sharpe ratio (10%) and the alpha t-stat (10%). On each measure a team scores its percentile among all the teams, so a score moves as more teams run. A missing measure counts as the lowest. Alpha a year and the t-stat are still shown next to each team, with a tick at 2 or more.

## Try it on your computer, free

```
pip install -r requirements.txt -r arena/requirements.txt
python -m arena.server --demo
```

Open http://127.0.0.1:8800. The demo uses a stand-in that is **not AI** (`arena/demo_model.py`) that reads a few plain words such as "cautious", "aggressive", "gold" and "bonds". Team codes are `demo-team-01` to `demo-team-08`, and the organiser code is `demo-organiser`.

## Deploy for the workshop (organiser)

1. Give the arena its own code on the model server, with a bigger daily cap, and redeploy it from the private server folder. Add `WORKSHOP_ARENA_CODES=<arena code>` to the `workshop-model` secret next to the values already there, then `modal deploy modal_app.py`.
2. Create the arena secret, typing the values yourself:

   ```
   modal secret create workshop-arena ARENA_PROXY_URL=<model server url> ARENA_PROXY_CODE=<the same arena code> ARENA_TEAM_CODES=team-code-001,team-code-002 ARENA_ADMIN_CODE=<organiser code>
   ```

3. From the root of this repository: `modal deploy arena/modal_app.py`. Modal prints the arena URL.
4. Give each team its code and the URL.

Organiser controls (the page has none, so use a terminal):

```
curl -s -X POST <arena url>/api/admin -H 'Content-Type: application/json' -d '{"code":"<organiser code>","action":"close"}'
```

`action` is `open`, `close`, `reveal` (show everyone's guidance and decisions) or `hide`.

After the session: `modal app stop workshop-arena`, and rotate or remove the codes.

## Cost and limits

An entry makes about 20 to 60 model calls. A class of 20 teams using a handful of runs each is a few thousand calls, which is pence with the workshop model. The arena code on the model server still allows only 3,000 requests a day in total, so that is the real limit on how many runs happen. The arena runs as one always-on container (about the cost of a small web server) with 6 parallel workers and a queue of at most 60 entries. State is saved to a Modal Dict, and unfinished entries run again if the container restarts.
