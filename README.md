# CBB Totals Model

A production-ready NCAA Men's College Basketball Over/Under game total projection system combining physics-based modeling, machine learning, and real-time odds comparison.

---

## Overview

This system projects the expected total score (combined points) for every NCAA Men's Basketball game on a given day, then compares those projections against current sportsbook lines to identify edges. It uses a two-layer approach:

1. **Baseline model** — A possessions × efficiency physics engine (KenPom-style)
2. **ML ensemble** — XGBoost + LightGBM + Ridge regression trained on historical features
3. **Ensemble blend** — Weighted combination of both layers with a composite confidence score

The result is a ranked table of today's games sorted by projected edge (differential between model and market).

---

## Quick Start

### 1. Clone or download the project

```bash
git clone https://github.com/yourname/cbb-totals.git
cd cbb-totals
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API key(s)
```

At minimum you need `THE_ODDS_API_KEY` for odds data. Everything else is optional.

### 4. Initialize the database

The database is created automatically on first run:

```bash
python main.py run
```

### 5. Collect historical data (for ML training)

```bash
python main.py collect --start 2023-11-01 --end 2024-03-31
```

### 6. Train ML models

```bash
python main.py train
```

### 7. Run the dashboard

```bash
python main.py dashboard
```

---

## API Keys

### The Odds API (required for market lines)

- Sign up at https://the-odds-api.com
- Free tier: 500 requests/month (sufficient for daily use during the season)
- The model uses the `totals` market for `basketball_ncaab`
- Set `THE_ODDS_API_KEY` in your `.env` file

**Free tier tip:** The odds refresh runs 3x/day. At ~1 request per call, that is ~90 requests/month for a full season — well within the free tier.

### Optional: Discord Webhook

Create a webhook in your Discord server settings and set `DISCORD_WEBHOOK_URL` in `.env`. When enabled, the model sends alerts when edges exceed the configured threshold.

### Optional: Database (PostgreSQL)

By default SQLite is used (`data/cbb_totals.db`). For production / shared deployments, set `DATABASE_URL` to a PostgreSQL connection string.

---

## Daily Automation Setup

### Windows Task Scheduler

1. Open Task Scheduler → Create Basic Task
2. Trigger: Daily at 9:00 AM
3. Action: Start a program
   - Program: `C:\path\to\python.exe`
   - Arguments: `C:\path\to\cbb-totals\main.py run`
   - Start in: `C:\path\to\cbb-totals\`

For intraday refreshes, create additional tasks at 12:00 PM and 5:00 PM using the `refresh-odds` command.

### Linux / macOS cron

```bash
# Edit crontab
crontab -e

# Add these lines:
0 9  * * * cd /path/to/cbb-totals && /usr/bin/python3 main.py run >> logs/cron.log 2>&1
0 12 * * * cd /path/to/cbb-totals && /usr/bin/python3 main.py refresh-odds >> logs/cron.log 2>&1
0 17 * * * cd /path/to/cbb-totals && /usr/bin/python3 main.py refresh-odds >> logs/cron.log 2>&1
```

### APScheduler Daemon

The project ships with a built-in scheduler that runs as a foreground process:

```bash
python main.py schedule
```

Or use the scheduler module directly:

```bash
python scheduler.py
```

The scheduler reads run times from `config/config.yaml` under `scheduler:`.

### GitHub Actions

Create `.github/workflows/daily-run.yml`:

```yaml
name: CBB Totals Daily Run

on:
  schedule:
    # 9 AM ET = 14:00 UTC (adjust for DST)
    - cron: '0 14 * * *'
    - cron: '0 17 * * *'
    - cron: '0 22 * * *'
  workflow_dispatch:

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run pipeline
        env:
          THE_ODDS_API_KEY: ${{ secrets.THE_ODDS_API_KEY }}
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: |
          if [ "${{ github.event.schedule }}" = "0 14 * * *" ]; then
            python main.py run
          else
            python main.py refresh-odds
          fi

      - name: Upload outputs
        uses: actions/upload-artifact@v4
        with:
          name: cbb-edges-${{ github.run_id }}
          path: outputs/*.csv
```

Add `THE_ODDS_API_KEY` as a GitHub Actions secret in your repository settings.

---

## Project Structure

```
cbb-totals/
├── config/
│   ├── config.yaml              # Master configuration
│   └── team_name_mapping.yaml   # Team name variant → canonical mapping
├── data/
│   ├── raw/                     # Cached API responses (Torvik CSVs, etc.)
│   ├── processed/               # Cleaned/processed datasets
│   └── cbb_totals.db            # SQLite database (auto-created)
├── src/
│   ├── utils/
│   │   ├── logger.py            # Rich + rotating file logging
│   │   ├── db.py                # SQLAlchemy database manager
│   │   └── alerts.py            # Discord webhook alert manager
│   ├── data/
│   │   ├── base_adapter.py      # Abstract adapter interfaces + factory
│   │   ├── espn_adapter.py      # ESPN public API (schedule + stats)
│   │   ├── torvik_adapter.py    # Bart Torvik T-Rank data
│   │   ├── odds_adapter.py      # The Odds API v4
│   │   └── team_normalizer.py   # Fuzzy team name resolution
│   ├── features/
│   │   ├── feature_engineering.py  # Full feature vector builder
│   │   └── recent_form.py          # Rolling stats calculator
│   ├── models/
│   │   ├── baseline_model.py    # Physics-based possessions × efficiency model
│   │   ├── ml_model.py          # XGBoost + LightGBM + Ridge trainer/predictor
│   │   └── ensemble.py          # Weighted blend + confidence scoring
│   ├── pipeline/
│   │   ├── daily_pipeline.py    # Main daily orchestration pipeline
│   │   └── edge_calculator.py   # Differential computation + output formatting
│   └── backtest/
│       └── backtest.py          # Historical backtest engine + report
├── models/                      # Saved ML model files (.joblib)
├── outputs/                     # CSV exports (timestamped edge tables)
├── logs/                        # Rotating log files
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # pytest test suite
├── dashboard/
│   └── app.py                   # Streamlit 6-page dashboard
├── main.py                      # CLI entry point
├── scheduler.py                 # APScheduler daemon
├── requirements.txt
└── .env.example
```

---

## How the Model Works

### Baseline Model (35% weight by default)

The baseline model uses a KenPom-style possessions × efficiency formula:

```
projected_possessions = blend(home_tempo, away_tempo)
                       with regression toward mean when tempos differ by > 5

home_points_per_possession = (home_adj_OE / league_avg_OE)
                           × (away_adj_DE / league_avg_DE)
                           × league_avg_PPP

home_score = home_ppp × projected_possessions + home_court_adj (+1.75 pts)
away_score = away_ppp × projected_possessions - home_court_adj (-1.75 pts)
total      = home_score + away_score
```

Additional adjustments: turnover rate correction (reduces PPP), free throw rate correction (inflates PPP slightly).

**Data source:** Primarily Bart Torvik's adjusted efficiency ratings (adj_oe, adj_de, adj_tempo). Falls back to ESPN raw stats.

### ML Ensemble (65% weight by default)

Three models are trained on the full feature vector (~100 features):

| Model     | Weight | Strengths |
|-----------|--------|-----------|
| XGBoost   | 40%    | Non-linear patterns, feature interactions |
| LightGBM  | 40%    | Fast, handles sparse features well |
| Ridge     | 20%    | Stable, good out-of-sample regression |

**Training:** Time-aware (no data leakage — models only see data from before each game). Rolling window cross-validation used to evaluate before full training.

### Ensemble Blend

```
final_total = 0.35 × baseline + 0.65 × ml_ensemble
```

When baseline and ML disagree by more than 8 points, a conservative blend is applied (both components regressed 30% toward the league average before blending).

### Feature Categories

| Category | Example Features |
|----------|-----------------|
| Tempo/Pace | adj_tempo, expected_possessions, tempo_differential |
| Offense | adj_oe, efg_pct, three_p_pct, three_pa_rate, tov_rate |
| Defense | adj_de, opp_efg_pct, forced_tov_rate, drb_rate |
| Matchup | off_vs_def ratio, tov_environment, ft_environment |
| Recent Form | last_3/5/10 avg points, scoring trend, pace trend |
| Environment | neutral_site, rest_days, conference_game, sos |

---

## Output Interpretation

### Edge Table

```
# | Away          | Home          | Mkt   | Model | Diff   | Edge  | Conf | Poss
1 | Kansas        | Baylor        | 145.5 | 151.3 | +5.8   | OVER  | 0.71 | 71.2
2 | Auburn        | Alabama       | 152.0 | 146.1 | -5.9   | UNDER | 0.65 | 69.8
```

- **Mkt Total:** The sportsbook's consensus over/under line
- **Model Total:** This system's projection
- **Diff:** Model − Market. Positive = lean OVER, negative = lean UNDER
- **Edge:** OVER (model thinks game goes over) or UNDER
- **Conf:** Composite confidence score (0–1)
- **Poss:** Projected possessions per team

### Confidence Score

The confidence score (0–1) is a composite of four factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Data completeness | 25% | What fraction of features had real values |
| Model agreement | 35% | How closely baseline + ML + sub-models agree |
| Edge magnitude | 25% | Larger edge → higher score |
| Line movement | 15% | Line moving toward projection = confidence boost |

A conservative discount (0.85×) is applied to all scores to avoid overconfidence.

**Interpretation:**
- ≥ 0.75 → Very High confidence
- 0.55–0.75 → High
- 0.35–0.55 → Medium
- < 0.35 → Low

### Edge Buckets

Games are grouped by edge size for performance tracking:
- **0–2 pts:** Essentially a pick — skip or use for small plays
- **2–4 pts:** Small edge — light action
- **4–6 pts:** Solid edge — standard unit
- **6–8 pts:** Strong edge — consider larger play
- **8+ pts:** Maximum edge — rare, verify data quality

---

## Backtesting

### Running a Backtest

```bash
# Default: last 90 days
python main.py backtest

# Custom date range
python main.py backtest --start 2023-01-01 --end 2023-03-31
```

### What the Backtest Measures

- **MAE / RMSE:** Raw projection accuracy against actual totals
- **ATS record by bucket:** Win/loss/push when betting each edge tier
- **ROI by bucket:** Simulated ROI at -110 vig (need 52.4% to break even)
- **Calibration:** Are 150-pt projections actually averaging 150?

### Interpreting Backtest Results

A model with MAE around 7–9 points is normal for college basketball totals (it is an inherently noisy sport). Focus less on raw MAE and more on:

1. **Monotonic improvement by bucket:** 4–6 pt edges should outperform 0–2 pt edges
2. **Calibration shape:** Projections should neither systematically over- nor under-project across the total range
3. **Sample size:** Need 200+ games per bucket before trusting results

---

## Improving the Model Over Time

### Add More Historical Data

```bash
# Collect 3 full seasons
python main.py collect --start 2022-11-01 --end 2025-03-31
python main.py train
```

### Feature Ideas to Add

- **Player-level data:** Starter injury status, top-scorer availability
- **Travel distance:** Cross-country road games vs regional matchups
- **Altitude:** Teams playing in Denver/Colorado Springs
- **Referee tendencies:** Some refs call more fouls → more FTs → higher totals
- **TV game effects:** nationally-televised games tend to attract public money
- **Weather:** For tournament games at outdoor/partially outdoor venues

### Model Tuning Tips

After collecting 1,000+ games, tune hyperparameters with Optuna:

```python
import optuna
# Optimize XGBoost n_estimators, max_depth, learning_rate
# Optimize ensemble weights (baseline_weight, ml_weight)
# Optimize confidence factor weights
```

### When to Retrain

- At the start of each new season (November)
- After significant rule changes
- When MAE on the last 50 games is > 2 points higher than historical MAE
- When model calibration chart shows systematic bias

---

## Swapping Data Sources

All data sources implement abstract base classes in `src/data/base_adapter.py`. Adding a new source requires implementing three methods.

### Adding a New Schedule Source

```python
from src.data.base_adapter import BaseScheduleAdapter

class MyScheduleAdapter(BaseScheduleAdapter):
    def get_todays_schedule(self) -> list[dict]:
        # Fetch from your source
        # Must return dicts with: game_id, date, home_team, home_team_id,
        #   away_team, away_team_id, neutral_site, status, home_score, away_score
        ...

    def get_schedule_by_date(self, date: str) -> list[dict]:
        ...
```

Then register it in `DataAdapterFactory.get_schedule_adapter()` in `base_adapter.py`, and set `data_sources.schedule.primary: my_source` in `config.yaml`.

### Adding a New Odds Source

Implement `BaseOddsAdapter.get_current_odds()` and `get_odds_by_game()`. The system needs dicts with: `game_id, home_team, away_team, sportsbook, total, over_price, under_price`.

### Adapter Interface Requirements

All adapters must:
1. Extend the relevant abstract base class
2. Return dicts with the documented key names (for DB compatibility)
3. Handle errors gracefully (log and return empty list, never raise to pipeline)
4. Respect rate limits (use `tenacity` for retries)

---

## Extending to Other Sports

### NBA Adaptation

NBA totals have much smaller MAE potential (~4–5 pts) due to more consistent play. Key differences:

- `league_avg_possessions` ≈ 100 (vs 68 for NCAA)
- `league_avg_points_per_100` ≈ 115 (vs 105)
- `home_court_advantage` ≈ 2.5 pts (weaker than college)
- Add fatigue / back-to-back features (NBA plays every 2 days)
- ESPN API: change sport slug to `basketball/nba`
- The Odds API: change sport to `basketball_nba`

### NHL Adaptation

NHL totals are low-scoring (typical line: 5.5–6.5). Key differences:

- Target variable changes to goals (not points)
- Add goalie stats as key features (SV%, GAA)
- Pace becomes shots-on-goal rather than possessions
- The Odds API: sport = `icehockey_nhl`

### Player Props Adaptation

For player scoring props (points over/under):

- Replace game-level features with player-level usage, role, matchup
- Add opponent's defensive rating against position
- Feature: player's last-10 scoring vs team's avg scoring
- Target: individual player points
- Much higher noise than game totals; larger edge thresholds needed

---

## Troubleshooting

### "No games found" for today

1. Check the ESPN scoreboard URL manually in a browser
2. May be an off day (early November, late April)
3. ESPN API occasionally goes down — wait 15 minutes and retry
4. Verify `data_sources.schedule.primary: espn` in config.yaml

### "THE_ODDS_API_KEY is not set"

1. Copy `.env.example` to `.env`
2. Add your key: `THE_ODDS_API_KEY=your_actual_key`
3. The model will still run in baseline-only mode without odds (just no differential)

### ML models not loading

1. Run `python main.py collect` to gather data first
2. Run `python main.py train` — requires ≥ 50 completed games with features
3. Check `models/cbb_totals_models.joblib` exists

### Very low data completeness scores

This usually means stats are not in the database yet. Run:

```bash
python main.py collect --start 2024-11-01 --end 2025-03-01
```

Then re-run the pipeline. Stats are fetched from ESPN on first run and cached.

### SQLite "database is locked" error

Multiple pipeline instances running simultaneously. Kill other processes or switch to PostgreSQL:

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost/cbb_totals
```

### Streamlit dashboard shows empty data

The dashboard reads from the database. Run the pipeline at least once:

```bash
python main.py run
```

Then refresh the dashboard (click "Refresh Data" in the sidebar).

### Torvik data not loading

Barttorvik.com occasionally changes its CSV format. Check:

1. Visit `https://barttorvik.com/trank.php?year=2025&csv=1` in your browser
2. Verify column names match what `TorVikAdapter._TORVIK_COLS` expects
3. Update `_TORVIK_COLS` in `src/data/torvik_adapter.py` if needed

---

## License

MIT — use freely, no warranty. Past betting performance does not guarantee future results. Always gamble responsibly.
