# Staley NFL Prediction Model — Analysis & Improvement Plan

## Project Synopsis

**Staley** is an NFL game prediction system with an R data pipeline feeding Python ML models.

### Architecture

1. **Data Pipeline** (`staley_driver.R`): Fetches play-by-play data via `nflfastR`/`nflreadr` and engineers **15 features per team** (30 total + 1 divisional flag):
   - EPA metrics (rush/pass, offensive/defensive)
   - First down rates (offense/defense)
   - Turnover differential
   - Explosive play rates (15+ yard plays)
   - O-line / D-line metrics (QB hits + sacks)
   - Penalty rates
   - Divisional game indicator

2. **Model** (`train_staley_v3.py`): A **5-member XGBoost ensemble** wrapped in `MultiOutputRegressor`. Each model uses 987 parallel trees, a learning rate of 0.0095, and gamma of 0.1. Targets are one-hot encoded point scores. The five models' outputs are averaged, and `argmax` comparison determines the winner. Tiebreaker uses aggregate EPA.

3. **Prediction** (`staley_says_v3.py`): Loads the 5 serialized `.staley` models, scales incoming week data with `StandardScaler`, runs ensemble inference, and pushes results to Google Sheets via `gspread`.

4. **Validation** (`test_staley.py`): Replays predictions against actual outcomes, tracking correct/incorrect/tie results. Reported accuracy: **62.1%** on the 2020-2021 season.

5. **Infrastructure**: Docker Compose with PostgreSQL 14 + Python 3.10. An API layer appears partially scaffolded but incomplete.

---

## XGBoost vs Neural Network: Pros and Cons

The current V3 uses XGBoost (despite the README mentioning "neural network" from V1).

### XGBoost (Current Approach)

**Pros:**
- **Strong baseline for tabular data.** Gradient-boosted trees consistently win tabular benchmarks (Kaggle, academic studies). With only 31 input features, this is firmly in XGBoost's wheelhouse.
- **Less data hungry.** The training set is ~1,800 games across 21 seasons. That's modest. XGBoost handles small-to-medium datasets well without overfitting as aggressively as neural nets.
- **Interpretability.** Feature importance scores are built-in. You can see which stats (e.g., pass EPA, turnover differential) drive predictions most, which is valuable for debugging and trust.
- **Fast training and iteration.** Training takes seconds/minutes, not hours. This lets you experiment with features, hyperparameters, and data windows quickly.
- **Robust to feature scale.** Trees don't technically need scaling (though the model currently scales with `StandardScaler`). Less preprocessing sensitivity.
- **Handles missing data natively.** XGBoost can route missing values in splits without imputation.

**Cons:**
- **No temporal/sequential learning.** XGBoost treats each game as i.i.d. It can't natively learn that a team is on a hot streak, or that momentum/injury trajectories matter across weeks.
- **Limited interaction learning.** While trees capture some feature interactions, they don't learn complex nonlinear feature combinations as flexibly as deep networks.
- **Ensembling is manual.** The model runs 5 independent models and averages. A neural net can learn ensemble-like behavior internally.
- **Plateaus with more data.** If you significantly expand features (e.g., player-level data, play-by-play sequences), XGBoost won't scale as well as deep learning.

### Neural Network (V1 Approach / Potential V4)

**Pros:**
- **Flexible architecture.** Recurrent layers (LSTM/GRU) could model team performance over time, or attention mechanisms could weigh recent weeks more heavily — learning the optimal window rather than hardcoding the 18-week rule.
- **End-to-end feature learning.** A neural net can discover feature interactions not explicitly engineered. With enough data, it could learn that "high pass EPA + low explosive play rate" means something different than either feature alone.
- **Embedding layers for categorical data.** Teams, divisions, and matchup types can be embedded into learned vector spaces, capturing relationships (e.g., NFC West teams play differently against each other).
- **Scales with richer inputs.** If incorporating player-level stats, betting lines, weather, play-by-play sequences, or text data (injury reports), neural nets handle heterogeneous inputs naturally.
- **Multi-task learning.** A single network can jointly predict winner, point spread, and over/under, sharing learned representations across tasks.

**Cons:**
- **Data scarcity is the killer.** ~1,800 games is very small for a neural net. NFL seasons produce only ~270 regular season games/year. Overfitting is likely without aggressive regularization, and even then may not outperform XGBoost.
- **Harder to tune.** Architecture choices (layers, units, dropout, learning rate schedules, batch size) create a large hyperparameter space. XGBoost has fewer knobs and is more forgiving.
- **Less interpretable.** Understanding *why* a neural net picks one team is harder. SHAP values exist but are more expensive to compute for deep models.
- **Slower iteration cycle.** Training takes longer, debugging is harder, and reproducibility requires more care (random seeds, GPU variance).
- **Diminishing returns on tabular data.** Recent research (Grinsztajn et al., 2022) shows tree-based methods still outperform deep learning on medium-sized tabular datasets. The 31-feature setup is exactly this regime.

### Recommendation

**Stick with XGBoost as the primary model**, but consider targeted improvements:

1. **Feature engineering over architecture changes.** The biggest accuracy gains will likely come from better features, not a different model. Consider adding: Elo ratings, rest days, home/away splits, quarterback-specific EPA, red zone efficiency, or rolling-window features (last 3-5 games vs season-long).
2. **Hyperparameter tuning.** 987 parallel trees with a 0.0095 learning rate is unusual. A proper Bayesian hyperparameter search (Optuna) could find a better configuration.
3. **Hybrid approach if scaling up.** If adding significantly richer data (player-level, sequential game data), a **TabNet** or **FT-Transformer** model bridges the gap — they're neural architectures purpose-built for tabular data that outperform vanilla MLPs while being more data-efficient.
4. **Neural nets make sense only if** moving to sequence modeling (predicting based on the trajectory of a season) or incorporating unstructured data (injury reports, news). For the current 31-feature, ~1,800-sample tabular setup, XGBoost is the right call.

The 62.1% accuracy is reasonable for NFL prediction (Vegas implied accuracy is roughly 66-68%), and the gap is more likely closed by better data than a different model architecture.

---

## Current CSV Data

All five data files share the same 15-feature-per-team structure:

| File | Rows | Notes |
|---|---|---|
| `training_data_with_points.csv` | 5,608 | Main training set (~21 seasons), has `AWAY_PTS` / `HOME_PTS` |
| `raw_data.csv` | 5,608 | Same data but uses `PTS_DIFF` instead of separate point columns |
| `training_data_2021_with_points.csv` | 272 | 2021 subset for validation |
| `2021_season_games.csv` | 256 | 2021 games with `PTS_DIFF` format |
| `teams_logos_colors.csv` | 36 | Reference data (logos, colors, abbreviations) |

### Current Features (15 per team)

| Category | Features |
|---|---|
| EPA | Rush EPA, Pass EPA (offense & defense) |
| Efficiency | First Down Rate (offense & defense) |
| Turnovers | Offensive TO, Defensive TO |
| Explosiveness | Offensive Explosive Rate, Defensive Explosive Rate |
| Penalties | Count, Yards |
| Line Play | OL Metric, DL Metric (sacks + QB hits) |
| Context | Divisional game flag |

**Data quality issue:** Row 12 in both training files contains all `NA` values.

---

## What's Available but Not Being Used

The `nflfastR` play-by-play dataset has **300+ columns**. The R driver (`staley_driver.R`) only touches about 15 of them. Below is what's being left on the table, grouped by likely impact.

### Tier 1 — High Impact, Easy to Add

These use fields already in `nflfastR` and require minimal new engineering:

- **Third-down conversion rate.** First downs are tracked but not situational efficiency. Third-down offense/defense is one of the strongest predictors of game outcome. Fields: `down`, `first_down`.

- **Red zone efficiency.** EPA inside the 20 is very different from midfield EPA. A team that moves the ball but can't score is overvalued by raw EPA. Fields: `yardline_100`, `epa`, `touchdown`.

- **Win Probability Added (WPA).** `wpa` is a per-play metric that weights by game leverage. It captures clutch performance that flat EPA misses.

- **Completion percentage / air yards.** `air_yards`, `yards_after_catch`, `complete_pass` — separating YAC from air yards tells you whether a passing game is schemed (YAC-heavy) or talent-driven (deep passing). These break differently under pressure.

- **Pressure rate (not just sacks).** The model uses `sack` + `qb_hit` for the line metric, but `nflfastR` also has hurries and knockdowns. A team that pressures on 35% of dropbacks but only sacks 5% is very different from one that gets sacks without consistent pressure.

### Tier 2 — High Impact, Moderate Effort

These require additional data joins or external sources but are well worth it:

- **Rest days / short weeks.** Thursday games, Monday-to-Sunday turnarounds, and bye weeks significantly affect performance. The schedule data from `nflreadr` already has game dates — computing days between games is straightforward.

- **Home/away performance splits.** The model currently uses a single set of season stats per team. Computing separate home and away EPA/efficiency metrics would capture the real home-field advantage (which varies dramatically by team and stadium).

- **Rolling/recency-weighted stats.** The driver uses a flat average of all games through the current week (or an 18-week window for week 20+). A recency-weighted approach (e.g., exponential decay or last-5-games rolling average) would better capture team trajectory and mid-season changes (trades, injuries, scheme adjustments).

- **Scoring rate by drive.** `nflfastR` has `fixed_drive` and `fixed_drive_result` columns. TD rate per drive, FG rate per drive, and turnover-on-downs rate per drive are strong efficiency signals distinct from per-play EPA.

### Tier 3 — Potentially High Impact, More Work Required

These require external data sources or significant engineering:

- **Quarterback-specific metrics.** QB changes mid-season are common and massively affect team output. Tracking passer EPA by `passer_player_id` and flagging QB changes would capture this. `nflfastR` has `passer_player_name` and `passer_player_id`.

- **Injury/roster availability.** No play-by-play field for this, but `nflreadr` has injury report data. Even a simple "number of starters on injury report" feature would help.

- **Weather.** `nflfastR` includes `weather`, `wind`, `temp` in some seasons. Cold, wind, and rain suppress passing EPA and affect game totals.

- **Betting lines (as a feature, not a target).** `nflfastR` includes `spread_line` and `total_line`. Vegas lines encode enormous information (injuries, matchups, public sentiment). Using the opening line as a feature (not the closing line) is a common and powerful approach in sports modeling.

- **Defensive coverage metrics.** `defenders_in_box`, `number_of_pass_rushers` — these tell you about defensive scheme tendencies (blitz-heavy vs coverage-heavy) which affect how offenses perform against them.

### Tier 4 — Lower Priority

- **Penalty type breakdowns** (offensive holding vs defensive PI — very different signals)
- **Time of possession** (derivable from play counts per drive)
- **Special teams metrics** (punt/kick return EPA, field goal percentage)
- **Coaching data** (head coach win rate, offensive/defensive coordinator tendencies — requires external source)

---

## Where to Focus

The biggest bang-for-buck improvements, roughly ordered:

1. **Third-down conversion rate** — trivial to add, strong signal
2. **Red zone EPA** — trivial to add, directly predicts scoring
3. **Rest days** — easy from schedule data, known predictive factor
4. **Rolling/recency-weighted stats** — replaces flat averages, captures momentum
5. **Home/away splits** — separates a team's road and home performance
6. **Spread line as feature** — if comfortable using Vegas data, this alone could add several percentage points
7. **QB-specific EPA** — captures the single most impactful roster variable

Adding features 1-5 requires only changes to `staley_driver.R` and retraining. No new data sources needed — it's all already in `nflfastR`.
