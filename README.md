# Staley NFL Prediction Model

NFL game prediction using XGBoost regression.

## Setup

```bash
make initialize
```

This installs R, nflfastR, creates a Python venv, and installs pip dependencies.

To set up manually:

```bash
sudo apt-get install -y r-base
sudo Rscript -e 'install.packages("nflfastR", repos="https://cloud.r-project.org")'
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python src/model.py train --n-estimators 500 --max-depth 8 --learning-rate 0.05
```

### Arguments

| Argument             | Default | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `--n-estimators`     | `100`   | Number of boosting rounds            |
| `--max-depth`        | `6`     | Maximum tree depth                   |
| `--learning-rate`    | `0.1`   | Boosting learning rate               |
| `--subsample`        | `0.8`   | Fraction of training data per tree   |
| `--colsample-bytree` | `0.8`   | Fraction of features per tree        |
| `--reg-alpha`        | `0.0`   | L1 regularization                    |
| `--reg-lambda`       | `1.0`   | L2 regularization                    |

### Training Output

Each training run creates a timestamped folder in `models/` (e.g., `20260130T200000_a1b2c3d4e5`) containing:

- `{hash}-training.log` -- training params, duration, and metrics (RMSE, MAE, R², feature importance)

### Tracked Metrics

- **RMSE** (train and validation) -- root mean squared error per boosting round
- **MAE** (train and validation) -- mean absolute error per boosting round
- **R²** -- final coefficient of determination on the validation set
- **Feature importance** -- per-feature importance scores from the trained model

## Project Structure

```
data/
  teams_logos_colors.csv  # Team metadata
src/
  model.py                # XGBoost model, training CLI, and metrics tracking
models/                   # Saved training runs (auto-created per run)
info/
  ensemble-analysis.md    # Ensemble approach analysis
  model-analysis.md       # Model performance analysis
```
