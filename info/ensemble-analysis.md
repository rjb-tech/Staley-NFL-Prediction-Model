# Ensemble Analysis: 5 Averaged Models vs Single Model

## Verdict

The 5-model averaging approach is almost certainly not adding meaningful value in the current setup. A single model would work just as well.

## Why the Current Ensemble Doesn't Help

### XGBoost is already an ensemble internally

Each of the 5 models uses `num_parallel_tree=987`, which means each individual model is already building 987 decision trees in a random-forest-like structure within the gradient boosting framework. That's effectively 5 x 987 = 4,935 trees. Variance reduction from averaging follows a 1/sqrt(N) curve — going from 987 to 4,935 trees buys almost nothing on that curve.

### The 5 models lack diversity

Ensemble averaging works when individual models make *different* errors that cancel out. The classic ways to get diversity are:

- Different training data subsets (bagging)
- Different feature subsets
- Different algorithms entirely (e.g., XGBoost + logistic regression + random forest)
- Different hyperparameters

All 5 models use the same data, same features, and same hyperparameters. The only variation comes from XGBoost's internal randomization (random seeds). That's the weakest form of diversity — it produces near-identical models that make near-identical errors, so averaging them barely moves the needle.

### The cost is real

- 5x model storage (`.staley` files)
- 5x inference time
- More complex code for loading/averaging

That complexity has a maintenance cost with no proportional accuracy payoff.

## What Would Actually Help

If keeping an ensemble concept, make it *useful*:

### Different algorithms
Average an XGBoost model with a LightGBM model and a logistic regression. Different inductive biases produce genuinely different error patterns.

### Different feature windows
One model trained on season-long stats, another on last-5-games rolling stats. They'd capture different signals.

### Stacking
Train a meta-model on the outputs of diverse base models, rather than naive averaging.

## Recommendation

Drop to a single XGBoost model and invest the freed-up complexity budget into:

1. **Better hyperparameter tuning** — Optuna/Bayesian search over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
2. **Feature additions** — Third-down rate, red zone EPA, rest days, rolling stats, home/away splits (see `model-analysis.md`)

These will move accuracy more than any ensemble scheme on identical models.
