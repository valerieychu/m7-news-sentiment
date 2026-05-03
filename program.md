# AutoResearch Agent Instructions

## Objective

Maximize **validation macro F1** on the binary-classification task:
given 39 predictors derived from GDELT news sentiment and yfinance prices,
predict whether each Magnificent-7 stock's next-day return will be positive
(`direction_t1 = 1`) or not (`direction_t1 = 0`).

## Rules

1. You may **ONLY** modify `model.py`.
2. `prepare.py` and `run.py` are **FROZEN** — do not touch them.
3. `build_model()` must return an sklearn-compatible estimator (a `Pipeline` is strongly preferred so that preprocessing travels with the estimator).
4. Training + evaluation must complete in **under 60 seconds** on CPU.
5. No additional data sources, external downloads, or new files.
6. **Never call `prepare.load_test()`** during the search phase. The test set is locked and should only be unsealed at the final evaluation, after the search is complete.

## Workflow

```
1. Read current model.py
2. Propose one modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check val_f1_macro in output (higher = better)
6. If improved:  git add model.py && git commit -m "feat: <description>"
7. If worse:     git checkout model.py   (revert)
8. Repeat from step 1
```

## Ideas to explore

- **Regularization**: `LogisticRegression(penalty='l2', C=...)`, `penalty='l1'` for feature selection, `penalty='elasticnet'` with `l1_ratio`.
- **Class balance**: add `class_weight='balanced'` — train is slightly up-biased while val may not be, so re-weighting can help macro F1 even when accuracy flat-lines.
- **Tree ensembles**: `RandomForestClassifier`, `GradientBoostingClassifier`, `HistGradientBoostingClassifier` (fastest of the three). These don't need scaling.
- **Feature engineering**: `PolynomialFeatures(degree=2, interaction_only=True)` on the sentiment columns, or targeted interactions like `tone_weighted * realized_vol5`.
- **Preprocessing swaps**: `RobustScaler` (tone has outliers around crash days), `QuantileTransformer(output_distribution='normal')`.
- **Feature subsetting**: drop price features (`daily_return`, `realized_vol5`, `earnings_week`) to isolate the pure sentiment signal; or drop sentiment to show price features alone are not enough.
- **Hyperparameter tuning** on the best model so far — grid over `C`, `max_depth`, `learning_rate`, `min_samples_leaf`, etc.

## What NOT to do

- Do not modify `prepare.py` — the split, metric, and plotting are frozen.
- Do not call `load_test()` — that's reserved for the final evaluation.
- Do not add new files or new dependencies beyond `sklearn`, `pandas`, `numpy`, `matplotlib`.
- Do not hard-code validation-set features or labels into the model.
- Do not change the function signature of `build_model()`.
- Do not random-shuffle the data or call `train_test_split`; the split is strictly time-based in `prepare.py`.

## Validation-set noise floor

The validation set is ~210 rows (1.5 months × 7 tickers). A single extra-good day on one ticker moves macro F1 by roughly 0.01–0.02. Treat sub-0.01 improvements as noise. Require at least a **0.02 macro-F1 gain** over the current best before marking a change as KEEP — otherwise you will accept noise and overfit to val.

## Final evaluation (end of project only)

After the search phase ends and `model.py` is pinned to the best-kept version:

1. Commit the final `model.py`.
2. Write a short `final_eval.py` that:
   - imports `build_model` from `model.py`,
   - calls `load_data()` to get train + val, concatenates them,
   - fits `build_model()` on the combined data,
   - calls `load_test()` (which emits a `RuntimeWarning` — that is expected; it is the only legitimate call to this function in the whole project),
   - reports `evaluate(model, X_test, y_test)` → macro F1, accuracy, recall.
3. Record the test-set numbers in your project report. They do **not** go into `results.tsv` — `results.tsv` is the search trajectory, and the test score is a single post-search number.
