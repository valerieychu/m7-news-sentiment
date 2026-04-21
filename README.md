# News Sentiment Predicting Tech Stock Direction

STAT 390 capstone AutoResearch project. Tests whether daily news-sentiment
signals about Magnificent-7 tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, META,
TSLA) can predict next-day price direction beyond chance.

---

## Problem

- **Task**: binary classification.
- **Target**: `direction_t1` — 1 if next-day close > today's close, 0 otherwise.
- **Metric**: validation **macro F1** (higher is better). Accuracy and
  positive-class recall are logged alongside for diagnostics.
- **Features (32)**: article counts, unique source counts, word-count-weighted
  tone scores (overall / positive / negative / polarity), per-day tone stats
  (mean / stddev / min / max), positive / negative / neutral article counts,
  activity-reference density, lag-1/2/3 tone / articles / polarity,
  5-day rolling tone mean, tone momentum, tone volatility, plus price-derived
  features (daily return, realized 5-day volatility) and an earnings-week
  calendar flag.
- **Span**: Jan 2025 through April 13 2026 (~2,233 labelled rows after
  dropping NaN lag-edges and last-day rows).
- **Baseline**: logistic regression with no penalty — establishes the
  first row in `results.tsv`.

## Split (LOCKED in `prepare.py`)

| Split | Date window                    | Approx. rows | Purpose |
|-------|--------------------------------|--------------|-----------------------------------|
| Train | 2025-01-01 → 2025-12-31        | ~1,750       | Fit models.                       |
| Val   | 2026-01-01 → 2026-02-14        | ~210         | Iterate / select during search.   |
| Test  | 2026-02-15 → 2026-04-13        | ~270         | LOCKED. Touched once at the end.  |

`load_data()` returns train + val only. `load_test()` is gated behind a
`RuntimeWarning` and should be called **exactly once**, at the final
evaluation after the search phase ends.

## Project Structure

```
m7-news-sentiment/
├── dataset.csv        # Frozen dataset (from GDELT + yfinance)
├── data_acquisition/  # FROZEN - Raw pull scripts + data acquisition (archive)
├── prepare.py         # FROZEN — data loading, evaluation, plotting
├── model.py           # EDITABLE — the only file the agent modifies
├── run.py             # FROZEN — runs one experiment, logs result
├── demo.py            # Standalone: runs 8 demo iterations end-to-end
├── program.md         # Agent instructions (rules + workflow + ideas)
├── README.md          # This file
├── results.tsv        # Experiment log (auto-generated, append-only)
└── performance.png    # Search trajectory plot (auto-generated)
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Setup

### 1. Install an AI coding agent (CLI)

You need a CLI coding agent that can read files, edit files, and run shell commands. 
Two recommended options — **neither requires an API key**.

#### Option A: Claude Code CLI (recommended)

```bash
# macOS / Linux / WSL — one-line install
curl -fsSL https://claude.ai/install.sh | bash

# macOS — or via Homebrew
brew install --cask claude-code

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

# Windows — or via WinGet
winget install Anthropic.ClaudeCode
```

Then launch:

```bash
cd m7-news-sentiment
claude
```

First launch opens a browser for login — **no API key needed**.
Works with any Claude subscription (Pro, Max, or Team).

Docs: https://code.claude.com/docs

#### Option B: OpenAI Codex CLI

```bash
npm install -g @openai/codex
cd m7-news-sentiment
codex
```

First launch opens a browser for ChatGPT login — **no API key needed**.
Works with ChatGPT Plus or higher.

Docs: https://github.com/openai/codex

### 2. Install Python environment

Requires **Python 3.10+** with `scikit-learn`, `pandas`, `numpy`, and
`matplotlib`.

```bash
# Check version
python3 --version                # expect 3.10 or higher

# Install Python if needed (choose one)
pip install scikit-learn pandas numpy matplotlib
# -or-
conda install scikit-learn pandas numpy matplotlib
```

| Package      | Version  | Purpose                          |
|--------------|----------|----------------------------------|
| scikit-learn | >= 1.3   | Models, pipelines, metrics       |
| pandas       | >= 2.0   | CSV loading, date filtering      |
| numpy        | >= 1.24  | Array operations                 |
| matplotlib   | >= 3.7   | Performance plotting             |

No GPU, no PyTorch, no heavy downloads — everything runs on CPU.

### 3. Verify setup

```bash
# Quick check: all imports work
python3 -c "import sklearn, pandas, numpy, matplotlib; print('All good')"

# Full check: run one experiment
python3 run.py "sanity run" --baseline
# Expected output:
#   Data: ~1750 train, ~210 val, 32 features
#   val_f1_macro: 0.4XXX
#   val_accuracy: 0.XXXX
#   val_recall:   0.XXXX
#   Result logged to results.tsv (status=baseline)

# Clean up the sanity result if you want a fresh log
rm -f results.tsv
```

---

## How to Run the Agent Loop

### Quick start (copy-paste this prompt into your agent)

```
Read program.md for your instructions, then read model.py.
Run `python run.py "baseline" --baseline` to establish the baseline macro F1.
Then enter the AutoResearch loop:

1. Propose one modification to model.py (e.g., different classifier,
   regularization, feature engineering, hyperparameter change).
2. Edit model.py with your change.
3. Run: python run.py "<short description of what you changed>"
4. Compare the new val_f1_macro to the current best (higher = better).
   - If improved by at least 0.02: KEEP the change, note the new best.
   - Otherwise:                    REVERT model.py to the previous version.
5. Repeat from step 1. Try at least 6 different ideas.

Do NOT call load_test() — the test set is reserved for final evaluation.

After all iterations, run `python prepare.py` to regenerate performance.png.
Print a summary table of all experiments and which were kept vs discarded.
```

### More specific prompt (if you want to direct the search)

```
You are an AutoResearch agent. Read program.md for rules.

Your job: maximize val_f1_macro on News-Sentiment Tech-Stock Direction
by modifying model.py only.

Constraints:
- model.py must define build_model() returning an sklearn estimator
- Do NOT modify prepare.py or run.py
- Do NOT call load_test()
- Each experiment must finish in < 60 seconds on CPU

Search strategy:
1. Start with baseline (LogisticRegression, no penalty, StandardScaler)
2. Try regularized linear models: L2, L1, ElasticNet, plus class_weight='balanced'
3. Try feature engineering (PolynomialFeatures interaction_only, RobustScaler)
4. Try tree ensembles (RandomForest, GradientBoosting, HistGradientBoosting)
5. Try hyperparameter tuning on the best model so far

For each experiment:
- Run: python run.py "<description>"
- If val_f1_macro improved ≥ 0.02 → keep
- If val_f1_macro did not improve ≥ 0.02 → revert model.py to previous version
- Log your reasoning for each decision

After finishing, run: python prepare.py
```

---

## Example Demo Run

A pre-scripted 8-iteration example is included as `demo.py`. It runs the full
search non-interactively so you can see the agent loop's output and trajectory
before handing control to a real agent:

```bash
rm -f results.tsv performance.png
python demo.py
```

It tries, in order:

| #  | Model                                     | Role                          |
|----|-------------------------------------------|-------------------------------|
| 1  | LogReg, no penalty                        | baseline                      |
| 2  | LogReg, L2 (C=1.0)                        | regularization sweep          |
| 3  | LogReg, L1 (C=0.5)                        | sparse feature selection      |
| 4  | PolyFeatures(2, interaction) + LogReg L2  | feature engineering           |
| 5  | PolyFeatures(3, full) + LogReg            | deliberate overshoot (discard)|
| 6  | RandomForest(n=200)                       | non-linear tree baseline      |
| 7  | GradientBoosting(n=200, depth=3)          | boosted trees                 |
| 8  | HistGBT(iter=300, tuned)                  | modern GBM                    |

After it finishes, open `performance.png` to see the trajectory.

---

## Plotting Results

Any time after runs have been logged:

```bash
python prepare.py
# Regenerates performance.png from results.tsv
```

The plot has two panels:

- **Top**: validation **macro F1** over experiments — the primary metric.
- **Bottom**: validation **accuracy** over experiments.

Points are color-coded by status: blue = baseline, green = keep, red = discard.
A green best-so-far envelope traces the running maximum. Dashed grey line at
0.5 marks approximate chance level for both metrics.

---

## Final Evaluation on the Locked Test Set

Once the search phase is complete and `model.py` is pinned to the best-kept
version, do exactly one pass on the test set. Create `final_eval.py`:

```python
import warnings
import numpy as np
from model import build_model
from prepare import load_data, load_test, evaluate

X_train, y_train, X_val, y_val, _ = load_data()

# After the search, selection is done — combine train + val for the final fit.
X_all = np.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])

model = build_model()
model.fit(X_all, y_all)

# Expect a RuntimeWarning from load_test — this is the only legitimate call.
X_test, y_test = load_test()
f1, acc, recall = evaluate(model, X_test, y_test)
print(f"FINAL TEST  f1_macro={f1:.4f}  accuracy={acc:.4f}  recall={recall:.4f}")
```

The test-set number goes in the project report. It is **not** appended to
`results.tsv` — `results.tsv` is the search trajectory, and the test score
is a single post-search evaluation. If you ever re-run `final_eval.py` after
seeing its result and then go back to tweak `model.py`, you have compromised
the holdout, and the result is no longer trustworthy.

---

## Adapting This Structure for Another Project

The template that produced this repo is project-agnostic. To port it:

1. **Replace `prepare.py`** with your own data loading, split, evaluation
   metric, and plotting. Keep it frozen — the agent must not touch it.
2. **Replace `model.py`** with whatever the agent should modify
   (model definition, hyperparameters, feature engineering, etc.).
3. **Update `program.md`** with your specific rules, constraints, and
   search ideas.
4. **Update `run.py`** if your training loop or logging signature differs.

The key principle: **separate what changes (`model.py`) from what measures
(`prepare.py`)**. The agent operates on the first; the evaluator lives
in the second.
