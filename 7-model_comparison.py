"""
model_comparison.py — head-to-head comparison of model families.
Each family: 5 random seeds, mean ± std of val macro F1.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier,
                              ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier

from prepare import load_data, evaluate

# Optional — comment out either line if you haven't installed them
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

X_train, y_train, X_val, y_val, _ = load_data()

def make_logistic(penalty, seed):
    """Logistic with optional L1/L2 regularization. Needs scaling."""
    kw = dict(class_weight="balanced", random_state=seed, max_iter=2000)
    if penalty == "l1":
        return Pipeline([("scaler", StandardScaler()),
                         ("model", LogisticRegression(penalty="l1", solver="liblinear",
                                                       C=0.1, **kw))])
    elif penalty == "l2":
        return Pipeline([("scaler", StandardScaler()),
                         ("model", LogisticRegression(penalty="l2", C=0.1, **kw))])
    else:
        return Pipeline([("scaler", StandardScaler()),
                         ("model", LogisticRegression(penalty=None, **kw))])

def make_knn(seed):
    """KNN needs scaling because it's distance-based."""
    return Pipeline([("scaler", StandardScaler()),
                     ("model", KNeighborsClassifier(n_neighbors=15, n_jobs=1))])

# (name, factory). Factory takes seed and returns a fitted-able sklearn estimator.
MODELS = [
    ("Null (predict majority)", lambda s: DummyClassifier(strategy="most_frequent")),
    ("Logistic, no penalty",    lambda s: make_logistic(None, s)),
    ("Logistic, L1 (lasso)",    lambda s: make_logistic("l1", s)),
    ("Logistic, L2 (ridge)",    lambda s: make_logistic("l2", s)),
    ("KNN k=15",                lambda s: make_knn(s)),
    ("RandomForest n=300 d=6",  lambda s: RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=s, n_jobs=1)),
    ("ExtraTrees n=300 d=6",    lambda s: ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=s, n_jobs=1)),
    ("HistGradBoost it=300 d=4 lr=0.05", lambda s: HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05, random_state=s)),
]
if HAS_LGBM:
    MODELS.append(("LightGBM n=300 d=6 lr=0.05", lambda s: LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=s, n_jobs=1, verbose=-1)))
if HAS_XGB:
    MODELS.append(("XGBoost n=300 d=6 lr=0.05", lambda s: XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=s, n_jobs=1, eval_metric="logloss",
        use_label_encoder=False)))

SEEDS = list(range(5))

print(f"{'Model':40s}  {'mean F1':>8s}  {'std':>6s}  {'min':>6s}  {'max':>6s}")
print("-" * 78)

results = []
for name, factory in MODELS:
    f1s = []
    for seed in SEEDS:
        model = factory(seed)
        model.fit(X_train, y_train)
        f1, _, _ = evaluate(model, X_val, y_val)
        f1s.append(f1)
    mean, std = np.mean(f1s), np.std(f1s)
    print(f"{name:40s}  {mean:8.4f}  {std:6.4f}  {min(f1s):6.4f}  {max(f1s):6.4f}")
    results.append({"model": name, "mean_f1": mean, "std_f1": std,
                    "min_f1": min(f1s), "max_f1": max(f1s)})

pd.DataFrame(results).to_csv("output/model_comparison.csv", index=False)
print("\nSaved output/model_comparison.csv")