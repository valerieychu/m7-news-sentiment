# 7_top_models.py
# tune top models to fact check autoresearch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from prepare import load_data, evaluate
import pandas as pd

X_train, y_train, X_val, y_val, _ = load_data()
tscv = TimeSeriesSplit(n_splits=3)

CONFIGS = [
    ("ExtraTrees",
     ExtraTreesClassifier(class_weight="balanced", random_state=42, n_jobs=1),
     {"n_estimators": [200, 300, 500],
      "max_depth": [4, 6, 8, None],
      "min_samples_leaf": [3, 5, 10],
      "max_features": ["sqrt", 0.5]}),
    ("KNN",
     Pipeline([("s", StandardScaler()), ("model", KNeighborsClassifier(n_jobs=1))]),
     {"model__n_neighbors": [5, 15, 30, 50, 75, 100],
      "model__weights": ["uniform", "distance"],
      "model__metric": ["euclidean", "manhattan"]}),
    ("RandomForest",
     RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
     {"n_estimators": [200, 300, 500],
      "max_depth": [4, 6, 8, None],
      "min_samples_leaf": [3, 5, 10],
      "max_features": ["sqrt", 0.5]}),
]

results = []
for name, est, grid in CONFIGS:
    print(f"\n=== Tuning {name} ===")
    gs = GridSearchCV(est, grid, cv=tscv, scoring="f1_macro", n_jobs=1, verbose=1)
    gs.fit(X_train, y_train)
    val_f1, val_acc, val_rec = evaluate(gs.best_estimator_, X_val, y_val)
    print(f"  best params:   {gs.best_params_}")
    print(f"  best CV F1:    {gs.best_score_:.4f}")
    print(f"  actual val F1: {val_f1:.4f}")
    print(f"  CV–val gap:    {gs.best_score_ - val_f1:+.4f}")
    results.append({
        "model": name,
        "best_params": str(gs.best_params_),
        "cv_f1": round(gs.best_score_, 4),
        "val_f1": round(val_f1, 4),
        "val_acc": round(val_acc, 4),
        "val_recall": round(val_rec, 4),
    })

pd.DataFrame(results).to_csv("output/tuning_results.csv", index=False)
print("\nSaved output/tuning_results.csv")