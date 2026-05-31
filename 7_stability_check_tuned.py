# 7_stability_check_tuned.py — does the tuned config survive across seeds?
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from prepare import load_data, evaluate

X_train, y_train, X_val, y_val, _ = load_data()

f1s_current, f1s_tuned = [], []
for seed in range(10):
    # Current model.py config
    cur = ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                min_samples_leaf=5, class_weight="balanced",
                                random_state=seed, n_jobs=1)
    cur.fit(X_train, y_train)
    f1, _, _ = evaluate(cur, X_val, y_val)
    f1s_current.append(f1)

    # Newly tuned config
    new = ExtraTreesClassifier(n_estimators=200, max_depth=8,
                                min_samples_leaf=5, max_features="sqrt",
                                class_weight="balanced",
                                random_state=seed, n_jobs=1)
    new.fit(X_train, y_train)
    f1, _, _ = evaluate(new, X_val, y_val)
    f1s_tuned.append(f1)

print(f"Current model.py   mean={np.mean(f1s_current):.4f}  std={np.std(f1s_current):.4f}")
print(f"Tuned (depth=8)    mean={np.mean(f1s_tuned):.4f}  std={np.std(f1s_tuned):.4f}")
print(f"Δ (tuned − current): {np.mean(f1s_tuned) - np.mean(f1s_current):+.4f}")
print(f"\nKEEP if Δ ≥ +0.02; otherwise DISCARD and stick with current.")