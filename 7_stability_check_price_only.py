# 7_stability_check_priceonly.py
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from prepare import load_data, evaluate

X_train, y_train, X_val, y_val, feature_names = load_data()

SENTIMENT = [n for n in feature_names if any(k in n for k in
    ["tone", "polarity", "article", "source", "activity"])]
drop_idx = [feature_names.index(c) for c in SENTIMENT]
keep_idx = [i for i in range(X_train.shape[1]) if i not in drop_idx]

X_tr = X_train[:, keep_idx]
X_vl = X_val[:, keep_idx]
print(f"PRICE-only: kept {len(keep_idx)} of {len(feature_names)} features")

f1s = []
for seed in range(10):
    model = ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=seed, n_jobs=1)
    model.fit(X_tr, y_train)
    f1, _, _ = evaluate(model, X_vl, y_val)
    f1s.append(f1)
    print(f"seed={seed}: f1={f1:.4f}")

print(f"\nPRICE-only: mean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}")
print(f"ALL-features (for comparison): mean=0.5591, std=0.0064")