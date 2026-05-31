# 7_stability_check.py
# on current model.py
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from prepare import load_data, evaluate

X_train, y_train, X_val, y_val, feats = load_data()
f1s = []
for seed in range(10):
    pipe = Pipeline([("model", ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=-1))])
    pipe.fit(X_train, y_train)
    f1, _, _ = evaluate(pipe, X_val, y_val)
    f1s.append(f1)
    print(f"seed={seed}: f1={f1:.4f}")
print(f"\nmean={np.mean(f1s):.4f}, std={np.std(f1s):.4f}")
print(f"min={np.min(f1s):.4f}, max={np.max(f1s):.4f}")

# Since the std-dev is 0.0064 (which is ≤ 0.01), the result is stable and the mean F1-score of 0.5591. 
# The validation set is large enough that a single-seed F1 is a reliable estimate. 

# Output: 
# seed=0: f1=0.5626
# seed=1: f1=0.5612
# seed=2: f1=0.5626
# seed=3: f1=0.5667
# seed=4: f1=0.5599
# seed=5: f1=0.5585
# seed=6: f1=0.5641
# seed=7: f1=0.5459
# seed=8: f1=0.5485
# seed=9: f1=0.5612

# mean=0.5591, std=0.0064
# min=0.5459, max=0.5667