# 7_calibrated_extratrees.py
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from prepare import load_data, evaluate

X_train, y_train, X_val, y_val, _ = load_data()

print(f"{'Method':30s}  {'mean F1':>8s}  {'std':>6s}")
for method in ["isotonic", "sigmoid"]:
    f1s = []
    for seed in range(5):
        pipe = Pipeline([("model", CalibratedClassifierCV(
            ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=seed, n_jobs=1),
            method=method, cv=3))])
        pipe.fit(X_train, y_train)
        f1, _, _ = evaluate(pipe, X_val, y_val)
        f1s.append(f1)
    print(f"Calibrated ({method:9s}) ExtraTrees     {np.mean(f1s):8.4f}  {np.std(f1s):6.4f}")