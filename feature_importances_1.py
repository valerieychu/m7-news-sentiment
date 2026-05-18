"""Extract & plot feature importances from the current model.py."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prepare import load_data
from model import build_model

# 1. Same data as run.py — train + val, with feature names
X_train, y_train, X_val, y_val, feature_names = load_data()

# 2. Build + fit the pipeline
pipe = build_model()
pipe.fit(X_train, y_train)

# 3. Reach into the pipeline to get the underlying estimator
#    (your pipeline is [("model", ExtraTreesClassifier(...))])
forest = pipe.named_steps["model"]
importances = forest.feature_importances_

# 4. Pair with feature names so output is interpretable
fi = (pd.Series(importances, index=feature_names)
        .sort_values(ascending=False))
print(fi)
fi.to_csv("feature_importance_results_1.csv")

# 5. Plot
fi.plot(kind="barh", figsize=(8, 10))
plt.gca().invert_yaxis()
plt.title("ExtraTrees feature importances (week 8)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_bar_1.png", dpi=150)
print("Saved feature_importance_bar_1.png")