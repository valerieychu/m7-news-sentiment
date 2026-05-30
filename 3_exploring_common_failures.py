#### Exploring Common Failure Modes ####


#### Looking At Misclassified Examples ####
import numpy as np
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — saves PNGs instead of showing windows
import matplotlib.pyplot as plt

# ── make sure imports resolve whether script is run from project root or sub-dir ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare import load_data
from model import build_model

X_train, y_train, X_val, y_val, feature_names = load_data()
model = build_model()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# Find mistakes
errors = np.where(y_pred != y_val)[0]

print(f"Total errors: {len(errors)}")

# Show a few examples
for i in errors[:10]:
    print("----")
    print(f"True: {y_val[i]}")
    print(f"Pred: {y_pred[i]}")
    print(f"Text: {X_val[i]}")


#### Check Confusion Matrix ####

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

#### Inspect “High Confidence But Wrong” ####
probs = model.predict_proba(X_val)
y_pred = probs.argmax(axis=1)
confidence = probs.max(axis=1)

errors = np.where(y_pred != y_val)[0]

# Sort errors by confidence (descending)
high_conf_errors = sorted(errors, key=lambda i: -confidence[i])

for i in high_conf_errors[:10]:
    print("----")
    print(f"True: {y_val[i]}")
    print(f"Pred: {y_pred[i]}")
    print(f"Confidence: {confidence[i]:.3f}")
    print(f"Text: {X_val[i]}")


#### Find Underperforming Classes ####
from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred))
