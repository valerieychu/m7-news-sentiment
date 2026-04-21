"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for the News-Sentiment Stock-Direction task.
build_model() must return an sklearn-compatible classifier with .fit() and
.predict() methods. A Pipeline is strongly preferred so that scaling /
feature engineering lives alongside the estimator and is applied identically
to train, val, and test.

Starter model: logistic regression with no penalty (mirrors the hand-coded
Week 2 baseline) wrapped in a StandardScaler. The agent should replace this
with increasingly capable models as the search progresses.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty=None, max_iter=1000, random_state=42)),
    ])
