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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline


def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    return Pipeline([
        ("model", ExtraTreesClassifier(n_estimators=300, max_depth=6,
                                        min_samples_leaf=5,
                                        class_weight="balanced",
                                        random_state=42, n_jobs=-1)),
    ])
