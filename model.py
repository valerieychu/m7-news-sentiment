"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for California Housing regression.
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    return Pipeline([
        ("model", HistGradientBoostingRegressor(
            max_iter=300, max_depth=8, learning_rate=0.08,
            min_samples_leaf=20, random_state=42,
        )),
    ])
