from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def get_model_grid(name: str) -> tuple[Any, dict[str, list[Any]]]:
    if name == "gradient_boosting":
        return GradientBoostingClassifier(), {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5],
        }
    if name == "random_forest":
        return RandomForestClassifier(), {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10],
        }
    if name == "svm":
        return SVC(), {
            "clf__C": [1, 10],
            "clf__kernel": ["linear", "rbf"],
        }
    if name == "knn":
        return KNeighborsClassifier(), {
            "clf__n_neighbors": [3, 5, 7],
        }
    raise ValueError(f"Unknown model name: {name!r}")


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("clf", estimator)])
