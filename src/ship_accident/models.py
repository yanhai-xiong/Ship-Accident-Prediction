from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def make_selector(fs_cfg: dict[str, Any], *, random_state: int) -> SelectFromModel:
    """Pre-model feature selector (fit on each CV fold when inside Pipeline)."""
    est_name = fs_cfg.get("estimator", "random_forest")
    n_jobs = int(fs_cfg.get("n_jobs", -1))
    if est_name == "random_forest":
        base = RandomForestClassifier(
            n_estimators=int(fs_cfg.get("n_estimators", 200)),
            random_state=random_state,
            n_jobs=n_jobs,
        )
    elif est_name == "logistic_l1":
        base = LogisticRegression(
            penalty="l1",
            solver="saga",
            multi_class="multinomial",
            max_iter=int(fs_cfg.get("max_iter", 4000)),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown feature_selection.estimator: {est_name!r}")
    thr = fs_cfg.get("threshold", "median")
    return SelectFromModel(base, threshold=thr)


def merge_param_grid_with_feature_selection(
    clf_grid: dict[str, list[Any]],
    fs_cfg: dict[str, Any],
) -> dict[str, list[Any]]:
    """Copy clf grid (already ``clf__*`` keys); add ``select__threshold`` grid if provided."""
    out = dict(clf_grid)
    if fs_cfg.get("threshold_grid"):
        out["select__threshold"] = list(fs_cfg["threshold_grid"])
    return out


def get_model_grid(name: str, cfg: dict[str, Any] | None = None) -> tuple[Any, dict[str, list[Any]]]:
    cfg = cfg or {}
    rs = int(cfg.get("random_state", 42))
    cw = cfg.get("class_weight")

    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=rs), {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5],
        }
    if name == "hist_gradient_boosting":
        hgb = HistGradientBoostingClassifier(random_state=rs)
        if cw is not None:
            hgb.set_params(class_weight=cw)
        return hgb, {
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_iter": [200, 400],
            "clf__max_depth": [None, 12],
        }
    if name == "random_forest":
        rf = RandomForestClassifier(random_state=rs, n_jobs=-1)
        if cw is not None:
            rf.set_params(class_weight=cw)
        return rf, {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 15],
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


def make_pipeline_with_selection(estimator: Any, selector: SelectFromModel) -> Pipeline:
    return Pipeline([("select", selector), ("clf", estimator)])
