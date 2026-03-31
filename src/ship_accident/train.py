from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from ship_accident.data import extract_target, prepare_raw_frame, read_table
from ship_accident.features_tabular import encode_tabular_train_test, tabular_to_float_dense
from ship_accident.features_text import build_text_blocks_train_test
from ship_accident.metrics import compute_metrics
from ship_accident.models import (
    get_model_grid,
    make_pipeline,
    make_pipeline_with_selection,
    make_selector,
    merge_param_grid_with_feature_selection,
)


def run_training(
    cfg: dict[str, Any],
    *,
    data_path: str | Path,
) -> dict[str, Any]:
    df = read_table(data_path)
    df = prepare_raw_frame(df, cfg)
    target_column = cfg["target_column"]
    X, y, target_mapping = extract_target(df, target_column)

    mode = cfg.get("mode", "tabular_only")
    text_columns = list(cfg.get("text_columns") or [])

    stratify = y if cfg.get("stratify", True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.get("test_size", 0.3),
        random_state=cfg.get("random_state", 42),
        stratify=stratify,
    )

    tab_drop = [c for c in text_columns if c in X_train.columns]
    X_train_tab = X_train.drop(columns=tab_drop, errors="ignore")
    X_test_tab = X_test.drop(columns=tab_drop, errors="ignore")

    X_train_e, X_test_e = encode_tabular_train_test(X_train_tab, X_test_tab)
    tab_tr = tabular_to_float_dense(X_train_e)
    tab_te = tabular_to_float_dense(X_test_e)

    text_tr, text_te = build_text_blocks_train_test(
        X_train,
        X_test,
        text_columns,
        mode,
        tfidf_max_features=int(cfg.get("tfidf_max_features", 1000)),
    )

    if text_tr is not None:
        Xtr = np.hstack([tab_tr, text_tr.astype(np.float64)])
        Xte = np.hstack([tab_te, text_te.astype(np.float64)])
    else:
        Xtr, Xte = tab_tr, tab_te

    labels_in_test = sorted(np.unique(np.concatenate([y_test, y_train])))
    target_names = [target_mapping[i] for i in labels_in_test]

    cv = int(cfg.get("grid_search_cv", 5))
    scoring = cfg.get("scoring", "accuracy")
    results: dict[str, Any] = {}
    fs_cfg = cfg.get("feature_selection") or {}
    fs_enabled = bool(fs_cfg.get("enabled"))
    rs = int(cfg.get("random_state", 42))

    for name in cfg.get("model_names", []):
        est, param_grid = get_model_grid(name, cfg)
        if fs_enabled:
            selector = make_selector(fs_cfg, random_state=rs)
            pipe = make_pipeline_with_selection(est, selector)
            param_grid = merge_param_grid_with_feature_selection(param_grid, fs_cfg)
        else:
            pipe = make_pipeline(est)
        n_jobs = int(cfg.get("n_jobs", 1))
        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        grid.fit(Xtr, y_train)
        y_pred = grid.predict(Xte)
        metrics = compute_metrics(
            y_test,
            y_pred,
            labels=labels_in_test,
            target_names=target_names,
        )
        n_sel: int | None = None
        est_fit = grid.best_estimator_
        if fs_enabled and hasattr(est_fit, "named_steps") and "select" in est_fit.named_steps:
            n_sel = int(est_fit.named_steps["select"].get_support().sum())
        row: dict[str, Any] = {
            "best_params": grid.best_params_,
            "best_cv_score": float(grid.best_score_),
            **metrics,
        }
        if n_sel is not None:
            row["n_features_after_selection"] = n_sel
        results[name] = row

    return {
        "target_mapping": {str(k): v for k, v in target_mapping.items()},
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "feature_selection_enabled": fs_enabled,
        "models": results,
    }
