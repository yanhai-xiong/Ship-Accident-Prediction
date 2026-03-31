from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p, **read_kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(p, **read_kwargs)
    raise ValueError(f"Unsupported file type: {p} (use .csv or .xlsx)")


def prepare_raw_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    # Drop stray index column if present
    for c in ("Unnamed: 0",):
        if c in out.columns:
            out = out.drop(columns=[c])

    target = cfg["target_column"]
    if target not in out.columns:
        raise KeyError(f"Target column {target!r} not in data columns: {list(out.columns)}")

    drop_cols = list(cfg.get("drop_always", [])) + list(cfg.get("drop_for_target", []))
    drop_cols = [c for c in drop_cols if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")

    mode = cfg.get("mode", "tabular_only")
    text_cols = cfg.get("text_columns") or []
    if mode == "tabular_only":
        out = out.drop(columns=[c for c in text_cols if c in out.columns], errors="ignore")

    if cfg.get("drop_na", True):
        out = out.dropna(axis=0)
    return out


def extract_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, Any, dict[int, str]]:
    if target_column not in df.columns:
        raise KeyError(target_column)
    y = df[target_column].astype(str)
    y_encoded, y_labels = pd.factorize(y)
    target_mapping = {i: y_labels[i] for i in range(len(y_labels))}
    X = df.drop(columns=[target_column])
    return X, y_encoded, target_mapping
