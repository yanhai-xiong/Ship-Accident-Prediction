from __future__ import annotations

import numpy as np
import pandas as pd


def encode_tabular_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame, *, drop_first: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_e = pd.get_dummies(X_train, drop_first=drop_first)
    X_test_e = pd.get_dummies(X_test, drop_first=drop_first)
    X_test_e = X_test_e.reindex(columns=X_train_e.columns, fill_value=0)
    return X_train_e, X_test_e


def tabular_to_float_dense(X: pd.DataFrame) -> np.ndarray:
    return X.astype(np.float64).values
