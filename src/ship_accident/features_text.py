from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer


def fit_transform_tfidf_train_test(
    train_texts: pd.DataFrame,
    test_texts: pd.DataFrame,
    *,
    max_features: int = 1000,
) -> tuple[csr_matrix, csr_matrix, list[TfidfVectorizer]]:
    blocks_train: list[csr_matrix] = []
    blocks_test: list[csr_matrix] = []
    vectorizers: list[TfidfVectorizer] = []
    for col in train_texts.columns:
        vec = TfidfVectorizer(max_features=max_features)
        tr = vec.fit_transform(train_texts[col].fillna("").astype(str))
        te = vec.transform(test_texts[col].fillna("").astype(str))
        blocks_train.append(tr)
        blocks_test.append(te)
        vectorizers.append(vec)
    if not blocks_train:
        empty = csr_matrix((train_texts.shape[0], 0))
        return empty, empty, []
    return hstack(blocks_train, format="csr"), hstack(blocks_test, format="csr"), vectorizers


def bert_embed_series(texts: pd.Series, *, batch_size: int = 16) -> np.ndarray:
    try:
        import torch
        from transformers import BertModel, BertTokenizer
    except ImportError as e:
        raise ImportError(
            "tabular_bert requires optional deps: pip install 'ship-accident[bert]'"
        ) from e

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    arr_list: list[np.ndarray] = []
    values = texts.fillna("").astype(str).tolist()
    for i in range(0, len(values), batch_size):
        batch = values[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        arr_list.append(emb)
    return np.vstack(arr_list)


def build_text_blocks_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    text_columns: list[str],
    mode: str,
    *,
    tfidf_max_features: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode == "tabular_only" or not text_columns:
        return None, None
    cols = [c for c in text_columns if c in df_train.columns and c in df_test.columns]
    if not cols:
        return None, None

    tr_txt = df_train[cols]
    te_txt = df_test[cols]

    if mode == "tabular_tfidf":
        tr_sp, te_sp, _ = fit_transform_tfidf_train_test(
            tr_txt, te_txt, max_features=tfidf_max_features
        )
        return tr_sp.toarray(), te_sp.toarray()

    if mode == "tabular_bert":
        tr_blocks: list[np.ndarray] = []
        te_blocks: list[np.ndarray] = []
        for col in cols:
            tr_blocks.append(bert_embed_series(df_train[col]))
            te_blocks.append(bert_embed_series(df_test[col]))
        return np.hstack(tr_blocks), np.hstack(te_blocks)

    raise ValueError(f"Unknown mode: {mode}")
