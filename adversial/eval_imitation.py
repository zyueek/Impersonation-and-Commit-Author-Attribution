#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from adversial.common import Metrics, seed_everything, write_json
from adversial.imitate_style import targeted_attack
from adversial.load_data import load_commit_json
from adversial.style_profile import build_profile


def _resolve_device(device: str):
    import torch

    device = (device or "auto").lower().strip()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --gcb_device=cuda but CUDA is not available")
        return torch.device("cuda")
    raise ValueError("Invalid --gcb_device (use: auto|cpu|cuda)")


@lru_cache(maxsize=16)
def _load_hf_encoder_cached(model_name: str, local_files_only: bool):
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(local_files_only))
    try:
        model = AutoModel.from_pretrained(model_name, local_files_only=bool(local_files_only), add_pooling_layer=False)
    except TypeError:
        model = AutoModel.from_pretrained(model_name, local_files_only=bool(local_files_only))
    return tok, model


@lru_cache(maxsize=8)
def _load_t5_encoder_cached(model_name: str, local_files_only: bool):
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=bool(local_files_only))
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = T5ForConditionalGeneration.from_pretrained(model_name, local_files_only=bool(local_files_only))
    return tok, model


def _fit_logreg_and_predict(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    solver: str,
    max_iter: int,
    scale: bool,
) -> list[int]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    solver = (solver or "lbfgs").strip().lower()
    if solver not in {"lbfgs", "saga", "newton-cg"}:
        raise ValueError("Invalid embed solver (use: lbfgs|saga|newton-cg)")

    clf = LogisticRegression(
        max_iter=int(max_iter),
        solver=solver,
        n_jobs=-1 if solver in {"saga"} else None,
        multi_class="auto",
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    return clf.predict(x_test).astype(int).tolist()


def _embed_hf_cls(
    *,
    texts: list[str],
    model_name: str,
    local_files_only: bool,
    device: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    dev = _resolve_device(device)

    tok, model = _load_hf_encoder_cached(model_name, bool(local_files_only))
    model.to(dev)
    model.eval()

    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), int(batch_size)):
            batch = texts[i : i + int(batch_size)]
            enc = tok(batch, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
            vecs.append(cls)

    hidden = int(getattr(model.config, "hidden_size", 768))
    return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, hidden), dtype=np.float32)


def _embed_t5_mean(
    *,
    texts: list[str],
    model_name: str,
    local_files_only: bool,
    device: str,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    dev = _resolve_device(device)

    tok, model = _load_t5_encoder_cached(model_name, bool(local_files_only))
    model.to(dev)
    model.eval()

    vecs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), int(batch_size)):
            batch = texts[i : i + int(batch_size)]
            enc = tok(batch, truncation=True, padding=True, max_length=int(max_length), return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model.encoder(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"), return_dict=True)
            hs = out.last_hidden_state  # (B, T, H)
            mask = enc.get("attention_mask")
            if mask is None:
                pooled = hs.mean(dim=1)
            else:
                m = mask.unsqueeze(-1).float()
                pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            vecs.append(pooled.detach().cpu().numpy())

    hidden = int(getattr(model.config, "d_model", 768))
    return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, hidden), dtype=np.float32)


def _embed_model_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    embedder,
    embedder_kwargs: dict,
    solver: str,
    max_iter: int,
    scale: bool,
) -> tuple[list[int], list[int], dict[int, str]]:
    seed_everything(seed)

    labels = sorted(set(train_df["label"].tolist()))
    label_map = {a: i for i, a in enumerate(labels)}
    inv = {v: str(k) for k, v in label_map.items()}

    y_train = np.array([label_map[v] for v in train_df["label"].tolist()], dtype=np.int64)
    y_test = np.array([label_map[v] for v in test_df["label"].tolist()], dtype=np.int64)

    x_train = embedder(texts=train_df["code"].tolist(), **embedder_kwargs)
    x_test = embedder(texts=test_df["code"].tolist(), **embedder_kwargs)

    pred = _fit_logreg_and_predict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        seed=seed,
        solver=solver,
        max_iter=max_iter,
        scale=scale,
    )
    return y_test.tolist(), pred, inv


def _embed_multimodal_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    code_embedder,
    code_embedder_kwargs: dict,
    text_embedder,
    text_embedder_kwargs: dict,
    include_message: bool,
    include_filename: bool,
    solver: str,
    max_iter: int,
    scale: bool,
) -> tuple[list[int], list[int], dict[int, str]]:
    seed_everything(seed)

    labels = sorted(set(train_df["label"].tolist()))
    label_map = {a: i for i, a in enumerate(labels)}
    inv = {v: str(k) for k, v in label_map.items()}

    y_train = np.array([label_map[v] for v in train_df["label"].tolist()], dtype=np.int64)
    y_test = np.array([label_map[v] for v in test_df["label"].tolist()], dtype=np.int64)

    x_code_train = code_embedder(texts=train_df["code"].tolist(), **code_embedder_kwargs)
    x_code_test = code_embedder(texts=test_df["code"].tolist(), **code_embedder_kwargs)

    feats_train = [x_code_train]
    feats_test = [x_code_test]

    if include_message:
        x_msg_train = text_embedder(texts=train_df["message"].astype(str).tolist(), **text_embedder_kwargs)
        x_msg_test = text_embedder(texts=test_df["message"].astype(str).tolist(), **text_embedder_kwargs)
        feats_train.append(x_msg_train)
        feats_test.append(x_msg_test)

    if include_filename:
        x_fn_train = text_embedder(texts=train_df["filename"].astype(str).tolist(), **text_embedder_kwargs)
        x_fn_test = text_embedder(texts=test_df["filename"].astype(str).tolist(), **text_embedder_kwargs)
        feats_train.append(x_fn_train)
        feats_test.append(x_fn_test)

    x_train = np.concatenate(feats_train, axis=1) if feats_train else np.zeros((len(train_df), 0), dtype=np.float32)
    x_test = np.concatenate(feats_test, axis=1) if feats_test else np.zeros((len(test_df), 0), dtype=np.float32)

    pred = _fit_logreg_and_predict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        seed=seed,
        solver=solver,
        max_iter=max_iter,
        scale=scale,
    )
    return y_test.tolist(), pred, inv


def _embed_model_train_predict_pair(
    train_df: pd.DataFrame,
    test_df_clean: pd.DataFrame,
    test_df_attacked: pd.DataFrame,
    *,
    seed: int,
    embedder,
    embedder_kwargs: dict,
    solver: str,
    max_iter: int,
    scale: bool,
) -> tuple[list[int], list[int], list[int], dict[int, str]]:
    seed_everything(seed)

    labels = sorted(set(train_df["label"].tolist()))
    label_map = {a: i for i, a in enumerate(labels)}
    inv = {v: str(k) for k, v in label_map.items()}

    y_train = np.array([label_map[v] for v in train_df["label"].tolist()], dtype=np.int64)
    y_clean = np.array([label_map[v] for v in test_df_clean["label"].tolist()], dtype=np.int64)
    y_attacked = np.array([label_map[v] for v in test_df_attacked["label"].tolist()], dtype=np.int64)
    if len(y_clean) != len(y_attacked) or (y_clean != y_attacked).any():
        raise ValueError("Clean and attacked test sets are misaligned (labels differ)")

    x_train = embedder(texts=train_df["code"].tolist(), **embedder_kwargs)
    x_clean = embedder(texts=test_df_clean["code"].tolist(), **embedder_kwargs)
    x_attacked = embedder(texts=test_df_attacked["code"].tolist(), **embedder_kwargs)

    from sklearn.preprocessing import StandardScaler

    scaler = None
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_clean = scaler.transform(x_clean)
        x_attacked = scaler.transform(x_attacked)

    from sklearn.linear_model import LogisticRegression

    solver = (solver or "lbfgs").strip().lower()
    if solver not in {"lbfgs", "saga", "newton-cg"}:
        raise ValueError("Invalid embed solver (use: lbfgs|saga|newton-cg)")
    clf = LogisticRegression(
        max_iter=int(max_iter),
        solver=solver,
        n_jobs=-1 if solver in {"saga"} else None,
        multi_class="auto",
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    pred_clean = clf.predict(x_clean).astype(int).tolist()
    pred_attack = clf.predict(x_attacked).astype(int).tolist()
    return y_clean.astype(int).tolist(), pred_clean, pred_attack, inv


def _embed_multimodal_train_predict_pair(
    train_df: pd.DataFrame,
    test_df_clean: pd.DataFrame,
    test_df_attacked: pd.DataFrame,
    *,
    seed: int,
    code_embedder,
    code_embedder_kwargs: dict,
    text_embedder,
    text_embedder_kwargs: dict,
    include_message: bool,
    include_filename: bool,
    solver: str,
    max_iter: int,
    scale: bool,
) -> tuple[list[int], list[int], list[int], dict[int, str]]:
    seed_everything(seed)

    labels = sorted(set(train_df["label"].tolist()))
    label_map = {a: i for i, a in enumerate(labels)}
    inv = {v: str(k) for k, v in label_map.items()}

    y_train = np.array([label_map[v] for v in train_df["label"].tolist()], dtype=np.int64)
    y_clean = np.array([label_map[v] for v in test_df_clean["label"].tolist()], dtype=np.int64)
    y_attacked = np.array([label_map[v] for v in test_df_attacked["label"].tolist()], dtype=np.int64)
    if len(y_clean) != len(y_attacked) or (y_clean != y_attacked).any():
        raise ValueError("Clean and attacked test sets are misaligned (labels differ)")

    x_code_train = code_embedder(texts=train_df["code"].tolist(), **code_embedder_kwargs)
    x_code_clean = code_embedder(texts=test_df_clean["code"].tolist(), **code_embedder_kwargs)
    x_code_attacked = code_embedder(texts=test_df_attacked["code"].tolist(), **code_embedder_kwargs)

    feats_train = [x_code_train]
    feats_clean = [x_code_clean]
    feats_attack = [x_code_attacked]

    if include_message:
        x_msg_train = text_embedder(texts=train_df["message"].astype(str).tolist(), **text_embedder_kwargs)
        x_msg_test = text_embedder(texts=test_df_clean["message"].astype(str).tolist(), **text_embedder_kwargs)
        feats_train.append(x_msg_train)
        feats_clean.append(x_msg_test)
        feats_attack.append(x_msg_test)

    if include_filename:
        x_fn_train = text_embedder(texts=train_df["filename"].astype(str).tolist(), **text_embedder_kwargs)
        x_fn_test = text_embedder(texts=test_df_clean["filename"].astype(str).tolist(), **text_embedder_kwargs)
        feats_train.append(x_fn_train)
        feats_clean.append(x_fn_test)
        feats_attack.append(x_fn_test)

    x_train = np.concatenate(feats_train, axis=1) if feats_train else np.zeros((len(train_df), 0), dtype=np.float32)
    x_clean = np.concatenate(feats_clean, axis=1) if feats_clean else np.zeros((len(test_df_clean), 0), dtype=np.float32)
    x_attacked = np.concatenate(feats_attack, axis=1) if feats_attack else np.zeros((len(test_df_attacked), 0), dtype=np.float32)

    from sklearn.preprocessing import StandardScaler

    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_clean = scaler.transform(x_clean)
        x_attacked = scaler.transform(x_attacked)

    from sklearn.linear_model import LogisticRegression

    solver = (solver or "lbfgs").strip().lower()
    if solver not in {"lbfgs", "saga", "newton-cg"}:
        raise ValueError("Invalid embed solver (use: lbfgs|saga|newton-cg)")
    clf = LogisticRegression(
        max_iter=int(max_iter),
        solver=solver,
        n_jobs=-1 if solver in {"saga"} else None,
        multi_class="auto",
        random_state=seed,
    )
    clf.fit(x_train, y_train)

    pred_clean = clf.predict(x_clean).astype(int).tolist()
    pred_attack = clf.predict(x_attacked).astype(int).tolist()
    return y_clean.astype(int).tolist(), pred_clean, pred_attack, inv


def _augment_train_for_gcb(
    train_df: pd.DataFrame,
    *,
    seed: int,
    mode: str,
    n_copies: int,
    p: float,
    use_ast: bool,
    token_rename_max: int,
) -> pd.DataFrame:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return train_df
    if mode != "imitate_random":
        raise ValueError("Invalid --gcb_train_aug (use: none|imitate_random)")

    rng = random.Random(seed)
    authors = sorted({str(a) for a in train_df["author_id"].tolist() if a is not None and str(a)})
    if len(authors) < 2:
        return train_df

    per_author = {a: [] for a in authors}
    for row in train_df.to_dict(orient="records"):
        a = str(row.get("author_id") or "")
        if a in per_author:
            per_author[a].append(row["code"])
    profiles = {a: build_profile(per_author[a]) for a in authors}

    augmented_rows: list[dict] = []
    for row in train_df.to_dict(orient="records"):
        true_author = str(row.get("author_id") or "")
        raw = row.get("raw") or {}
        language = str(raw.get("language", "") or "")
        for _ in range(int(n_copies)):
            if rng.random() > float(p):
                continue
            if true_author:
                target = rng.choice([a for a in authors if a != true_author])
            else:
                target = rng.choice(authors)
            res = targeted_attack(
                text=row["code"],
                target_label=target,
                target_profile=profiles[target],
                language=language,
                use_ast=use_ast,
                token_rename_max=int(token_rename_max),
                rng=rng,
            )
            row2 = dict(row)
            row2["code"] = res.attacked_text
            row2["augmented"] = True
            augmented_rows.append(row2)

    if not augmented_rows:
        return train_df
    return pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)


def _compute_metrics(y_true, y_pred) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return Metrics(
        accuracy=acc,
        precision_macro=float(p_ma),
        recall_macro=float(r_ma),
        f1_macro=float(f_ma),
        precision_micro=float(p_mi),
        recall_micro=float(r_mi),
        f1_micro=float(f_mi),
        precision_weighted=float(p_w),
        recall_weighted=float(r_w),
        f1_weighted=float(f_w),
    )


def _rf_baseline_train_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, *, seed: int):
    return _rf_baseline_train_predict_mode(train_df, test_df, seed=seed, mode="weak")


def _rf_baseline_train_predict_mode(train_df: pd.DataFrame, test_df: pd.DataFrame, *, seed: int, mode: str):
    # Reuse baseline logic by importing from the repo’s baseline scripts (no edits to baseline/ needed).
    import importlib.util

    mode = (mode or "weak").strip().lower()
    if mode not in {"weak", "weak_info", "strong"}:
        raise ValueError("Invalid rf mode (use: weak|weak_info|strong)")
    info_mode = mode in {"weak_info", "strong"}

    baseline_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "baseline",
        "train_rf_baseline.py" if not info_mode else "train_rf_baseline_info.py",
    )
    spec = importlib.util.spec_from_file_location("_rf_info", baseline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {baseline_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[assignment]

    # Build synthetic items in the expected shape (added_code/message/filename/emailname).
    def to_item(row) -> dict:
        return {
            "added_code": row["code"].splitlines(),
            "message": row.get("message", ""),
            "filename": row.get("filename", ""),
            "emailname": str(row["label"]),
        }

    train_items = [to_item(r) for r in train_df.to_dict(orient="records")]
    test_items = [to_item(r) for r in test_df.to_dict(orient="records")]

    # Use the same text builder used by the baseline.
    rng = np.random.RandomState(seed)
    if not info_mode:
        build = lambda it: mod._make_fragment(  # noqa: E731
            mod._build_text(it, include_message=False, include_filename=False),
            fragment_chars=80,
            fragment_mode="random",
            rng=rng,
        )
    else:
        build = lambda it: mod._build_text(  # noqa: E731
            it,
            fragment_chars=80,
            fragment_mode="random",
            rng=rng,
            msg_bucket_chars=50,
            msg_bucket_words=10,
            include_message_stats=True,
            include_filename_tokens=True,
            filename_mode="full",
            stats_bucket_lines=5,
            stats_bucket_chars=50,
        )
    x_train = [build(it) for it in train_items]
    x_test = [build(it) for it in test_items]

    # Labels: baseline expects ints.
    labels = sorted({it["emailname"] for it in train_items})
    label_map = {a: i for i, a in enumerate(labels)}
    y_train = [label_map[it["emailname"]] for it in train_items]
    y_test = [label_map[it["emailname"]] for it in test_items]

    if not info_mode:
        # Match baseline weak settings: word TF-IDF only.
        pipe = mod._make_pipeline(
            word_max_features=800,
            char_max_features=800,
            rf=mod._rf_params(max_depth=5),
            seed=seed,
            use_char=False,
            use_layout=False,
        )
    else:
        pipe = mod._make_pipeline(
            word_max_features=800,
            char_max_features=800,
            rf=mod._rf_params("weak", max_depth=5),
            seed=seed,
            use_char=True,
            use_layout=True,
        )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test).astype(int).tolist()
    inv = {v: k for k, v in label_map.items()}
    return y_test, pred, inv


def _gcb_embed_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
    local_files_only: bool,
    device: str,
    solver: str,
    max_iter: int,
    scale: bool,
):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    seed_everything(seed)
    torch.manual_seed(seed)

    dev = _resolve_device(device)

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    try:
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only, add_pooling_layer=False)
    except TypeError:
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(dev)
    model.eval()

    labels = sorted(set(train_df["label"].tolist()))
    label_map = {a: i for i, a in enumerate(labels)}
    inv = {v: k for k, v in label_map.items()}

    def embed(texts: list[str], batch_size: int = 8) -> np.ndarray:
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tok(batch, truncation=True, padding=True, max_length=256, return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                out = model(**enc)
                cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
                vecs.append(cls)
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    x_train = embed(train_df["code"].tolist())
    x_test = embed(test_df["code"].tolist())
    y_train = np.array([label_map[v] for v in train_df["label"].tolist()], dtype=np.int64)
    y_test = np.array([label_map[v] for v in test_df["label"].tolist()], dtype=np.int64)

    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    solver = (solver or "lbfgs").strip().lower()
    if solver not in {"lbfgs", "saga", "newton-cg"}:
        raise ValueError("Invalid gcb embed solver (use: lbfgs|saga|newton-cg)")

    clf = LogisticRegression(
        max_iter=int(max_iter),
        solver=solver,
        n_jobs=-1 if solver in {"saga"} else None,
        multi_class="auto",
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test).astype(int).tolist()
    return y_test.tolist(), pred, inv


def _codebert_embed_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
    local_files_only: bool,
    device: str,
    solver: str,
    max_iter: int,
    scale: bool,
    max_length: int,
    batch_size: int,
):
    return _embed_model_train_predict(
        train_df,
        test_df,
        seed=seed,
        embedder=_embed_hf_cls,
        embedder_kwargs={
            "model_name": model_name,
            "local_files_only": bool(local_files_only),
            "device": device,
            "max_length": int(max_length),
            "batch_size": int(batch_size),
        },
        solver=solver,
        max_iter=int(max_iter),
        scale=bool(scale),
    )


def _codet5_embed_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
    local_files_only: bool,
    device: str,
    solver: str,
    max_iter: int,
    scale: bool,
    max_length: int,
    batch_size: int,
):
    return _embed_model_train_predict(
        train_df,
        test_df,
        seed=seed,
        embedder=_embed_t5_mean,
        embedder_kwargs={
            "model_name": model_name,
            "local_files_only": bool(local_files_only),
            "device": device,
            "max_length": int(max_length),
            "batch_size": int(batch_size),
        },
        solver=solver,
        max_iter=int(max_iter),
        scale=bool(scale),
    )


def _info_embed_train_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    code_embedder,
    code_embedder_kwargs: dict,
    text_model_name: str,
    local_files_only: bool,
    device: str,
    solver: str,
    max_iter: int,
    scale: bool,
    code_max_length: int,
    text_max_length: int,
    batch_size: int,
    include_message: bool,
    include_filename: bool,
):
    return _embed_multimodal_train_predict(
        train_df,
        test_df,
        seed=seed,
        code_embedder=code_embedder,
        code_embedder_kwargs={
            **code_embedder_kwargs,
            "local_files_only": bool(local_files_only),
            "device": device,
            "max_length": int(code_max_length),
            "batch_size": int(batch_size),
        },
        text_embedder=_embed_hf_cls,
        text_embedder_kwargs={
            "model_name": text_model_name,
            "local_files_only": bool(local_files_only),
            "device": device,
            "max_length": int(text_max_length),
            "batch_size": int(batch_size),
        },
        include_message=bool(include_message),
        include_filename=bool(include_filename),
        solver=solver,
        max_iter=int(max_iter),
        scale=bool(scale),
    )


def _gcb_finetune_train(
    train_df: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
    local_files_only: bool,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    out_dir: str,
    device: str,
    fp16: bool,
):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    seed_everything(seed)
    torch.manual_seed(seed)

    dev = _resolve_device(device)
    use_cuda = dev.type == "cuda"

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        num_labels=len(sorted(set(train_df["label"].tolist()))),
    )
    model.to(dev)

    class _Ds(torch.utils.data.Dataset):
        def __init__(self, texts: list[str], ys: list[int]):
            self.enc = tok(texts, truncation=True, padding=True, max_length=max_length)
            self.ys = ys

        def __len__(self) -> int:
            return len(self.ys)

        def __getitem__(self, idx: int):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[idx])
            return item

    labels = sorted(set(train_df["label"].tolist()))
    label_map = {a: i for i, a in enumerate(labels)}
    inv = {v: k for k, v in label_map.items()}

    x_train = train_df["code"].tolist()
    y_train = [label_map[v] for v in train_df["label"].tolist()]

    train_ds = _Ds(x_train, y_train)

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "gcb_finetune_tmp"),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        no_cuda=not use_cuda,
        fp16=bool(fp16) if use_cuda else False,
        seed=int(seed),
        data_seed=int(seed),
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()
    return tok, model, label_map, inv


def _gcb_finetune_predict(
    *,
    tok,
    model,
    label_map: dict,
    df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: str,
) -> tuple[list[int], list[int]]:
    import torch
    dev = _resolve_device(device)
    model.to(dev)

    class _Ds(torch.utils.data.Dataset):
        def __init__(self, texts: list[str], ys: list[int]):
            self.enc = tok(texts, truncation=True, padding=True, max_length=max_length)
            self.ys = ys

        def __len__(self) -> int:
            return len(self.ys)

        def __getitem__(self, idx: int):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.ys[idx])
            return item

    xs = df["code"].tolist()
    ys = [label_map[v] for v in df["label"].tolist()]
    ds = _Ds(xs, ys)

    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy().astype(int).tolist()
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(dim=1).cpu().numpy().astype(int).tolist()
            y_true.extend(labels)
            y_pred.extend(preds)
    return y_true, y_pred


def _make_attacked_test(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    fixed_target: str | None,
    use_ast: bool,
    token_rename_max: int,
) -> tuple[pd.DataFrame, list[str]]:
    rng = random.Random(seed)
    authors = sorted({str(a) for a in train_df["author_id"].tolist() if a is not None and str(a)})
    if not authors:
        raise ValueError("No author ids available for style imitation (need 'emailname' or 'author' in JSON)")

    per_author = {a: [] for a in authors}
    for row in train_df.to_dict(orient="records"):
        a = row.get("author_id")
        a = str(a) if a is not None else ""
        if a in per_author:
            per_author[a].append(row["code"])
    profiles = {a: build_profile(per_author[a]) for a in authors}

    attacked = []
    targets = []
    for row in test_df.to_dict(orient="records"):
        true_a = str(row.get("author_id") or "")
        raw = row.get("raw") or {}
        language = str(raw.get("language", "") or "")
        if fixed_target:
            target = fixed_target
            if target == true_a and len(authors) > 1:
                target = rng.choice([a for a in authors if a != true_a])
        else:
            target = rng.choice([a for a in authors if a != true_a]) if len(authors) > 1 else true_a
        res = targeted_attack(
            text=row["code"],
            target_label=target,
            target_profile=profiles[target],
            language=language,
            use_ast=use_ast,
            token_rename_max=int(token_rename_max),
            rng=rng,
        )
        row2 = dict(row)
        row2["code"] = res.attacked_text
        row2["attack_target"] = res.target_label
        attacked.append(row2)
        targets.append(res.target_label)
    return pd.DataFrame(attacked), targets


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate clean vs style-imitated commits (GraphCodeBERT + RF baseline).")
    ap.add_argument("--json_path", required=True)
    ap.add_argument("--task", choices=["authorship", "binary"], default="authorship")
    ap.add_argument(
        "--models",
        default="rf,gcb",
        help="Comma-separated: rf,gcb,lang,t5,gcb_info,lang_info,t5_info (all are embedding-based except gcb finetune).",
    )
    ap.add_argument(
        "--rf_mode",
        choices=["weak", "weak_info", "strong"],
        default="weak",
        help="weak: code-only weak RF; weak_info: weak RF with commit info features; strong: alias of weak_info for back-compat.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_samples_per_author", type=int, default=2)
    ap.add_argument("--fixed_target", type=str, default=None, help="Optional fixed target author id for targeted imitation")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--use_ast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Python AST/UDC transformations before text-level transforms (Python only).",
    )
    ap.add_argument(
        "--token_rename_max",
        type=int,
        default=0,
        help="If >0, perform token-aware identifier renaming for go/java/js/php (stronger attack).",
    )

    ap.add_argument("--gcb_model", default="microsoft/graphcodebert-base")
    ap.add_argument("--codebert_model", default="microsoft/codebert-base")
    ap.add_argument("--t5_model", default="Salesforce/codet5p-220m")
    ap.add_argument("--info_text_model", default="distilbert-base-uncased")
    ap.add_argument("--info_text_max_length", type=int, default=128)
    ap.add_argument("--gcb_local_files_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gcb_mode", choices=["embed", "finetune"], default="embed")
    ap.add_argument("--gcb_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--gcb_fp16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gcb_embed_solver", default="lbfgs", help="LogReg solver for --gcb_mode=embed (lbfgs|saga|newton-cg)")
    ap.add_argument("--gcb_embed_max_iter", type=int, default=400)
    ap.add_argument("--gcb_embed_scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gcb_train_aug", choices=["none", "imitate_random"], default="none")
    ap.add_argument("--gcb_train_aug_n", type=int, default=1, help="How many augmented copies per train sample (upper bound).")
    ap.add_argument("--gcb_train_aug_p", type=float, default=0.5, help="Probability of generating each augmented copy.")
    ap.add_argument(
        "--gcb_train_aug_use_ast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AST/CFG/UDC layer when creating augmentation (Python only).",
    )
    ap.add_argument("--gcb_epochs", type=float, default=1.0)
    ap.add_argument("--gcb_batch_size", type=int, default=8)
    ap.add_argument("--gcb_lr", type=float, default=5e-5)
    ap.add_argument("--gcb_max_length", type=int, default=256)
    args = ap.parse_args()

    seed_everything(args.seed)

    ds = load_commit_json(args.json_path, task=args.task, min_samples_per_author=args.min_samples_per_author)
    df = ds.df

    strat = df["label"] if args.task in {"authorship", "binary"} else None
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=strat)

    attacked_test_df, targets = _make_attacked_test(
        train_df=train_df,
        test_df=test_df,
        seed=args.seed,
        fixed_target=args.fixed_target,
        use_ast=bool(args.use_ast),
        token_rename_max=int(args.token_rename_max),
    )

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    results = {
        "json_path": os.path.abspath(args.json_path),
        "task": args.task,
        "seed": args.seed,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "models": {},
    }

    os.makedirs(args.out_dir, exist_ok=True)

    for model_name in models:
        if model_name == "rf":
            y_true, pred_clean, inv = _rf_baseline_train_predict_mode(train_df, test_df, seed=args.seed, mode=args.rf_mode)
            y_true2, pred_attack, _inv2 = _rf_baseline_train_predict_mode(train_df, attacked_test_df, seed=args.seed, mode=args.rf_mode)
            assert y_true == y_true2

            m_clean = _compute_metrics(y_true, pred_clean)
            m_attack = _compute_metrics(y_true, pred_attack)
            drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}

            targeted_success = None
            if args.task == "authorship":
                rev = {lbl: idx for idx, lbl in inv.items()}
                target_ids = [rev.get(t) for t in targets]
                ok = [(p == tid) for p, tid in zip(pred_attack, target_ids) if tid is not None]
                targeted_success = float(sum(ok) / len(ok)) if ok else None

            results["models"]["rf"] = {
                "clean": asdict(m_clean),
                "attacked": asdict(m_attack),
                "metric_drop": drop,
                "targeted_success_rate": targeted_success,
                "rf_mode": args.rf_mode,
            }

        elif model_name == "gcb":
            train_df_gcb = _augment_train_for_gcb(
                train_df,
                seed=args.seed,
                mode=args.gcb_train_aug,
                n_copies=args.gcb_train_aug_n,
                p=args.gcb_train_aug_p,
                use_ast=bool(args.gcb_train_aug_use_ast),
                token_rename_max=int(args.token_rename_max),
            )
            if args.gcb_mode == "finetune":
                tok, model, label_map, inv = _gcb_finetune_train(
                    train_df_gcb,
                    seed=args.seed,
                    model_name=args.gcb_model,
                    local_files_only=args.gcb_local_files_only,
                    epochs=args.gcb_epochs,
                    batch_size=args.gcb_batch_size,
                    learning_rate=args.gcb_lr,
                    max_length=args.gcb_max_length,
                    out_dir=args.out_dir,
                    device=args.gcb_device,
                    fp16=bool(args.gcb_fp16),
                )
                y_true, pred_clean = _gcb_finetune_predict(
                    tok=tok,
                    model=model,
                    label_map=label_map,
                    df=test_df,
                    max_length=args.gcb_max_length,
                    batch_size=args.gcb_batch_size,
                    device=args.gcb_device,
                )
                y_true2, pred_attack = _gcb_finetune_predict(
                    tok=tok,
                    model=model,
                    label_map=label_map,
                    df=attacked_test_df,
                    max_length=args.gcb_max_length,
                    batch_size=args.gcb_batch_size,
                    device=args.gcb_device,
                )
            else:
                y_true, pred_clean, pred_attack, inv = _embed_model_train_predict_pair(
                    train_df_gcb,
                    test_df,
                    attacked_test_df,
                    seed=args.seed,
                    embedder=_embed_hf_cls,
                    embedder_kwargs={
                        "model_name": args.gcb_model,
                        "local_files_only": bool(args.gcb_local_files_only),
                        "device": args.gcb_device,
                        "max_length": int(args.gcb_max_length),
                        "batch_size": int(args.gcb_batch_size),
                    },
                    solver=args.gcb_embed_solver,
                    max_iter=args.gcb_embed_max_iter,
                    scale=bool(args.gcb_embed_scale),
                )
            if args.gcb_mode == "finetune":
                assert y_true == y_true2

            m_clean = _compute_metrics(y_true, pred_clean)
            m_attack = _compute_metrics(y_true, pred_attack)
            drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}

            targeted_success = None
            if args.task == "authorship":
                rev = {lbl: idx for idx, lbl in inv.items()}
                target_ids = [rev.get(t) for t in targets]
                ok = [(p == tid) for p, tid in zip(pred_attack, target_ids) if tid is not None]
                targeted_success = float(sum(ok) / len(ok)) if ok else None

            results["models"]["gcb"] = {
                "clean": asdict(m_clean),
                "attacked": asdict(m_attack),
                "metric_drop": drop,
                "targeted_success_rate": targeted_success,
                "gcb_mode": args.gcb_mode,
                "gcb_train_aug": args.gcb_train_aug,
                "gcb_train_aug_n": int(args.gcb_train_aug_n),
                "gcb_train_aug_p": float(args.gcb_train_aug_p),
            }
        elif model_name in {"lang", "t5", "gcb_info", "lang_info", "t5_info"}:
            if model_name == "lang":
                y_true, pred_clean, pred_attack, inv = _embed_model_train_predict_pair(
                    train_df,
                    test_df,
                    attacked_test_df,
                    seed=args.seed,
                    embedder=_embed_hf_cls,
                    embedder_kwargs={
                        "model_name": args.codebert_model,
                        "local_files_only": bool(args.gcb_local_files_only),
                        "device": args.gcb_device,
                        "max_length": int(args.gcb_max_length),
                        "batch_size": int(args.gcb_batch_size),
                    },
                    solver=args.gcb_embed_solver,
                    max_iter=args.gcb_embed_max_iter,
                    scale=bool(args.gcb_embed_scale),
                )
            elif model_name == "t5":
                y_true, pred_clean, pred_attack, inv = _embed_model_train_predict_pair(
                    train_df,
                    test_df,
                    attacked_test_df,
                    seed=args.seed,
                    embedder=_embed_t5_mean,
                    embedder_kwargs={
                        "model_name": args.t5_model,
                        "local_files_only": bool(args.gcb_local_files_only),
                        "device": args.gcb_device,
                        "max_length": int(args.gcb_max_length),
                        "batch_size": int(args.gcb_batch_size),
                    },
                    solver=args.gcb_embed_solver,
                    max_iter=args.gcb_embed_max_iter,
                    scale=bool(args.gcb_embed_scale),
                )
            else:
                if model_name == "gcb_info":
                    code_embedder = _embed_hf_cls
                    code_kwargs = {"model_name": args.gcb_model}
                elif model_name == "lang_info":
                    code_embedder = _embed_hf_cls
                    code_kwargs = {"model_name": args.codebert_model}
                else:
                    code_embedder = _embed_t5_mean
                    code_kwargs = {"model_name": args.t5_model}

                y_true, pred_clean, pred_attack, inv = _embed_multimodal_train_predict_pair(
                    train_df,
                    test_df,
                    attacked_test_df,
                    seed=args.seed,
                    code_embedder=code_embedder,
                    code_embedder_kwargs={
                        **code_kwargs,
                        "local_files_only": bool(args.gcb_local_files_only),
                        "device": args.gcb_device,
                        "max_length": int(args.gcb_max_length),
                        "batch_size": int(args.gcb_batch_size),
                    },
                    text_embedder=_embed_hf_cls,
                    text_embedder_kwargs={
                        "model_name": args.info_text_model,
                        "local_files_only": bool(args.gcb_local_files_only),
                        "device": args.gcb_device,
                        "max_length": int(args.info_text_max_length),
                        "batch_size": int(args.gcb_batch_size),
                    },
                    include_message=True,
                    include_filename=True,
                    solver=args.gcb_embed_solver,
                    max_iter=args.gcb_embed_max_iter,
                    scale=bool(args.gcb_embed_scale),
                )
            m_clean = _compute_metrics(y_true, pred_clean)
            m_attack = _compute_metrics(y_true, pred_attack)
            drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}

            targeted_success = None
            if args.task == "authorship":
                rev = {lbl: idx for idx, lbl in inv.items()}
                target_ids = [rev.get(t) for t in targets]
                ok = [(p == tid) for p, tid in zip(pred_attack, target_ids) if tid is not None]
                targeted_success = float(sum(ok) / len(ok)) if ok else None

            results["models"][model_name] = {
                "clean": asdict(m_clean),
                "attacked": asdict(m_attack),
                "metric_drop": drop,
                "targeted_success_rate": targeted_success,
                "embed_solver": args.gcb_embed_solver,
                "embed_max_iter": int(args.gcb_embed_max_iter),
                "embed_scale": bool(args.gcb_embed_scale),
            }
        else:
            raise SystemExit(f"Unknown model: {model_name}")

    out_json = os.path.join(args.out_dir, "imitation_results.json")
    write_json(out_json, results)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
