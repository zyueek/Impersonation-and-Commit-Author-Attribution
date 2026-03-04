#!/usr/bin/env python3
"""
SCAP baseline (Frantzeskou et al., byte-level n-gram author profiles) adapted to this repo's dataset format:
language/combined_{lang}/*.json where each item has added_code (list[str]) and emailname.

This is NOT a bit-for-bit reproduction of any paper's dataset; it reproduces the SCAP-style method on the
same per-repository split setting used elsewhere in this repo.
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def _join_code_lines(item: dict) -> str:
    added = item.get("added_code", [])
    if isinstance(added, list):
        return "\n".join(map(str, added))
    return str(added or "")


def _build_text(item: dict, include_message: bool, include_filename: bool) -> str:
    parts = [_join_code_lines(item)]
    if include_message:
        parts.append(str(item.get("message", "")))
    if include_filename:
        parts.append(str(item.get("filename", "")))
    return "\n".join(p for p in parts if p)


def _make_fragment(text: str, fragment_chars: int | None, fragment_mode: str, rng: np.random.RandomState) -> str:
    if fragment_chars is None:
        return text
    if fragment_chars <= 0:
        return ""
    if len(text) <= fragment_chars:
        return text

    mode = fragment_mode.lower().strip()
    if mode == "head":
        return text[:fragment_chars]
    if mode == "tail":
        return text[-fragment_chars:]
    if mode == "random":
        start = int(rng.randint(0, len(text) - fragment_chars + 1))
        return text[start : start + fragment_chars]
    raise ValueError(f"Unknown fragment_mode={fragment_mode!r} (use: head|tail|random)")


def _bytes_ngrams(text: str, n: int) -> Counter:
    b = text.encode("utf-8", errors="ignore")
    if len(b) < n or n <= 0:
        return Counter()
    return Counter(b[i : i + n] for i in range(len(b) - n + 1))


def _topk(counter: Counter, k: int) -> list[bytes]:
    if k <= 0:
        return []
    return [ng for ng, _ in counter.most_common(k)]


def _load_df(
    json_path: str,
    *,
    min_samples_per_author: int,
    max_samples_per_author: int | None,
    include_message: bool,
    include_filename: bool,
    fragment_chars: int | None,
    fragment_mode: str,
    seed: int,
) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    rows = []
    for item in data:
        text = _build_text(item, include_message=include_message, include_filename=include_filename)
        text = _make_fragment(text, fragment_chars=fragment_chars, fragment_mode=fragment_mode, rng=rng)
        rows.append({"text": text, "emailname": item.get("emailname", ""), "raw": item})

    df = pd.DataFrame(rows)
    df = df[df["emailname"].astype(bool)]
    df = df[df["text"].astype(bool)]

    counts = df["emailname"].value_counts()
    keep = counts[counts >= min_samples_per_author].index
    df = df[df["emailname"].isin(keep)].copy()
    if df.empty:
        return df

    if max_samples_per_author is not None:
        df = (
            df.groupby("emailname", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max_samples_per_author), random_state=seed))
            .reset_index(drop=True)
        )

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _build_author_profiles(train_df: pd.DataFrame, label_map: dict[str, int], n: int, profile_k: int) -> dict[int, set[bytes]]:
    by_author: dict[int, Counter] = {idx: Counter() for idx in label_map.values()}
    for _, row in train_df.iterrows():
        author = row["emailname"]
        idx = label_map[author]
        by_author[idx].update(_bytes_ngrams(row["text"], n=n))
    return {idx: set(_topk(cnt, profile_k)) for idx, cnt in by_author.items()}


def _predict_one(
    text: str,
    author_profiles: dict[int, set[bytes]],
    n: int,
    sample_k: int,
) -> tuple[int, float]:
    sample_profile = set(_topk(_bytes_ngrams(text, n=n), sample_k))
    if not sample_profile:
        # Arbitrary fallback: return smallest class id with confidence 0
        best = min(author_profiles.keys())
        return best, 0.0

    best_idx = None
    best_score = -1
    for idx, prof in author_profiles.items():
        score = len(sample_profile & prof)  # intersection count
        if score > best_score:
            best_score = score
            best_idx = idx

    confidence = float(best_score) / float(len(sample_profile)) if sample_profile else 0.0
    return int(best_idx), float(confidence)


def _save_results(
    output_dir: str,
    accuracy: float,
    predicted: list[int],
    true_labels: list[int],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    label_map: dict[str, int],
    confidence: list[float],
    meta: dict,
):
    os.makedirs(output_dir, exist_ok=True)
    inv_label_map = {v: k for k, v in label_map.items()}

    per_label = {}
    for author, idx in label_map.items():
        train_count = int((train_df["emailname"] == author).sum())
        mask = [i for i, lbl in enumerate(true_labels) if lbl == idx]
        if mask:
            correct = sum(1 for i in mask if predicted[i] == idx)
            acc = correct / len(mask)
        else:
            acc = None
        per_label[author] = {"accuracy": acc, "count": train_count}

    items = []
    for i, row in enumerate(test_df.to_dict(orient="records")):
        raw = row.pop("raw", None) or {}
        row["raw"] = raw
        row["predicted_author"] = inv_label_map.get(predicted[i], str(predicted[i]))
        row["true_author"] = inv_label_map.get(true_labels[i], str(true_labels[i]))
        row["confidence"] = float(confidence[i])
        items.append(row)

    results = {
        "accuracy": float(accuracy),
        "predictions": predicted,
        "true_labels": true_labels,
        "per_label_metrics": per_label,
        "test_items": items,
        "meta": meta,
    }

    out_path = os.path.join(output_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


def run_one(
    json_path: str,
    out_root: str,
    *,
    min_samples_per_author: int,
    max_samples_per_author: int | None,
    test_size: float,
    seed: int,
    include_message: bool,
    include_filename: bool,
    fragment_chars: int | None,
    fragment_mode: str,
    ngram_n: int,
    profile_k: int,
    sample_k: int,
):
    print(f"Processing: {json_path}")
    df = _load_df(
        json_path,
        min_samples_per_author=min_samples_per_author,
        max_samples_per_author=max_samples_per_author,
        include_message=include_message,
        include_filename=include_filename,
        fragment_chars=fragment_chars,
        fragment_mode=fragment_mode,
        seed=seed,
    )
    if df.empty or df["emailname"].nunique() < 2:
        print("Skip: not enough data after filtering.")
        return

    label_map = {author: i for i, author in enumerate(sorted(df["emailname"].unique()))}
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["emailname"])

    author_profiles = _build_author_profiles(train_df, label_map, n=ngram_n, profile_k=profile_k)

    predicted = []
    confidence = []
    for text in test_df["text"].tolist():
        pred, conf = _predict_one(text, author_profiles, n=ngram_n, sample_k=sample_k)
        predicted.append(pred)
        confidence.append(conf)

    true = test_df["emailname"].map(label_map).astype(int).tolist()
    acc = accuracy_score(true, predicted)

    out_dir = os.path.join(out_root, os.path.basename(json_path).replace(".json", ""))
    _save_results(
        out_dir,
        accuracy=acc,
        predicted=predicted,
        true_labels=true,
        test_df=test_df,
        train_df=train_df,
        label_map=label_map,
        confidence=confidence,
        meta={
            "method": "scap",
            "json_path": json_path,
            "n_samples": int(len(df)),
            "n_authors": int(df["emailname"].nunique()),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "include_message": include_message,
            "include_filename": include_filename,
            "fragment_chars": fragment_chars,
            "fragment_mode": fragment_mode,
            "ngram_n": ngram_n,
            "profile_k": profile_k,
            "sample_k": sample_k,
            "min_samples_per_author": min_samples_per_author,
            "max_samples_per_author": max_samples_per_author,
            "test_size": test_size,
            "seed": seed,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="SCAP (byte n-gram author profile) baseline on this repo's dataset.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to run: go, java, js, php, python; or 'all'; or comma-separated list (e.g. 'python,js')",
    )
    parser.add_argument("--json_dir", type=str, default=None)
    parser.add_argument("--json_glob", type=str, default="*.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: baseline/scap_{lang}")

    parser.add_argument("--min_samples_per_author", type=int, default=5)
    parser.add_argument("--max_samples_per_author", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--include_message", action="store_true")
    parser.add_argument("--include_filename", action="store_true")

    # Default to a harder setting (shorter fragments).
    parser.add_argument("--fragment_chars", type=int, default=20)
    parser.add_argument("--fragment_mode", type=str, default="random", choices=["head", "tail", "random"])

    parser.add_argument("--ngram_n", type=int, default=5)
    parser.add_argument("--profile_k", type=int, default=200)
    parser.add_argument("--sample_k", type=int, default=120)

    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, os.pardir))

    raw_langs = [p.strip().lower() for p in args.lang.split(",") if p.strip()]
    if not raw_langs:
        raise SystemExit("Invalid --lang")
    if "all" in raw_langs:
        langs = ["go", "java", "js", "php", "python"]
    else:
        langs = raw_langs

    def resolve_json_dir(base: str | None, lang: str) -> str:
        if base is None:
            return os.path.join(repo_root, "language", f"combined_{lang}")
        base = os.path.abspath(os.path.expanduser(base))
        if glob.glob(os.path.join(base, args.json_glob)):
            return base
        candidate = os.path.join(base, f"combined_{lang}")
        if os.path.isdir(candidate):
            return candidate
        return base

    def resolve_out_dir(base: str | None, lang: str) -> str:
        if base is None:
            return os.path.join(this_dir, f"scap_{lang}")
        base = os.path.abspath(os.path.expanduser(base))
        if len(langs) > 1 and "{lang}" not in base:
            return os.path.join(base, f"scap_{lang}")
        return base.format(lang=lang)

    for lang in langs:
        json_dir = resolve_json_dir(args.json_dir, lang)
        out_dir = resolve_out_dir(args.out_dir, lang)
        json_files = sorted(glob.glob(os.path.join(json_dir, args.json_glob)))
        if not json_files:
            msg = f"No JSON files matched for lang={lang}: {os.path.join(json_dir, args.json_glob)}"
            if len(langs) > 1:
                print(msg)
                continue
            raise SystemExit(msg)

        for json_path in json_files:
            try:
                run_one(
                    json_path=json_path,
                    out_root=out_dir,
                    min_samples_per_author=args.min_samples_per_author,
                    max_samples_per_author=args.max_samples_per_author,
                    test_size=args.test_size,
                    seed=args.seed,
                    include_message=args.include_message,
                    include_filename=args.include_filename,
                    fragment_chars=args.fragment_chars,
                    fragment_mode=args.fragment_mode,
                    ngram_n=args.ngram_n,
                    profile_k=args.profile_k,
                    sample_k=args.sample_k,
                )
            except Exception as e:
                print(f"Error processing {json_path}: {e}")


if __name__ == "__main__":
    main()
