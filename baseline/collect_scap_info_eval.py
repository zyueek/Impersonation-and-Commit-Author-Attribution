#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


def _coerce_labels(predictions, true_labels) -> Tuple[list[int], list[int]]:
    if predictions is None or true_labels is None:
        return [], []
    if len(predictions) != len(true_labels):
        return [], []
    if len(predictions) == 0:
        return [], []

    if isinstance(predictions[0], (int, np.integer)) and isinstance(true_labels[0], (int, np.integer)):
        return [int(x) for x in predictions], [int(x) for x in true_labels]

    encoder = LabelEncoder()
    encoder.fit([str(x) for x in list(true_labels) + list(predictions)])
    pred_enc = encoder.transform([str(x) for x in predictions]).tolist()
    true_enc = encoder.transform([str(x) for x in true_labels]).tolist()
    return pred_enc, true_enc


def _compute_metrics(predictions, true_labels) -> dict[str, float]:
    pred, true = _coerce_labels(predictions, true_labels)
    if not pred:
        return {
            "accuracy": np.nan,
            "accuracy_by_author": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "auc": np.nan,
        }

    accuracy = accuracy_score(true, pred)
    precision_macro = precision_score(true, pred, average="macro", zero_division=0)
    recall_macro = recall_score(true, pred, average="macro", zero_division=0)
    f1_macro = f1_score(true, pred, average="macro", zero_division=0)

    auc = np.nan
    if len(set(true)) == 2:
        auc = roc_auc_score(true, pred)

    return {
        "accuracy": float(accuracy),
        "accuracy_by_author": float(recall_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "auc": float(auc) if not np.isnan(auc) else np.nan,
    }


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def _mean_confidence(test_items: Any) -> float:
    if not isinstance(test_items, list) or not test_items:
        return np.nan
    vals = []
    for it in test_items:
        if isinstance(it, dict) and "confidence" in it:
            v = _safe_float(it.get("confidence"))
            if not np.isnan(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def collect(base_dir: str, out_csv: str) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file != "evaluation_results_info.json":
                continue
            path = os.path.join(root, file)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            if str(meta.get("method", "")).lower().strip() not in {"scap_info", ""}:
                continue

            predictions = data.get("predictions", [])
            true_labels = data.get("true_labels", [])
            metrics = _compute_metrics(predictions, true_labels)

            reported_accuracy = data.get("accuracy", data.get("eval_accuracy", None))
            reported_accuracy = _safe_float(reported_accuracy) if reported_accuracy is not None else np.nan
            if not np.isnan(reported_accuracy):
                metrics["accuracy"] = reported_accuracy

            test_items = data.get("test_items", [])
            mean_conf = _mean_confidence(test_items)
            repo = os.path.basename(os.path.dirname(path))

            rows.append(
                {
                    "file_path": path,
                    "repo": repo,
                    "method": meta.get("method", "scap_info") or "scap_info",
                    "fragment_chars": meta.get("fragment_chars", np.nan),
                    "fragment_mode": meta.get("fragment_mode", ""),
                    "msg_bucket_chars": meta.get("msg_bucket_chars", np.nan),
                    "msg_bucket_words": meta.get("msg_bucket_words", np.nan),
                    "include_message_stats": bool(meta.get("include_message_stats", True)),
                    "include_filename_tokens": bool(meta.get("include_filename_tokens", False)),
                    "filename_mode": meta.get("filename_mode", ""),
                    "stats_bucket_lines": meta.get("stats_bucket_lines", np.nan),
                    "stats_bucket_chars": meta.get("stats_bucket_chars", np.nan),
                    "ngram_n": meta.get("ngram_n", np.nan),
                    "profile_k": meta.get("profile_k", np.nan),
                    "sample_k": meta.get("sample_k", np.nan),
                    "n_test": len(true_labels) if isinstance(true_labels, list) else np.nan,
                    "n_authors_test": len(set(true_labels)) if isinstance(true_labels, list) else np.nan,
                    "n_authors_total": meta.get("n_authors", np.nan),
                    "confidence_mean": mean_conf,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect SCAP Com-Info baseline evaluation into CSV.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to collect: go, java, js, php, python; or 'all'; or comma-separated list (e.g. 'python,js')",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory to search; supports {lang}. Default: baseline/scap_{lang}_info",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path; supports {lang}. Default: baseline/evaluation_results_scap_{lang}_info.csv",
    )
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    raw_langs = [p.strip().lower() for p in args.lang.split(",") if p.strip()]
    if not raw_langs:
        raise SystemExit("Invalid --lang")
    if "all" in raw_langs:
        langs = ["go", "java", "js", "php", "python"]
    else:
        langs = raw_langs

    def resolve_base_dir(lang: str) -> str:
        if args.base_dir is None:
            return os.path.join(this_dir, f"scap_{lang}_info")
        base = os.path.abspath(os.path.expanduser(args.base_dir))
        if "{lang}" in base:
            return base.format(lang=lang)
        if len(langs) > 1:
            return os.path.join(base, f"scap_{lang}_info")
        return base

    def resolve_out_csv(lang: str) -> str:
        if args.out_csv is None:
            return os.path.join(this_dir, f"evaluation_results_scap_{lang}_info.csv")
        out = os.path.abspath(os.path.expanduser(args.out_csv))
        if "{lang}" in out:
            return out.format(lang=lang)
        if len(langs) > 1:
            if out.lower().endswith(".csv"):
                root, ext = os.path.splitext(out)
                return f"{root}_{lang}{ext}"
            return os.path.join(out, f"evaluation_results_scap_{lang}_info.csv")
        return out

    for lang in langs:
        base_dir = resolve_base_dir(lang)
        out_csv = resolve_out_csv(lang)
        df = collect(base_dir=base_dir, out_csv=out_csv)
        print(f"Saved: {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
