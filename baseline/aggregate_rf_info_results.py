#!/usr/bin/env python3
import argparse
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


_DEFAULT_PATTERN = r"^evaluation_results_rf_(go|java|js|php|python)_weak_info\.csv$"


def _discover_inputs(baseline_dir: str, filename_re: re.Pattern) -> list[str]:
    return sorted(os.path.join(baseline_dir, n) for n in os.listdir(baseline_dir) if filename_re.match(n))


def _infer_lang(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"evaluation_results_rf_([a-z]+)_weak_info\.csv$", base)
    return m.group(1) if m else "unknown"


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _safe_mean(x: pd.Series) -> float:
    x = _to_float(x)
    if x.notna().any():
        return float(x.mean(skipna=True))
    return float("nan")


def _concat_unique(values: Iterable[str]) -> str:
    uniq = sorted({str(v) for v in values if str(v) and str(v) != "nan"})
    return ";".join(uniq)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", ""}:
        return False
    return False


def aggregate(inputs: list[str], out_repo_csv: str, out_overall_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        df = pd.read_csv(path)
        df["lang"] = _infer_lang(path)
        df["source_csv"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise SystemExit("No input CSVs found.")

    all_df = pd.concat(frames, ignore_index=True)

    for col in [
        "accuracy",
        "accuracy_by_author",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc",
        "confidence_mean",
        "n_test",
        "n_authors_test",
        "n_authors_total",
        "fragment_chars",
        "msg_bucket_chars",
        "msg_bucket_words",
        "stats_bucket_lines",
        "stats_bucket_chars",
        "train_count_min",
        "train_count_mean",
        "train_count_max",
    ]:
        if col in all_df.columns:
            all_df[col] = _to_float(all_df[col])

    if "include_message_stats" in all_df.columns:
        all_df["include_message_stats"] = all_df["include_message_stats"].map(_to_bool)
    if "include_filename_tokens" in all_df.columns:
        all_df["include_filename_tokens"] = all_df["include_filename_tokens"].map(_to_bool)

    group_cols = [
        "repo",
        "method",
        "feature_set",
        "fragment_chars",
        "fragment_mode",
        "msg_bucket_chars",
        "msg_bucket_words",
        "include_message_stats",
        "include_filename_tokens",
        "filename_mode",
        "stats_bucket_lines",
        "stats_bucket_chars",
    ]
    for col in group_cols:
        if col not in all_df.columns:
            raise SystemExit(f"Missing required column {col!r} in inputs; regenerate with baseline/collect_rf_info_eval.py")

    grouped = all_df.groupby(group_cols, dropna=False)
    repo_rows = []
    for key, g in grouped:
        (
            repo,
            method,
            feature_set,
            fragment_chars,
            fragment_mode,
            msg_bucket_chars,
            msg_bucket_words,
            include_message_stats,
            include_filename_tokens,
            filename_mode,
            stats_bucket_lines,
            stats_bucket_chars,
        ) = key
        langs = _concat_unique(g["lang"].tolist())
        n_langs = int(g["lang"].nunique())

        repo_rows.append(
            {
                "repo": repo,
                "method": method,
                "feature_set": feature_set,
                "fragment_chars": fragment_chars,
                "fragment_mode": fragment_mode,
                "msg_bucket_chars": msg_bucket_chars,
                "msg_bucket_words": msg_bucket_words,
                "include_message_stats": bool(include_message_stats),
                "include_filename_tokens": bool(include_filename_tokens),
                "filename_mode": filename_mode,
                "stats_bucket_lines": stats_bucket_lines,
                "stats_bucket_chars": stats_bucket_chars,
                "langs": langs,
                "n_langs": n_langs,
                "n_rows": int(len(g)),
                "accuracy": _safe_mean(g.get("accuracy", pd.Series(dtype=float))),
                "accuracy_by_author": _safe_mean(g.get("accuracy_by_author", pd.Series(dtype=float))),
                "precision_macro": _safe_mean(g.get("precision_macro", pd.Series(dtype=float))),
                "recall_macro": _safe_mean(g.get("recall_macro", pd.Series(dtype=float))),
                "f1_macro": _safe_mean(g.get("f1_macro", pd.Series(dtype=float))),
                "auc": _safe_mean(g.get("auc", pd.Series(dtype=float))),
                "confidence_mean": _safe_mean(g.get("confidence_mean", pd.Series(dtype=float))),
            }
        )

    repo_df = pd.DataFrame(repo_rows).sort_values(["n_langs", "repo"], ascending=[False, True])
    os.makedirs(os.path.dirname(out_repo_csv) or ".", exist_ok=True)
    repo_df.to_csv(out_repo_csv, index=False)

    overall = {
        "n_repos": int(len(repo_df)),
        "accuracy_mean": float(repo_df["accuracy"].mean(skipna=True)) if "accuracy" in repo_df.columns else float("nan"),
        "accuracy_by_author_mean": float(repo_df["accuracy_by_author"].mean(skipna=True)) if "accuracy_by_author" in repo_df.columns else float("nan"),
        "precision_macro_mean": float(repo_df["precision_macro"].mean(skipna=True)) if "precision_macro" in repo_df.columns else float("nan"),
        "recall_macro_mean": float(repo_df["recall_macro"].mean(skipna=True)) if "recall_macro" in repo_df.columns else float("nan"),
        "f1_macro_mean": float(repo_df["f1_macro"].mean(skipna=True)) if "f1_macro" in repo_df.columns else float("nan"),
        "auc_mean": float(repo_df["auc"].mean(skipna=True)) if "auc" in repo_df.columns else float("nan"),
        "confidence_mean_mean": float(repo_df["confidence_mean"].mean(skipna=True)) if "confidence_mean" in repo_df.columns else float("nan"),
    }
    overall_df = pd.DataFrame([overall])
    os.makedirs(os.path.dirname(out_overall_csv) or ".", exist_ok=True)
    overall_df.to_csv(out_overall_csv, index=False)

    return repo_df, overall_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate RF weak Com-Info evaluation across per-language CSVs.")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Directory containing evaluation_results_rf_{lang}_weak_info.csv (default: this script's directory).",
    )
    parser.add_argument("--inputs", nargs="*", default=None, help="Explicit input CSV paths (overrides discovery).")
    parser.add_argument("--out_repo_csv", type=str, default=None)
    parser.add_argument("--out_overall_csv", type=str, default=None)
    parser.add_argument("--match_re", type=str, default=_DEFAULT_PATTERN)
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = os.path.abspath(args.baseline_dir or this_dir)

    if args.inputs:
        inputs = [os.path.abspath(p) for p in args.inputs]
    else:
        inputs = _discover_inputs(baseline_dir, re.compile(args.match_re))

    if not inputs:
        raise SystemExit(f"No input RF-info CSVs found in {baseline_dir!r}")

    out_repo_csv = os.path.abspath(args.out_repo_csv or os.path.join(baseline_dir, "aggregate_rf_info_repo_weak.csv"))
    out_overall_csv = os.path.abspath(args.out_overall_csv or os.path.join(baseline_dir, "aggregate_rf_info_overall_weak.csv"))

    repo_df, _ = aggregate(inputs=inputs, out_repo_csv=out_repo_csv, out_overall_csv=out_overall_csv)
    print(f"Inputs: {len(inputs)} CSVs")
    print(f"Saved: {out_repo_csv} ({len(repo_df)} rows)")
    print(f"Saved: {out_overall_csv} (1 row)")


if __name__ == "__main__":
    main()
