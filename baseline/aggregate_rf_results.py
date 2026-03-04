#!/usr/bin/env python3
import argparse
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


_DEFAULT_PATTERN = r"^evaluation_results_rf_(go|java|js|php|python)(?:_weak)?\.csv$"


def _discover_inputs(baseline_dir: str, filename_re: re.Pattern) -> list[str]:
    inputs: list[str] = []
    for name in os.listdir(baseline_dir):
        if filename_re.match(name):
            inputs.append(os.path.join(baseline_dir, name))
    return sorted(inputs)

def _select_inputs_for_preset(baseline_dir: str, preset: str) -> list[str]:
    """
    Prefer preset-suffixed files (e.g., evaluation_results_rf_python_weak.csv),
    falling back to unsuffixed files (evaluation_results_rf_python.csv) if needed.
    """
    preset = "weak"
    langs = ["go", "java", "js", "php", "python"]
    selected: list[str] = []
    for lang in langs:
        preferred = os.path.join(baseline_dir, f"evaluation_results_rf_{lang}_{preset}.csv")
        fallback = os.path.join(baseline_dir, f"evaluation_results_rf_{lang}.csv")
        if os.path.isfile(preferred):
            selected.append(preferred)
        elif os.path.isfile(fallback):
            selected.append(fallback)
    return selected


def _infer_lang(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"evaluation_results_rf_([a-z]+)(?:_weak)?\.csv$", base)
    return m.group(1) if m else "unknown"


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _safe_mean(x: pd.Series) -> float:
    x = _to_float(x)
    if x.notna().any():
        return float(x.mean(skipna=True))
    return float("nan")


def _safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = _to_float(values)
    weights = _to_float(weights)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


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
    # Fallback: treat unknown strings as False to avoid bool("False") == True surprises.
    return False


def aggregate(
    inputs: list[str],
    out_repo_csv: str,
    out_overall_csv: str,
    rf_preset: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in inputs:
        df = pd.read_csv(path)
        df["lang"] = _infer_lang(path)
        df["source_csv"] = os.path.basename(path)
        frames.append(df)

    if not frames:
        raise SystemExit("No input CSVs found.")

    all_df = pd.concat(frames, ignore_index=True)

    # Expected columns from baseline/collect_baseline_eval.py
    metric_cols = [
        "accuracy",
        "accuracy_by_author",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc",
        "confidence_mean",
    ]
    for col in metric_cols + ["n_test", "n_authors_test", "n_authors_total"]:
        if col in all_df.columns:
            all_df[col] = _to_float(all_df[col])

    if "rf_preset" in all_df.columns:
        all_df["rf_preset"] = all_df["rf_preset"].astype(str).str.strip().str.lower()
    if "include_message" in all_df.columns:
        all_df["include_message"] = all_df["include_message"].map(_to_bool)
    if "include_filename" in all_df.columns:
        all_df["include_filename"] = all_df["include_filename"].map(_to_bool)

    group_cols = ["repo", "rf_preset", "include_message", "include_filename"]
    for col in group_cols:
        if col not in all_df.columns:
            raise SystemExit(f"Missing required column {col!r} in inputs; regenerate CSVs with baseline/collect_baseline_eval.py")

    if rf_preset is not None:
        rf_preset = rf_preset.strip().lower()
        all_df = all_df[all_df["rf_preset"] == rf_preset].copy()
        if all_df.empty:
            raise SystemExit(f"No rows matched rf_preset={rf_preset!r} in the provided inputs.")

    grouped = all_df.groupby(group_cols, dropna=False)
    repo_rows = []
    for key, g in grouped:
        repo, rf_preset, include_message, include_filename = key
        n_test_sum = float(g["n_test"].sum()) if "n_test" in g.columns else float("nan")
        langs = _concat_unique(g["lang"].tolist())
        n_langs = int(g["lang"].nunique())
        repo_rows.append(
            {
                "repo": repo,
                "rf_preset": rf_preset,
                "include_message": bool(include_message),
                "include_filename": bool(include_filename),
                "langs": langs,
                "n_langs": n_langs,
                "n_rows": int(len(g)),
                "n_test_sum": n_test_sum,
                "n_authors_total_mean": _safe_mean(g.get("n_authors_total", pd.Series(dtype=float))),
                # Plain mean across the (up to 5) files this repo appears in.
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

    # Overall summaries: plain mean across repositories.
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
    parser = argparse.ArgumentParser(
        description="Aggregate baseline RF evaluation across the 5 per-language CSVs into per-repo and overall summaries."
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Directory containing evaluation_results_rf_{lang}.csv (default: this script's directory).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit CSV paths (overrides auto-discovery).",
    )
    parser.add_argument(
        "--out_repo_csv",
        type=str,
        default=None,
        help="Per-repo output CSV (default: baseline/aggregate_rf_repo.csv).",
    )
    parser.add_argument(
        "--out_overall_csv",
        type=str,
        default=None,
        help="Overall output CSV (default: baseline/aggregate_rf_overall.csv).",
    )
    parser.add_argument(
        "--match_re",
        type=str,
        default=_DEFAULT_PATTERN,
        help=f"Regex for auto-discovery within baseline_dir (default: {_DEFAULT_PATTERN})",
    )
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_dir = os.path.abspath(args.baseline_dir or this_dir)

    if args.inputs:
        inputs = [os.path.abspath(p) for p in args.inputs]
    else:
        # If user didn't override match_re, pick exactly one CSV per language for the requested preset.
        if args.match_re == _DEFAULT_PATTERN:
            inputs = _select_inputs_for_preset(baseline_dir, "weak")
        else:
            inputs = _discover_inputs(baseline_dir, re.compile(args.match_re))

    if not inputs:
        raise SystemExit(f"No input RF CSVs found in {baseline_dir!r}")

    out_repo_csv = os.path.abspath(args.out_repo_csv or os.path.join(baseline_dir, "aggregate_rf_repo_weak.csv"))
    out_overall_csv = os.path.abspath(args.out_overall_csv or os.path.join(baseline_dir, "aggregate_rf_overall_weak.csv"))

    repo_df, overall_df = aggregate(
        inputs=inputs,
        out_repo_csv=out_repo_csv,
        out_overall_csv=out_overall_csv,
        rf_preset="weak",
    )
    print(f"Inputs: {len(inputs)} CSVs")
    print(f"Saved: {out_repo_csv} ({len(repo_df)} rows)")
    print(f"Saved: {out_overall_csv} (1 row)")


if __name__ == "__main__":
    main()
