#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Approach:
    name: str
    train: list[str]
    collect: list[str]
    aggregate: list[str]


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _lang_arg(lang: str) -> str:
    lang = lang.strip().lower()
    if not lang:
        raise SystemExit("Invalid --lang")
    # Allow comma-separated lists or 'all' passthrough.
    return lang


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline pipelines (train -> collect -> aggregate).")
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="go, java, js, php, python; or 'all'; or comma-separated list (e.g. 'python,js')",
    )
    parser.add_argument("--json_dir", type=str, default=None, help="Optional dataset root (e.g. /home/yueke/author/language)")
    parser.add_argument("--json_glob", type=str, default="*.json", help="Glob within combined_{lang} (default: *.json)")
    parser.add_argument(
        "--approaches",
        type=str,
        default="rf,rf_info,scap,scap_info",
        help="Comma-separated: rf, rf_info, scap, scap_info (default: all four)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    lang = _lang_arg(args.lang)

    json_dir_args: list[str] = []
    if args.json_dir:
        json_dir_args = ["--json_dir", os.path.abspath(os.path.expanduser(args.json_dir))]

    approaches: dict[str, Approach] = {
        "rf": Approach(
            name="rf",
            train=[py, os.path.join(baseline_dir, "train_rf_baseline.py"), "--lang", lang, *json_dir_args, "--json_glob", args.json_glob],
            collect=[py, os.path.join(baseline_dir, "collect_baseline_eval.py"), "--lang", lang],
            aggregate=[py, os.path.join(baseline_dir, "aggregate_rf_results.py"), "--baseline_dir", baseline_dir],
        ),
        "rf_info": Approach(
            name="rf_info",
            train=[
                py,
                os.path.join(baseline_dir, "train_rf_baseline_info.py"),
                "--lang",
                lang,
                *json_dir_args,
                "--json_glob",
                args.json_glob,
            ],
            collect=[py, os.path.join(baseline_dir, "collect_rf_info_eval.py"), "--lang", lang],
            aggregate=[py, os.path.join(baseline_dir, "aggregate_rf_info_results.py"), "--baseline_dir", baseline_dir],
        ),
        "scap": Approach(
            name="scap",
            train=[py, os.path.join(baseline_dir, "train_scap_baseline.py"), "--lang", lang, *json_dir_args, "--json_glob", args.json_glob],
            collect=[py, os.path.join(baseline_dir, "collect_scap_eval.py"), "--lang", lang],
            aggregate=[py, os.path.join(baseline_dir, "aggregate_scap_results.py"), "--baseline_dir", baseline_dir],
        ),
        "scap_info": Approach(
            name="scap_info",
            train=[
                py,
                os.path.join(baseline_dir, "train_scap_baseline_info.py"),
                "--lang",
                lang,
                *json_dir_args,
                "--json_glob",
                args.json_glob,
            ],
            collect=[py, os.path.join(baseline_dir, "collect_scap_info_eval.py"), "--lang", lang],
            aggregate=[py, os.path.join(baseline_dir, "aggregate_scap_info_results.py"), "--baseline_dir", baseline_dir],
        ),
        "dlcais": Approach(
            name="dlcais",
            train=[py, os.path.join(baseline_dir, "train_dlcais_baseline.py"), "--lang", lang, *json_dir_args, "--json_glob", args.json_glob],
            collect=[py, os.path.join(baseline_dir, "collect_dlcais_eval.py"), "--lang", lang],
            aggregate=[py, os.path.join(baseline_dir, "aggregate_dlcais_results.py"), "--baseline_dir", baseline_dir],
        ),
    }

    selected = [a.strip().lower() for a in args.approaches.split(",") if a.strip()]
    unknown = [a for a in selected if a not in approaches]
    if unknown:
        raise SystemExit(f"Unknown approaches: {unknown}. Use: rf,rf_info,scap,scap_info,dlcais")

    for name in selected:
        a = approaches[name]
        print(f"\n== {a.name}: Step 1/3 train ==")
        _run(a.train, dry_run=args.dry_run)
        print(f"== {a.name}: Step 2/3 collect ==")
        _run(a.collect, dry_run=args.dry_run)
        print(f"== {a.name}: Step 3/3 aggregate ==")
        _run(a.aggregate, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
