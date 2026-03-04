#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def _parse_langs(raw: str) -> list[str]:
    raw_langs = [p.strip().lower() for p in (raw or "").split(",") if p.strip()]
    if not raw_langs:
        raise SystemExit("Invalid --langs")
    if "all" in raw_langs:
        return ["go", "java", "js", "php", "python"]
    return raw_langs


def _iter_jsons(language_dir: str, langs: list[str], json_glob: str) -> list[str]:
    paths: list[str] = []
    for lang in langs:
        combined = os.path.join(language_dir, f"combined_{lang}")
        if not os.path.isdir(combined):
            continue
        paths.extend(sorted(glob.glob(os.path.join(combined, json_glob))))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run adversial.attack_dataset over all JSONs under language/combined_{lang}."
    )
    ap.add_argument("--language_dir", default="language", help="Dataset root containing combined_{lang}/")
    ap.add_argument("--langs", default="all", help="go,java,js,php,python or 'all' or comma-separated list")
    ap.add_argument("--json_glob", default="*.json", help="Glob for JSONs within each combined_{lang}/")
    ap.add_argument("--out_root", default=os.path.join("adversial", "attacked_language"))
    ap.add_argument("--task", choices=["authorship", "binary"], default="authorship")
    ap.add_argument("--attack", choices=["targeted"], default="targeted")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--fixed_target", type=str, default=None)
    ap.add_argument(
        "--use_ast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try Python AST/CFG/UDC transforms before text-level transforms (Python only).",
    )
    ap.add_argument(
        "--token_rename_max",
        type=int,
        default=0,
        help="If >0, perform token-aware identifier renaming for go/java/js/php (stronger attack).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    ap.add_argument("--manifest", default=None, help="Write JSONL manifest to this path (default: <out_root>/manifest.jsonl)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    language_dir = os.path.abspath(os.path.expanduser(args.language_dir))
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    os.makedirs(out_root, exist_ok=True)

    langs = _parse_langs(args.langs)
    json_paths = _iter_jsons(language_dir, langs, args.json_glob)
    if not json_paths:
        raise SystemExit(f"No JSONs matched under {language_dir} for langs={langs} glob={args.json_glob!r}")

    manifest_path = args.manifest or os.path.join(out_root, "manifest.jsonl")
    man_f = None
    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        man_f = open(manifest_path, "a", encoding="utf-8")

    try:
        t0 = time.time()
        done = 0
        skipped = 0
        failed = 0

        for i, json_in in enumerate(json_paths, start=1):
            rel = os.path.relpath(json_in, language_dir)
            json_out = os.path.join(out_root, rel)
            os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)

            if os.path.exists(json_out) and not args.overwrite:
                skipped += 1
                rec = {"input": json_in, "output": json_out, "status": "skipped_exists"}
                if man_f:
                    man_f.write(json.dumps(rec) + "\n")
                    man_f.flush()
                print(f"[{i}/{len(json_paths)}] skip (exists): {rel}")
                continue

            cmd = [
                sys.executable,
                "-m",
                "adversial.attack_dataset",
                "--json_in",
                json_in,
                "--json_out",
                json_out,
                "--attack",
                args.attack,
                "--seed",
                str(args.seed),
                "--task",
                args.task,
                "--fraction",
                str(args.fraction),
            ]
            cmd.append("--use_ast" if args.use_ast else "--no-use_ast")
            if int(args.token_rename_max) > 0:
                cmd.extend(["--token_rename_max", str(int(args.token_rename_max))])
            if args.fixed_target:
                cmd.extend(["--fixed_target", args.fixed_target])

            print(f"[{i}/{len(json_paths)}] attack: {rel}")
            if args.dry_run:
                done += 1
                rec = {"input": json_in, "output": json_out, "status": "dry_run", "cmd": cmd}
                if man_f:
                    man_f.write(json.dumps(rec) + "\n")
                    man_f.flush()
                continue

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                done += 1
                rec = {"input": json_in, "output": json_out, "status": "ok"}
            except subprocess.CalledProcessError as e:
                failed += 1
                rec = {"input": json_in, "output": json_out, "status": "error", "returncode": e.returncode}

            if man_f:
                man_f.write(json.dumps(rec) + "\n")
                man_f.flush()

        dt = time.time() - t0
        print(f"Finished: ok={done} skipped={skipped} failed={failed} in {dt:.1f}s")
        print(f"Outputs under: {out_root}")
        if man_f:
            print(f"Manifest: {manifest_path}")
    finally:
        if man_f:
            man_f.close()


if __name__ == "__main__":
    main()
