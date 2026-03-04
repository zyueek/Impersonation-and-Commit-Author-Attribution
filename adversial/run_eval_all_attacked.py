#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from dataclasses import asdict

import pandas as pd

from adversial.common import seed_everything, write_json


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


def _join_code_lines(item: dict) -> str:
    added = item.get("added_code", "")
    if isinstance(added, list):
        return "\n".join(map(str, added))
    return str(added or "")


def _detect_author_field(sample_item: dict) -> str | None:
    if "emailname" in sample_item:
        return "emailname"
    if "author" in sample_item:
        return "author"
    return None


def _detect_label_field(sample_item: dict, *, task: str) -> str:
    if task == "binary":
        if "label" not in sample_item:
            raise ValueError("task=binary requires 'label' field in JSON items")
        return "label"
    if "emailname" in sample_item:
        return "emailname"
    if "author" in sample_item:
        return "author"
    raise ValueError("Could not detect label field (expected 'emailname' or 'author')")


def _load_pair_df(
    *,
    json_clean: str,
    json_attacked: str,
    task: str,
    min_samples_per_author: int,
) -> pd.DataFrame:
    with open(json_clean, "r") as f:
        clean = json.load(f)
    with open(json_attacked, "r") as f:
        attacked = json.load(f)
    if not isinstance(clean, list) or not isinstance(attacked, list):
        raise ValueError("Expected list[dict] JSONs")
    if not clean:
        raise ValueError("Empty clean JSON")

    label_field = _detect_label_field(clean[0], task=task)
    author_field = _detect_author_field(clean[0])
    if author_field is None:
        raise ValueError("Need author id field ('emailname' or 'author') for imitation evaluation")

    n = min(len(clean), len(attacked))
    rows = []
    for i in range(n):
        c = clean[i]
        a = attacked[i]
        rows.append(
            {
                "idx": i,
                "code": _join_code_lines(c),
                "code_attacked": _join_code_lines(a),
                "message": str(c.get("message", "") or ""),
                "filename": str(c.get("filename", "") or ""),
                "label": c.get(label_field),
                "author_id": c.get(author_field),
                "attack_target": a.get("attack_target"),
                "raw": c,
            }
        )
    df = pd.DataFrame(rows)

    # Clean-based filtering (to keep alignment stable).
    df = df[df["code"].astype(bool)].copy()
    if task == "authorship":
        df = df[df["label"].astype(str).str.len() > 0]
        counts = df["label"].value_counts()
        keep = counts[counts >= min_samples_per_author].index
        df = df[df["label"].isin(keep)].copy()
    else:
        df = df[df["label"].notna()].copy()

    return df.reset_index(drop=True)


def _targeted_success_rate(pred_ids: list[int], inv: dict[int, str], targets: list[str | None]) -> float | None:
    pred_labels = [inv.get(i) for i in pred_ids]
    ok = [p == t for p, t in zip(pred_labels, targets) if t is not None and p is not None]
    return float(sum(ok) / len(ok)) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate RF vs GraphCodeBERT on clean vs pre-attacked datasets (attacked JSONs produced by adversial.run_attack_all)."
    )
    ap.add_argument("--language_dir", default="language")
    ap.add_argument("--attacked_root", required=True, help="Root produced by run_attack_all (mirrors language/)")
    ap.add_argument("--langs", default="all")
    ap.add_argument("--json_glob", default="*.json")
    ap.add_argument("--task", choices=["authorship", "binary"], default="authorship")
    ap.add_argument("--min_samples_per_author", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42, help="Default seed (used if --seeds not set)")
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seeds to average over (e.g. '0,1,2,3,4'); overrides --seed.",
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument(
        "--models",
        default="rf,gcb",
        help="Comma-separated: rf,gcb,lang,t5,gcb_info,lang_info,t5_info (all non-rf are embedding-based except gcb finetune).",
    )
    ap.add_argument(
        "--rf_mode",
        choices=["weak", "weak_info", "strong"],
        default="weak",
        help="weak: code-only weak RF; weak_info: weak RF with commit info features; strong: alias of weak_info for back-compat.",
    )
    ap.add_argument("--out_root", default=os.path.join("adversial", "eval_attacked_all"))
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
    ap.add_argument("--gcb_train_aug_n", type=int, default=1)
    ap.add_argument("--gcb_train_aug_p", type=float, default=0.5)
    ap.add_argument("--gcb_train_aug_use_ast", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--token_rename_max",
        type=int,
        default=0,
        help="If >0, use token-aware identifier renaming in GCB training augmentation (go/java/js/php).",
    )
    ap.add_argument("--gcb_epochs", type=float, default=1.0)
    ap.add_argument("--gcb_batch_size", type=int, default=8)
    ap.add_argument("--gcb_lr", type=float, default=5e-5)
    ap.add_argument("--gcb_max_length", type=int, default=256)
    args = ap.parse_args()

    try:
        from sklearn.model_selection import train_test_split  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Failed to import scikit-learn. This repo expects the conda env at "
            "`/home/yueke/miniconda3/envs/reason/bin/python` (Python 3.10).\n"
            "Try: `conda run -n reason python -m adversial.run_eval_all_attacked ...`\n"
            f"Underlying error: {e}"
        )

    seeds: list[int]
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise SystemExit("Invalid --seeds")
    else:
        seeds = [int(args.seed)]

    seed_everything(seeds[0])
    language_dir = os.path.abspath(os.path.expanduser(args.language_dir))
    attacked_root = os.path.abspath(os.path.expanduser(args.attacked_root))
    out_root = os.path.abspath(os.path.expanduser(args.out_root))
    os.makedirs(out_root, exist_ok=True)

    langs = _parse_langs(args.langs)
    json_paths = _iter_jsons(language_dir, langs, args.json_glob)
    if not json_paths:
        raise SystemExit("No JSONs found")

    from adversial.eval_imitation import (
        _augment_train_for_gcb,
        _compute_metrics,
        _embed_model_train_predict_pair,
        _embed_multimodal_train_predict_pair,
        _gcb_finetune_predict,
        _gcb_finetune_train,
        _embed_hf_cls,
        _embed_t5_mean,
        _rf_baseline_train_predict_mode,
    )

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    summary_rows: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()

    metric_fields = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    for i, clean_path in enumerate(json_paths, start=1):
        rel = os.path.relpath(clean_path, language_dir)
        attacked_path = os.path.join(attacked_root, rel)
        if not os.path.exists(attacked_path):
            errors.append({"dataset": rel, "status": "missing_attacked", "attacked_path": attacked_path})
            continue

        try:
            df = _load_pair_df(
                json_clean=clean_path,
                json_attacked=attacked_path,
                task=args.task,
                min_samples_per_author=args.min_samples_per_author,
            )
            if df.empty or (args.task == "authorship" and df["label"].nunique() < 2):
                errors.append({"dataset": rel, "status": "skip_not_enough_data", "n": int(len(df))})
                continue

            result = {
                "dataset": rel,
                "clean_path": os.path.abspath(clean_path),
                "attacked_path": os.path.abspath(attacked_path),
                "task": args.task,
                "seeds": seeds,
                "models": {},
            }

            # Collect per-seed metrics and then average across seeds (reduces quantization/tie effects).
            model_seed_metrics: dict[str, list[dict]] = {m: [] for m in models}
            model_seed_tsr: dict[str, list[float]] = {m: [] for m in models}

            for seed in seeds:
                strat = df["label"] if args.task in {"authorship", "binary"} else None
                train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=seed, stratify=strat)

                attacked_test_df = test_df.copy()
                attacked_test_df["code"] = attacked_test_df["code_attacked"]
                targets = attacked_test_df["attack_target"].tolist()

                # Keep the first seed's split sizes for reporting.
                if "n_train" not in result:
                    result["n_train"] = int(len(train_df))
                    result["n_test"] = int(len(test_df))

                for model in models:
                    if model == "rf":
                        y_true, pred_clean, inv = _rf_baseline_train_predict_mode(train_df, test_df, seed=seed, mode=args.rf_mode)
                        y_true2, pred_attack, _inv2 = _rf_baseline_train_predict_mode(train_df, attacked_test_df, seed=seed, mode=args.rf_mode)
                        assert y_true == y_true2
                        m_clean = _compute_metrics(y_true, pred_clean)
                        m_attack = _compute_metrics(y_true, pred_attack)
                        drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}
                        tsr = _targeted_success_rate(pred_attack, inv, targets) if args.task == "authorship" else None
                        model_seed_metrics[model].append({"clean": asdict(m_clean), "attacked": asdict(m_attack), "drop": drop})
                        if tsr is not None:
                            model_seed_tsr[model].append(float(tsr))

                    elif model == "gcb":
                        train_df_gcb = _augment_train_for_gcb(
                            train_df,
                            seed=seed,
                            mode=args.gcb_train_aug,
                            n_copies=args.gcb_train_aug_n,
                            p=args.gcb_train_aug_p,
                            use_ast=bool(args.gcb_train_aug_use_ast),
                            token_rename_max=int(args.token_rename_max),
                        )
                        if args.gcb_mode == "finetune":
                            tok, mdl, label_map, inv = _gcb_finetune_train(
                                train_df_gcb,
                                seed=seed,
                                model_name=args.gcb_model,
                                local_files_only=args.gcb_local_files_only,
                                epochs=args.gcb_epochs,
                                batch_size=args.gcb_batch_size,
                                learning_rate=args.gcb_lr,
                                max_length=args.gcb_max_length,
                                out_dir=out_root,
                                device=args.gcb_device,
                                fp16=bool(args.gcb_fp16),
                            )
                            y_true, pred_clean = _gcb_finetune_predict(
                                tok=tok,
                                model=mdl,
                                label_map=label_map,
                                df=test_df,
                                max_length=args.gcb_max_length,
                                batch_size=args.gcb_batch_size,
                                device=args.gcb_device,
                            )
                            y_true2, pred_attack = _gcb_finetune_predict(
                                tok=tok,
                                model=mdl,
                                label_map=label_map,
                                df=attacked_test_df,
                                max_length=args.gcb_max_length,
                                batch_size=args.gcb_batch_size,
                                device=args.gcb_device,
                            )
                            assert y_true == y_true2
                        else:
                            y_true, pred_clean, pred_attack, inv = _embed_model_train_predict_pair(
                                train_df_gcb,
                                test_df,
                                attacked_test_df,
                                seed=seed,
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
                        m_clean = _compute_metrics(y_true, pred_clean)
                        m_attack = _compute_metrics(y_true, pred_attack)
                        drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}
                        tsr = _targeted_success_rate(pred_attack, inv, targets) if args.task == "authorship" else None
                        model_seed_metrics[model].append({"clean": asdict(m_clean), "attacked": asdict(m_attack), "drop": drop})
                        if tsr is not None:
                            model_seed_tsr[model].append(float(tsr))

                    elif model == "lang":
                        y_true, pred_clean, pred_attack, inv = _embed_model_train_predict_pair(
                            train_df,
                            test_df,
                            attacked_test_df,
                            seed=seed,
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
                        m_clean = _compute_metrics(y_true, pred_clean)
                        m_attack = _compute_metrics(y_true, pred_attack)
                        drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}
                        tsr = _targeted_success_rate(pred_attack, inv, targets) if args.task == "authorship" else None
                        model_seed_metrics[model].append({"clean": asdict(m_clean), "attacked": asdict(m_attack), "drop": drop})
                        if tsr is not None:
                            model_seed_tsr[model].append(float(tsr))

                    elif model in {"gcb_info", "lang_info", "t5_info"}:
                        if model == "gcb_info":
                            code_embedder = _embed_hf_cls
                            code_kwargs = {"model_name": args.gcb_model}
                        elif model == "lang_info":
                            code_embedder = _embed_hf_cls
                            code_kwargs = {"model_name": args.codebert_model}
                        else:
                            code_embedder = _embed_t5_mean
                            code_kwargs = {"model_name": args.t5_model}

                        y_true, pred_clean, pred_attack, inv = _embed_multimodal_train_predict_pair(
                            train_df,
                            test_df,
                            attacked_test_df,
                            seed=seed,
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
                        tsr = _targeted_success_rate(pred_attack, inv, targets) if args.task == "authorship" else None
                        model_seed_metrics[model].append({"clean": asdict(m_clean), "attacked": asdict(m_attack), "drop": drop})
                        if tsr is not None:
                            model_seed_tsr[model].append(float(tsr))

                    elif model == "t5":
                        y_true, pred_clean, pred_attack, inv = _embed_model_train_predict_pair(
                            train_df,
                            test_df,
                            attacked_test_df,
                            seed=seed,
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
                        m_clean = _compute_metrics(y_true, pred_clean)
                        m_attack = _compute_metrics(y_true, pred_attack)
                        drop = {k: asdict(m_clean)[k] - asdict(m_attack)[k] for k in asdict(m_clean).keys()}
                        tsr = _targeted_success_rate(pred_attack, inv, targets) if args.task == "authorship" else None
                        model_seed_metrics[model].append({"clean": asdict(m_clean), "attacked": asdict(m_attack), "drop": drop})
                        if tsr is not None:
                            model_seed_tsr[model].append(float(tsr))
                    else:
                        raise ValueError(f"Unknown model: {model}")

            # Aggregate across seeds.
            for model in models:
                items = model_seed_metrics.get(model) or []
                if not items:
                    continue
                # mean across seeds
                def mean_metric(which: str, key: str) -> float:
                    vals = [float(it[which][key]) for it in items]
                    return float(sum(vals) / len(vals))

                def std_metric(which: str, key: str) -> float:
                    vals = [float(it[which][key]) for it in items]
                    if len(vals) <= 1:
                        return 0.0
                    mu = sum(vals) / len(vals)
                    return float((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)

                clean_mean = {k: mean_metric("clean", k) for k in metric_fields}
                attacked_mean = {k: mean_metric("attacked", k) for k in metric_fields}
                drop_mean = {k: mean_metric("drop", k) for k in metric_fields}
                clean_std = {k: std_metric("clean", k) for k in metric_fields}
                attacked_std = {k: std_metric("attacked", k) for k in metric_fields}
                drop_std = {k: std_metric("drop", k) for k in metric_fields}

                result_model_key = model
                if model == "rf":
                    result["models"][result_model_key] = {
                        "clean": clean_mean,
                        "attacked": attacked_mean,
                        "metric_drop": drop_mean,
                        "clean_std": clean_std,
                        "attacked_std": attacked_std,
                        "metric_drop_std": drop_std,
                        "targeted_success_rate": (sum(model_seed_tsr[model]) / len(model_seed_tsr[model])) if model_seed_tsr[model] else None,
                        "rf_mode": args.rf_mode,
                        "n_seeds": len(seeds),
                    }
                elif model == "gcb":
                    result["models"][result_model_key] = {
                        "clean": clean_mean,
                        "attacked": attacked_mean,
                        "metric_drop": drop_mean,
                        "clean_std": clean_std,
                        "attacked_std": attacked_std,
                        "metric_drop_std": drop_std,
                        "targeted_success_rate": (sum(model_seed_tsr[model]) / len(model_seed_tsr[model])) if model_seed_tsr[model] else None,
                        "gcb_mode": args.gcb_mode,
                        "gcb_train_aug": args.gcb_train_aug,
                        "gcb_train_aug_n": int(args.gcb_train_aug_n),
                        "gcb_train_aug_p": float(args.gcb_train_aug_p),
                        "n_seeds": len(seeds),
                    }
                else:
                    # All other models in this script are embedding-based classifiers.
                    result["models"][result_model_key] = {
                        "clean": clean_mean,
                        "attacked": attacked_mean,
                        "metric_drop": drop_mean,
                        "clean_std": clean_std,
                        "attacked_std": attacked_std,
                        "metric_drop_std": drop_std,
                        "targeted_success_rate": (sum(model_seed_tsr[model]) / len(model_seed_tsr[model])) if model_seed_tsr[model] else None,
                        "embed_solver": args.gcb_embed_solver,
                        "embed_max_iter": int(args.gcb_embed_max_iter),
                        "embed_scale": bool(args.gcb_embed_scale),
                        "n_seeds": len(seeds),
                    }

                # Summary row uses per-model mean metrics.
                if model == "rf":
                    model_name = f"rf_{args.rf_mode}"
                elif model == "gcb":
                    model_name = f"gcb_{args.gcb_mode}" if args.gcb_train_aug == "none" else f"gcb_{args.gcb_mode}_aug_{args.gcb_train_aug}"
                else:
                    model_name = model
                row = {"dataset": rel, "model": model_name}
                for mf in metric_fields:
                    row[f"clean_{mf}"] = clean_mean[mf]
                    row[f"attacked_{mf}"] = attacked_mean[mf]
                    row[f"drop_{mf}"] = drop_mean[mf]
                row["targeted_success_rate"] = (sum(model_seed_tsr[model]) / len(model_seed_tsr[model])) if model_seed_tsr[model] else None
                row["n_seeds"] = len(seeds)
                summary_rows.append(row)

            out_json = os.path.join(out_root, rel.replace(os.sep, "__") + ".json")
            write_json(out_json, result)
            print(f"[{i}/{len(json_paths)}] ok: {rel}")

        except Exception as e:
            errors.append({"dataset": rel, "status": "error", "error": str(e)})
            print(f"[{i}/{len(json_paths)}] error: {rel}: {e}")

    dt = time.time() - t0
    summary_path = os.path.join(out_root, "summary.csv")
    # Write summary without pandas (avoids pandas/numpy ABI issues across envs).
    if summary_rows:
        cols = list(summary_rows[0].keys())
        # Ensure stable ordering: dataset, model first.
        if "dataset" in cols and "model" in cols:
            cols = ["dataset", "model"] + [c for c in cols if c not in {"dataset", "model"}]
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in summary_rows:
                w.writerow(row)
    else:
        with open(summary_path, "w", newline="") as f:
            f.write("")

    errors_path = os.path.join(out_root, "errors.json")
    write_json(errors_path, {"errors": errors})

    print(f"Done in {dt:.1f}s")
    print(f"Summary: {summary_path}")
    print(f"Errors:  {errors_path}")


if __name__ == "__main__":
    main()
