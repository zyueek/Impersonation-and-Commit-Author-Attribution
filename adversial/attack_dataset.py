#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

from adversial.common import read_json, seed_everything, write_json
from adversial.load_data import detect_author_field, detect_label_field
from adversial.style_profile import build_profile
from adversial.imitate_style import targeted_attack


def _join_code_lines(item: dict) -> str:
    added = item.get("added_code", "")
    if isinstance(added, list):
        return "\n".join(map(str, added))
    return str(added or "")


def _split_code_to_lines(text: str) -> list[str]:
    return text.splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an attacked (style-imitated) dataset JSON.")
    ap.add_argument("--json_in", required=True)
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--attack", choices=["targeted"], default="targeted")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", choices=["authorship", "binary"], default="authorship")
    ap.add_argument("--fraction", type=float, default=1.0, help="Fraction of items to attack (default: 1.0)")
    ap.add_argument("--fixed_target", type=str, default=None, help="Optional fixed target author id")
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
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)

    data = read_json(args.json_in)
    if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
        raise SystemExit("Expected JSON list[dict]")

    label_field = detect_label_field(data[0], task=args.task)
    author_field = detect_author_field(data[0])

    if args.task == "authorship":
        authors = sorted({str(it.get(label_field) or "") for it in data if str(it.get(label_field) or "")})
    else:
        if author_field == "none":
            raise SystemExit("task=binary requires an author id field ('emailname' or 'author') for style imitation")
        authors = sorted({str(it.get(author_field) or "") for it in data if str(it.get(author_field) or "")})

    if args.task == "authorship" and len(authors) < 2:
        raise SystemExit("Need >= 2 authors for authorship attack")

    # Build per-author profiles from their code.
    per_author_texts = {a: [] for a in authors}
    for it in data:
        a = str(it.get(label_field) or "") if args.task == "authorship" else str(it.get(author_field) or "")
        if a in per_author_texts:
            per_author_texts[a].append(_join_code_lines(it))
    profiles = {a: build_profile(per_author_texts[a]) for a in authors}

    out = []
    for it in data:
        it2 = dict(it)
        if rng.random() > args.fraction:
            out.append(it2)
            continue

        true_a = str(it.get(label_field) or "") if args.task == "authorship" else str(it.get(author_field) or "")
        if args.task == "authorship":
            if args.fixed_target:
                target = args.fixed_target
                if target == true_a and len(authors) > 1:
                    target = rng.choice([a for a in authors if a != true_a])
            else:
                target = rng.choice([a for a in authors if a != true_a])
        else:
            target = args.fixed_target or rng.choice(authors)
            if target == true_a and len(authors) > 1:
                target = rng.choice([a for a in authors if a != true_a])

        res = targeted_attack(
            text=_join_code_lines(it),
            target_label=target,
            target_profile=profiles[target],
            language=str(it.get("language", "") or ""),
            use_ast=bool(args.use_ast),
            token_rename_max=int(args.token_rename_max),
            rng=rng,
        )
        it2["added_code"] = _split_code_to_lines(res.attacked_text)
        it2["attack_target"] = res.target_label
        if res.meta:
            it2["attack_meta"] = res.meta
        out.append(it2)

    write_json(args.json_out, out)
    print(f"Wrote: {args.json_out}")


if __name__ == "__main__":
    main()
