#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


_STRING_RE = re.compile(r"(\"([^\"\\\\]|\\\\.)*\"|'([^'\\\\]|\\\\.)*')")
_NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMMENT_LINE_RE = re.compile(r"^\s*(#|//|/\\*|\\*)")


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


def _extract_layout_features(text: str) -> dict:
    if not text:
        return {
            "num_chars": 0,
            "num_lines": 0,
            "avg_line_len": 0.0,
            "std_line_len": 0.0,
            "whitespace_ratio": 0.0,
            "tab_ratio": 0.0,
            "space_ratio": 0.0,
            "comment_line_ratio": 0.0,
            "string_literal_ratio": 0.0,
            "number_literal_ratio": 0.0,
            "identifier_ratio": 0.0,
            "newline_before_open_brace_ratio": 0.0,
            "tabs_lead_indented_lines_ratio": 0.0,
        }

    num_chars = len(text)
    lines = text.splitlines()
    line_lens = np.array([len(ln) for ln in lines], dtype=np.float32) if lines else np.array([], dtype=np.float32)
    num_lines = int(line_lens.size)

    whitespace = sum(ch.isspace() for ch in text)
    tabs = text.count("\t")
    spaces = text.count(" ")

    comment_lines = 0
    indented_lines = 0
    tab_indented = 0
    for ln in lines:
        if _COMMENT_LINE_RE.match(ln):
            comment_lines += 1
        if ln.startswith((" ", "\t")):
            indented_lines += 1
            if ln.startswith("\t"):
                tab_indented += 1

    open_braces = text.count("{")
    newline_open_braces = len(re.findall(r"\n\s*\{", text))

    string_literals = len(_STRING_RE.findall(text))
    number_literals = len(_NUMBER_RE.findall(text))
    identifiers = len(_IDENT_RE.findall(text))

    def safe_ratio(num: float, den: float) -> float:
        return float(num) / float(den) if den else 0.0

    return {
        "num_chars": float(num_chars),
        "num_lines": float(num_lines),
        "avg_line_len": float(line_lens.mean()) if num_lines else 0.0,
        "std_line_len": float(line_lens.std()) if num_lines else 0.0,
        "whitespace_ratio": safe_ratio(whitespace, num_chars),
        "tab_ratio": safe_ratio(tabs, num_chars),
        "space_ratio": safe_ratio(spaces, num_chars),
        "comment_line_ratio": safe_ratio(comment_lines, num_lines),
        "string_literal_ratio": safe_ratio(string_literals, num_chars),
        "number_literal_ratio": safe_ratio(number_literals, num_chars),
        "identifier_ratio": safe_ratio(identifiers, num_chars),
        "newline_before_open_brace_ratio": safe_ratio(newline_open_braces, open_braces),
        "tabs_lead_indented_lines_ratio": safe_ratio(tab_indented, indented_lines),
    }


def _layout_dicts(texts):
    return [_extract_layout_features(t) for t in texts]


def _load_and_preprocess(
    json_path: str,
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
        rows.append(
            {
                "text": text,
                "emailname": item.get("emailname", ""),
                "raw": item,
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["emailname"].astype(bool)]
    df = df[df["text"].astype(bool)]

    # Filter authors by sample count
    counts = df["emailname"].value_counts()
    keep = counts[counts >= min_samples_per_author].index
    df = df[df["emailname"].isin(keep)].copy()
    if df.empty:
        return df

    # Optional downsampling per author (for speed/consistency)
    if max_samples_per_author is not None:
        df = (
            df.groupby("emailname", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max_samples_per_author), random_state=42))
            .reset_index(drop=True)
        )

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@dataclass(frozen=True)
class RFParams:
    n_estimators: int
    max_features: str | int | float | None
    max_depth: int | None


def _rf_params(max_depth: int | None) -> RFParams:
    # Weak-only baseline (intentionally constrained).
    return RFParams(n_estimators=30, max_features=20, max_depth=max_depth)


def _make_pipeline(
    word_max_features: int,
    char_max_features: int,
    rf: RFParams,
    seed: int,
    *,
    use_char: bool,
    use_layout: bool,
) -> Pipeline:
    word_vec = TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|->|::|&&|\\|\\||[{}()\\[\\];,\\.<>+\\-*/%=&|^!~?:]",
        ngram_range=(1, 1),
        min_df=2,
        max_features=word_max_features,
        dtype=np.float32,
    )
    transformers = [("word_tfidf", word_vec)]

    if use_char:
        char_vec = TfidfVectorizer(
            analyzer="char",
            lowercase=False,
            ngram_range=(3, 6),
            min_df=2,
            max_features=char_max_features,
            dtype=np.float32,
        )
        transformers.append(("char_tfidf", char_vec))

    if use_layout:
        layout_pipe = Pipeline(
            steps=[
                ("to_dicts", FunctionTransformer(_layout_dicts, validate=False)),
                ("dictvec", DictVectorizer(sparse=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("layout", layout_pipe))

    features = FeatureUnion(transformers)

    clf = RandomForestClassifier(
        n_estimators=rf.n_estimators,
        max_features=rf.max_features,
        max_depth=rf.max_depth,
        n_jobs=-1,
        random_state=seed,
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("drop_constants", VarianceThreshold(0.0)),
            ("clf", clf),
        ]
    )


def _save_results(
    output_dir: str,
    accuracy: float,
    predicted: list[int],
    true_labels: list[int],
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    label_map: dict,
    confidence: list[float] | None,
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
        if confidence is not None:
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
    min_samples_per_author: int,
    max_samples_per_author: int | None,
    test_size: float,
    seed: int,
    rf_max_depth: int | None,
    include_message: bool,
    include_filename: bool,
    fragment_chars: int | None,
    fragment_mode: str,
    feature_set: str,
    word_max_features: int,
    char_max_features: int,
):
    print(f"Processing: {json_path}")
    df = _load_and_preprocess(
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
    y = df["emailname"].map(label_map).astype(int).to_numpy()

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["emailname"]
    )
    fs = feature_set.lower().strip()
    if fs == "auto":
        fs = "word_only"
    if fs == "full":
        use_char, use_layout = True, True
    elif fs == "word_only":
        use_char, use_layout = False, False
    else:
        raise ValueError(f"Unknown feature_set={feature_set!r} (use: auto|full|word_only)")

    pipeline = _make_pipeline(
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        rf=_rf_params(rf_max_depth),
        seed=seed,
        use_char=use_char,
        use_layout=use_layout,
    )

    pipeline.fit(train_df["text"].tolist(), train_df["emailname"].map(label_map).astype(int).tolist())
    pred = pipeline.predict(test_df["text"].tolist()).astype(int).tolist()
    true = test_df["emailname"].map(label_map).astype(int).tolist()
    acc = accuracy_score(true, pred)

    conf = None
    try:
        prob = pipeline.predict_proba(test_df["text"].tolist())
        conf = prob.max(axis=1).astype(float).tolist()
    except Exception:
        pass

    out_dir = os.path.join(out_root, os.path.basename(json_path).replace(".json", ""))
    _save_results(
        out_dir,
        accuracy=acc,
        predicted=pred,
        true_labels=true,
        test_df=test_df,
        train_df=train_df,
        label_map=label_map,
        confidence=conf,
        meta={
            "json_path": json_path,
            "n_samples": int(len(df)),
            "n_authors": int(df["emailname"].nunique()),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "rf_preset": "weak",
            "rf_max_depth": rf_max_depth,
            "include_message": include_message,
            "include_filename": include_filename,
            "fragment_chars": fragment_chars,
            "fragment_mode": fragment_mode,
            "feature_set": fs,
            "word_max_features": word_max_features,
            "char_max_features": char_max_features,
            "min_samples_per_author": min_samples_per_author,
            "max_samples_per_author": max_samples_per_author,
            "test_size": test_size,
            "seed": seed,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="Random-forest stylometry baseline for commit authorship attribution (code diffs + optional metadata)."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to run: go, java, js, php, python; or 'all'; or comma-separated list (e.g. 'python,js')",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=None,
        help="Override input directory (defaults to <repo_root>/language/combined_{lang}); can also point to <repo_root>/language",
    )
    parser.add_argument("--json_glob", type=str, default="*.json", help="Glob within json_dir (default: *.json)")
    parser.add_argument("--out_dir", type=str, default=None, help="Output root directory (default: <this_dir>/rf_{lang})")

    parser.add_argument("--min_samples_per_author", type=int, default=5)
    parser.add_argument("--max_samples_per_author", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # RF baseline is weak-only (no strong mode).
    parser.add_argument("--rf_max_depth", type=int, default=None)

    parser.add_argument("--include_message", action="store_true", help="Append commit message to the input text")
    parser.add_argument("--include_filename", action="store_true", help="Append edited filename to the input text")

    parser.add_argument(
        "--fragment_chars",
        type=int,
        default=None,
        help="If set, truncate each sample to this many characters (recommended to avoid unrealistically easy attribution from huge diffs).",
    )
    parser.add_argument(
        "--fragment_mode",
        type=str,
        default="random",
        choices=["head", "tail", "random"],
        help="How to take the fragment when --fragment_chars is set.",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="auto",
        choices=["auto", "full", "word_only"],
        help="Feature preset: auto=full for strong, word_only for weak; full=word+char+layout; word_only=word TF-IDF only.",
    )

    parser.add_argument("--word_max_features", type=int, default=None, help="Max word features (default: 2000).")
    parser.add_argument(
        "--char_max_features",
        type=int,
        default=None,
        help="Max char features (ignored for word_only); default: 2000.",
    )

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
        # If base is already a combined_{lang} directory (or contains JSONs), keep it.
        if glob.glob(os.path.join(base, args.json_glob)):
            return base
        # If base points to .../language, use .../language/combined_{lang} if it exists.
        candidate = os.path.join(base, f"combined_{lang}")
        if os.path.isdir(candidate):
            return candidate
        return base

    def resolve_out_dir(base: str | None, lang: str) -> str:
        suffix = "_weak"
        if base is None:
            return os.path.join(this_dir, f"rf_{lang}{suffix}")
        base = os.path.abspath(os.path.expanduser(base))
        # If running multiple languages, avoid collisions by nesting per-lang outputs.
        if len(langs) > 1 and "{lang}" not in base:
            return os.path.join(base, f"rf_{lang}{suffix}")
        return base.format(lang=lang)

    for lang in langs:
        fragment_chars = args.fragment_chars
        if fragment_chars is None:
            fragment_chars = 80

        rf_max_depth = args.rf_max_depth
        if rf_max_depth is None:
            rf_max_depth = 5

        word_max_features = args.word_max_features
        if word_max_features is None:
            word_max_features = 800

        char_max_features = args.char_max_features
        if char_max_features is None:
            char_max_features = 800

        json_dir = resolve_json_dir(args.json_dir, lang)
        out_dir = resolve_out_dir(args.out_dir, lang)

        json_files = sorted(glob.glob(os.path.join(json_dir, args.json_glob)))
        if not json_files:
            msg = f"No JSON files matched for lang={lang}: {os.path.join(json_dir, args.json_glob)}"
            if len(langs) > 1:
                print(msg)
                continue
            raise SystemExit(
                msg
                + "\nTip: run from anywhere with e.g. "
                + f"--json_dir {os.path.join(repo_root, 'language')} (or point directly to combined_{lang})."
            )

        for json_path in json_files:
            try:
                run_one(
                    json_path=json_path,
                    out_root=out_dir,
                    min_samples_per_author=args.min_samples_per_author,
                    max_samples_per_author=args.max_samples_per_author,
                    test_size=args.test_size,
                    seed=args.seed,
                    rf_max_depth=rf_max_depth,
                    include_message=args.include_message,
                    include_filename=args.include_filename,
                    fragment_chars=fragment_chars,
                    fragment_mode=args.fragment_mode,
                    feature_set=args.feature_set,
                    word_max_features=word_max_features,
                    char_max_features=char_max_features,
                )
            except Exception as e:
                print(f"Error processing {json_path}: {e}")


if __name__ == "__main__":
    main()
