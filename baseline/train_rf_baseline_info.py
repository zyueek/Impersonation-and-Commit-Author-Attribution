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
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MERGE_FROM_RE = re.compile(r"\bfrom\s+\S+")


def _message_stats(msg: str, *, bucket_chars: int | None, bucket_words: int | None) -> tuple[int, int]:
    msg = str(msg or "")
    msg = _EMAIL_RE.sub("<EMAIL>", msg)
    msg = _MERGE_FROM_RE.sub("from <REF>", msg)
    n_chars = _bucket(len(msg), bucket_chars)
    n_words = _bucket(len(msg.split()), bucket_words)
    return int(n_chars), int(n_words)


def _join_code_lines(item: dict) -> str:
    added = item.get("added_code", [])
    if isinstance(added, list):
        return "\n".join(map(str, added))
    return str(added or "")


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


def _path_tokens(path: str) -> list[str]:
    if not path:
        return []
    norm = path.replace("\\", "/")
    parts = [p for p in norm.split("/") if p and p != "."]
    tokens = []
    tokens.extend(parts)
    base = parts[-1] if parts else norm
    if "." in base:
        tokens.append(base.rsplit(".", 1)[-1])
    return tokens


def _bucket(val: int, step: int | None) -> int:
    if step is None or step <= 1:
        return int(val)
    return int(val // step) * int(step)


def _filename_repr(filename: str, filename_mode: str) -> str:
    mode = (filename_mode or "full").lower().strip()
    norm = filename.replace("\\", "/")
    base = norm.split("/")[-1] if norm else ""
    if mode == "full":
        return filename
    if mode == "basename":
        return base
    if mode == "ext":
        if "." in base:
            return base.rsplit(".", 1)[-1]
        return ""
    raise ValueError(f"Unknown filename_mode={filename_mode!r} (use: full|basename|ext)")


def _build_info_text(
    item: dict,
    *,
    msg_bucket_chars: int | None,
    msg_bucket_words: int | None,
    include_message_stats: bool,
    include_filename_tokens: bool,
    filename_mode: str,
    stats_bucket_lines: int | None,
    stats_bucket_chars: int | None,
) -> str:
    msg_chars, msg_words = _message_stats(item.get("message", ""), bucket_chars=msg_bucket_chars, bucket_words=msg_bucket_words)
    raw_filename = str(item.get("filename", "") or "")
    filename = _filename_repr(raw_filename, filename_mode=filename_mode)

    filename_tokens = _path_tokens(raw_filename) if (include_filename_tokens and raw_filename) else []

    parts = []
    if include_message_stats:
        parts.append(f"[MSGSTATS] chars={msg_chars} words={msg_words}")
    if filename:
        parts.append(f"[FILE] {filename}")
    if filename_tokens:
        parts.append("[FILETOK] " + " ".join(filename_tokens))

    # Lightweight stats as discrete tokens (helps when using word-only features).
    code = _join_code_lines(item)
    lines = _bucket(len(code.splitlines()), stats_bucket_lines)
    chars = _bucket(len(code), stats_bucket_chars)
    parts.append(f"[STATS] lines={lines} chars={chars}")
    return "\n".join(parts)


def _build_text(
    item: dict,
    *,
    fragment_chars: int | None,
    fragment_mode: str,
    rng: np.random.RandomState,
    msg_bucket_chars: int | None,
    msg_bucket_words: int | None,
    include_message_stats: bool,
    include_filename_tokens: bool,
    filename_mode: str,
    stats_bucket_lines: int | None,
    stats_bucket_chars: int | None,
) -> str:
    code = _join_code_lines(item)
    code_fragment = _make_fragment(code, fragment_chars=fragment_chars, fragment_mode=fragment_mode, rng=rng)
    if not str(code_fragment).strip():
        return ""
    info = _build_info_text(
        item,
        msg_bucket_chars=msg_bucket_chars,
        msg_bucket_words=msg_bucket_words,
        include_message_stats=include_message_stats,
        include_filename_tokens=include_filename_tokens,
        filename_mode=filename_mode,
        stats_bucket_lines=stats_bucket_lines,
        stats_bucket_chars=stats_bucket_chars,
    )
    return f"{info}\n[CODE]\n{code_fragment}".strip()


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
    *,
    min_samples_per_author: int,
    max_samples_per_author: int | None,
    seed: int,
    fragment_chars: int | None,
    fragment_mode: str,
    msg_bucket_chars: int | None,
    msg_bucket_words: int | None,
    include_message_stats: bool,
    include_filename_tokens: bool,
    filename_mode: str,
    stats_bucket_lines: int | None,
    stats_bucket_chars: int | None,
) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    rows = []
    for item in data:
        rows.append(
            {
                "text": _build_text(
                    item,
                    fragment_chars=fragment_chars,
                    fragment_mode=fragment_mode,
                    rng=rng,
                    msg_bucket_chars=msg_bucket_chars,
                    msg_bucket_words=msg_bucket_words,
                    include_message_stats=include_message_stats,
                    include_filename_tokens=include_filename_tokens,
                    filename_mode=filename_mode,
                    stats_bucket_lines=stats_bucket_lines,
                    stats_bucket_chars=stats_bucket_chars,
                ),
                "emailname": item.get("emailname", ""),
                "raw": item,
            }
        )
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


@dataclass(frozen=True)
class RFParams:
    n_estimators: int
    max_features: str | int | float | None
    max_depth: int | None


def _rf_params(preset: str, max_depth: int | None) -> RFParams:
    # Back-compat signature, but RF baseline is weak-only.
    _ = preset
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

    return Pipeline(steps=[("features", features), ("drop_constants", VarianceThreshold(0.0)), ("clf", clf)])


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

    out_path = os.path.join(output_dir, "evaluation_results_info.json")
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
    rf_max_depth: int | None,
    fragment_chars: int | None,
    fragment_mode: str,
    feature_set: str,
    msg_bucket_chars: int | None,
    msg_bucket_words: int | None,
    include_message_stats: bool,
    include_filename_tokens: bool,
    filename_mode: str,
    stats_bucket_lines: int | None,
    stats_bucket_chars: int | None,
    word_max_features: int,
    char_max_features: int,
):
    print(f"Processing: {json_path}")
    df = _load_and_preprocess(
        json_path,
        min_samples_per_author=min_samples_per_author,
        max_samples_per_author=max_samples_per_author,
        seed=seed,
        fragment_chars=fragment_chars,
        fragment_mode=fragment_mode,
        msg_bucket_chars=msg_bucket_chars,
        msg_bucket_words=msg_bucket_words,
        include_message_stats=include_message_stats,
        include_filename_tokens=include_filename_tokens,
        filename_mode=filename_mode,
        stats_bucket_lines=stats_bucket_lines,
        stats_bucket_chars=stats_bucket_chars,
    )
    if df.empty or df["emailname"].nunique() < 2:
        print("Skip: not enough data after filtering.")
        return

    label_map = {author: i for i, author in enumerate(sorted(df["emailname"].unique()))}
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["emailname"])

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
        rf=_rf_params("weak", rf_max_depth),
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
            "method": "rf_info",
            "json_path": json_path,
            "n_samples": int(len(df)),
            "n_authors": int(df["emailname"].nunique()),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "rf_preset": "weak",
            "rf_max_depth": rf_max_depth,
            "fragment_chars": fragment_chars,
            "fragment_mode": fragment_mode,
            "feature_set": fs,
            "msg_bucket_chars": msg_bucket_chars,
            "msg_bucket_words": msg_bucket_words,
            "include_message_stats": bool(include_message_stats),
            "include_filename_tokens": include_filename_tokens,
            "filename_mode": filename_mode,
            "stats_bucket_lines": stats_bucket_lines,
            "stats_bucket_chars": stats_bucket_chars,
            "word_max_features": word_max_features,
            "char_max_features": char_max_features,
            "min_samples_per_author": min_samples_per_author,
            "max_samples_per_author": max_samples_per_author,
            "test_size": test_size,
            "seed": seed,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="RF baseline with Com-Info-style commit metadata in the input.")
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language to run: go, java, js, php, python; or 'all'; or comma-separated list (e.g. 'python,js')",
    )
    parser.add_argument("--json_dir", type=str, default=None)
    parser.add_argument("--json_glob", type=str, default="*.json")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output root; supports {lang}. Default: baseline/rf_{lang}_weak_info",
    )

    parser.add_argument("--min_samples_per_author", type=int, default=5)
    parser.add_argument("--max_samples_per_author", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # RF baseline is weak-only (no strong mode).
    parser.add_argument("--rf_max_depth", type=int, default=None)

    parser.add_argument("--fragment_chars", type=int, default=None)
    parser.add_argument("--fragment_mode", type=str, default="random", choices=["head", "tail", "random"])
    parser.add_argument("--feature_set", type=str, default="auto", choices=["auto", "full", "word_only"])

    # Info-only controls. Do not include raw commit message content; use bucketed message stats only.
    parser.add_argument("--msg_bucket_chars", type=int, default=200, help="Bucket commit message char length to this step (<=1 disables bucketing).")
    parser.add_argument("--msg_bucket_words", type=int, default=20, help="Bucket commit message word count to this step (<=1 disables bucketing).")
    msg_group = parser.add_mutually_exclusive_group()
    msg_group.add_argument("--include_message_stats", dest="include_message_stats", action="store_true", help="Include bucketed message stats (default: enabled).")
    msg_group.add_argument("--no_message_stats", dest="include_message_stats", action="store_false", help="Disable message stats.")
    parser.set_defaults(include_message_stats=True)
    tok_group = parser.add_mutually_exclusive_group()
    tok_group.add_argument(
        "--include_filename_tokens",
        dest="include_filename_tokens",
        action="store_true",
        help="Include filename path-part tokens (default: disabled).",
    )
    tok_group.add_argument(
        "--no_filename_tokens",
        dest="include_filename_tokens",
        action="store_false",
        help="Disable filename path-part tokens.",
    )
    parser.set_defaults(include_filename_tokens=False)
    parser.add_argument("--filename_mode", type=str, default="ext", choices=["full", "basename", "ext"])
    parser.add_argument("--stats_bucket_lines", type=int, default=200, help="Bucket STATS lines to this step (<=1 disables bucketing).")
    parser.add_argument("--stats_bucket_chars", type=int, default=5000, help="Bucket STATS chars to this step (<=1 disables bucketing).")

    parser.add_argument("--word_max_features", type=int, default=None)
    parser.add_argument("--char_max_features", type=int, default=None)

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
            return os.path.join(this_dir, f"rf_{lang}_weak_info")
        base = os.path.abspath(os.path.expanduser(base))
        if len(langs) > 1 and "{lang}" not in base:
            return os.path.join(base, f"rf_{lang}_weak_info")
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
                    rf_max_depth=rf_max_depth,
                    fragment_chars=fragment_chars,
                    fragment_mode=args.fragment_mode,
                    feature_set=args.feature_set,
                    msg_bucket_chars=args.msg_bucket_chars,
                    msg_bucket_words=args.msg_bucket_words,
                    include_message_stats=args.include_message_stats,
                    include_filename_tokens=args.include_filename_tokens,
                    filename_mode=args.filename_mode,
                    stats_bucket_lines=args.stats_bucket_lines,
                    stats_bucket_chars=args.stats_bucket_chars,
                    word_max_features=word_max_features,
                    char_max_features=char_max_features,
                )
            except Exception as e:
                print(f"Error processing {json_path}: {e}")


if __name__ == "__main__":
    main()
