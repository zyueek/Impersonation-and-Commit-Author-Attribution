from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_STRING_RE = re.compile(r"(\"([^\"\\\\]|\\\\.)*\"|'([^'\\\\]|\\\\.)*')")


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _looks_camel(name: str) -> bool:
    return "_" not in name and any(ch.isupper() for ch in name[1:])


def _looks_snake(name: str) -> bool:
    return "_" in name and name.lower() == name and not name.startswith("_")


@dataclass(frozen=True)
class StyleProfile:
    indent_kind: str  # "tabs" | "spaces"
    indent_size: int  # if spaces
    brace_style: str  # "k&r" | "allman"
    space_after_keyword: bool
    quote_preference: str  # "single" | "double"
    ident_style: str  # "snake" | "camel" | "other"
    loop_preference: str  # "for" | "while" | "none"
    ident_pool: tuple[str, ...]  # common identifiers from the author


def build_profile(texts: Iterable[str]) -> StyleProfile:
    texts = [t for t in texts if t]
    if not texts:
        return StyleProfile(
            indent_kind="spaces",
            indent_size=4,
            brace_style="k&r",
            space_after_keyword=True,
            quote_preference="double",
            ident_style="snake",
            loop_preference="none",
            ident_pool=(),
        )

    leading_tabs = 0
    leading_spaces = 0
    space_indents = []

    newline_before_brace = 0
    total_braces = 0

    kw_space = 0
    kw_nospace = 0

    single_quotes = 0
    double_quotes = 0

    snake = 0
    camel = 0
    other = 0
    n_for = 0
    n_while = 0
    ident_freq: dict[str, int] = {}

    for text in texts:
        lines = text.splitlines()
        for ln in lines:
            if not ln.strip():
                continue
            m = re.match(r"^(\s+)", ln)
            if m:
                ws = m.group(1)
                if "\t" in ws and ws.lstrip("\t") == "":
                    leading_tabs += 1
                if " " in ws and ws.lstrip(" ") == "":
                    leading_spaces += 1
                    space_indents.append(len(ws))

        total_braces += text.count("{")
        newline_before_brace += len(re.findall(r"\n\s*\{", text))

        kw_space += len(re.findall(r"\b(if|for|while|switch|catch)\s+\(", text))
        kw_nospace += len(re.findall(r"\b(if|for|while|switch|catch)\(", text))

        for m in _STRING_RE.findall(text):
            lit = m[0]
            if lit.startswith("'"):
                single_quotes += 1
            elif lit.startswith('"'):
                double_quotes += 1

        for ident in _IDENT_RE.findall(text):
            if ident in {"if", "for", "while", "switch", "catch", "return", "class", "def", "import", "from"}:
                continue
            if _looks_snake(ident):
                snake += 1
            elif _looks_camel(ident):
                camel += 1
            else:
                other += 1
            if len(ident) >= 2 and not ident.startswith("_"):
                ident_freq[ident] = ident_freq.get(ident, 0) + 1

        # Control-flow preference (used by AST attack for Python).
        n_for += len(re.findall(r"\bfor\b", text))
        n_while += len(re.findall(r"\bwhile\b", text))

    indent_kind = "tabs" if leading_tabs > leading_spaces else "spaces"
    indent_size = 4
    if indent_kind == "spaces" and space_indents:
        # Pick a common indent size (2 or 4), fallback to median bucket.
        vals = np.array(space_indents, dtype=np.int32)
        for candidate in (2, 4):
            if (vals % candidate == 0).mean() >= 0.6:
                indent_size = candidate
                break
        else:
            indent_size = int(np.clip(int(np.median(vals)), 2, 8))

    brace_style = "allman" if _safe_ratio(newline_before_brace, total_braces) >= 0.5 else "k&r"
    space_after_keyword = kw_space >= kw_nospace
    quote_preference = "single" if single_quotes > double_quotes else "double"

    if snake >= camel and snake >= other:
        ident_style = "snake"
    elif camel >= snake and camel >= other:
        ident_style = "camel"
    else:
        ident_style = "other"

    if n_for == 0 and n_while == 0:
        loop_preference = "none"
    else:
        loop_preference = "while" if n_while > n_for else "for"

    # Keep a small pool of common identifiers (used by stronger token-aware renaming).
    ident_pool = tuple([k for k, _v in sorted(ident_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:200]])

    return StyleProfile(
        indent_kind=indent_kind,
        indent_size=indent_size,
        brace_style=brace_style,
        space_after_keyword=space_after_keyword,
        quote_preference=quote_preference,
        ident_style=ident_style,
        loop_preference=loop_preference,
        ident_pool=ident_pool,
    )
