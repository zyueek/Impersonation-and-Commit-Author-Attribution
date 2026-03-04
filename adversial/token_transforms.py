from __future__ import annotations

import random
import re
from dataclasses import dataclass

from adversial.style_profile import StyleProfile


_C_LIKE_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PHP_VAR = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def _norm_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang in {"py", "python"}:
        return "python"
    if lang in {"js", "javascript", "typescript", "node", "nodejs"}:
        return "js"
    if lang in {"go", "golang"}:
        return "go"
    if lang in {"java"}:
        return "java"
    if lang in {"php"}:
        return "php"
    return lang


def _to_snake(name: str) -> str:
    if "_" in name:
        return name.lower()
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _to_camel(name: str) -> str:
    if "_" not in name:
        return name[0].lower() + name[1:] if name else name
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    first = parts[0].lower()
    rest = "".join(p[:1].upper() + p[1:].lower() for p in parts[1:])
    return first + rest


def _compute_skip_spans(text: str, lang: str) -> list[tuple[int, int]]:
    """
    Returns spans to skip (strings/comments). Best-effort for C-like and PHP/JS/Go/Java.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    lang = (lang or "").lower()
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        # Line comments: // ... or # ... (php)
        if ch == "/" and nxt == "/":
            j = i + 2
            while j < n and text[j] != "\n":
                j += 1
            spans.append((i, j))
            i = j
            continue
        if lang in {"php"} and ch == "#":
            j = i + 1
            while j < n and text[j] != "\n":
                j += 1
            spans.append((i, j))
            i = j
            continue

        # Block comments: /* ... */
        if ch == "/" and nxt == "*":
            j = i + 2
            while j + 1 < n and not (text[j] == "*" and text[j + 1] == "/"):
                j += 1
            j = min(n, j + 2)
            spans.append((i, j))
            i = j
            continue

        # Strings: '...' and "..." with backslash escapes.
        if ch in {"'", '"'}:
            quote = ch
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote:
                    j += 1
                    break
                j += 1
            spans.append((i, min(n, j)))
            i = j
            continue

        # Backtick strings for js/go
        if ch == "`" and lang in {"js", "go"}:
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "`":
                    j += 1
                    break
                j += 1
            spans.append((i, min(n, j)))
            i = j
            continue

        i += 1

    spans.sort()
    return spans


def _in_spans(spans: list[tuple[int, int]], pos: int) -> bool:
    # spans are sorted; linear scan is ok for short snippets
    for s, e in spans:
        if pos < s:
            return False
        if s <= pos < e:
            return True
    return False


def _collect_declared_identifiers(masked: str, lang: str) -> list[str]:
    """
    Best-effort: find variable names declared in the snippet, per-language heuristics.
    masked is text with strings/comments already removed or replaced.
    """
    lang = (lang or "").lower()
    out: list[str] = []

    def add(name: str) -> None:
        if name and name not in out:
            out.append(name)

    if lang == "go":
        # var x, const x
        for m in re.finditer(r"\b(var|const)\s+([a-z_][A-Za-z0-9_]*)\b", masked):
            add(m.group(2))
        # short decl: x := ... or x, y := ...
        for m in re.finditer(r"\b([a-z_][A-Za-z0-9_]*(?:\s*,\s*[a-z_][A-Za-z0-9_]*)*)\s*:=", masked):
            names = [p.strip() for p in m.group(1).split(",")]
            for nm in names:
                add(nm)
        return out

    if lang == "js":
        for m in re.finditer(r"\b(let|const|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b", masked):
            add(m.group(2))
        # function params: function f(a,b) or (a,b)=> or a=> ...
        for m in re.finditer(r"\bfunction\b[^{(]*\(([^)]*)\)", masked):
            for p in m.group(1).split(","):
                p = p.strip()
                if re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", p):
                    add(p)
        for m in re.finditer(r"\(([^)]*)\)\s*=>", masked):
            for p in m.group(1).split(","):
                p = p.strip()
                if re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", p):
                    add(p)
        for m in re.finditer(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*=>", masked):
            add(m.group(1))
        for m in re.finditer(r"\bcatch\s*\(\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*\)", masked):
            add(m.group(1))
        return out

    if lang == "php":
        # variables: $x = ... ; foreach (... as $x => $y) ; function f($x,$y)
        for m in re.finditer(r"(\$[A-Za-z_][A-Za-z0-9_]*)\s*=", masked):
            add(m.group(1))
        for m in re.finditer(r"\bforeach\s*\(.*?\bas\s+(\$[A-Za-z_][A-Za-z0-9_]*)(?:\s*=>\s*(\$[A-Za-z_][A-Za-z0-9_]*))?", masked, flags=re.DOTALL):
            add(m.group(1))
            if m.group(2):
                add(m.group(2))
        for m in re.finditer(r"\bfunction\b[^{(]*\(([^)]*)\)", masked):
            for p in m.group(1).split(","):
                p = p.strip()
                if re.match(r"^\$[A-Za-z_][A-Za-z0-9_]*$", p):
                    add(p)
        return out

    if lang == "java":
        prim = r"(?:byte|short|int|long|float|double|boolean|char|String)"
        # primitive or String declarations: int x = ..., String name;
        for m in re.finditer(rf"\b{prim}\s+([a-z_][A-Za-z0-9_]*)\b", masked):
            add(m.group(1))
        # for loops: for (int i = ...; ...) or enhanced: for (Type x : ...)
        for m in re.finditer(rf"\bfor\s*\(\s*(?:final\s+)?(?:{prim}|[A-Z][A-Za-z0-9_<>\\[\\].]*)\s+([a-z_][A-Za-z0-9_]*)\s*(?:=|:)", masked):
            add(m.group(1))
        # catch: catch (Exception e)
        for m in re.finditer(r"\bcatch\s*\(\s*[A-Z][A-Za-z0-9_<>\\[\\].]*\s+([a-z_][A-Za-z0-9_]*)\s*\)", masked):
            add(m.group(1))
        return out

    return out


def _make_mapping(
    names: list[str],
    *,
    target: StyleProfile,
    lang: str,
    max_renames: int,
    rng: random.Random,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used = set(names)

    def fresh(base: str) -> str:
        if base not in used and base not in mapping.values():
            used.add(base)
            return base
        i = 2
        while True:
            cand = f"{base}{i}"
            if cand not in used and cand not in mapping.values():
                used.add(cand)
                return cand
            i += 1

    def convert(nm: str) -> str:
        if lang == "php" and nm.startswith("$"):
            core = nm[1:]
            core2 = _to_snake(core) if target.ident_style == "snake" else _to_camel(core) if target.ident_style == "camel" else core
            return "$" + core2
        return _to_snake(nm) if target.ident_style == "snake" else _to_camel(nm) if target.ident_style == "camel" else nm

    def normalize_candidate(nm: str) -> str:
        if lang == "php":
            nm = nm.lstrip("$")
        if target.ident_style == "snake":
            nm = _to_snake(nm)
        elif target.ident_style == "camel":
            nm = _to_camel(nm)
        return nm

    pool = [p for p in target.ident_pool if p and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", p)]

    reserved_js = {
        "console",
        "window",
        "document",
        "module",
        "exports",
        "require",
        "__dirname",
        "__filename",
        "process",
    }
    reserved_go = {"err", "ctx"}
    reserved_java = {"this", "super"}
    reserved_php = {"$this", "$GLOBALS"}

    for nm in names:
        if len(mapping) >= max_renames:
            break
        if lang == "js" and nm in reserved_js:
            continue
        if lang == "go" and nm in reserved_go:
            continue
        if lang == "java" and nm in reserved_java:
            continue
        if lang == "php" and nm in reserved_php:
            continue
        if lang == "php" and nm.startswith("$_"):
            continue
        # Try to pick a different identifier from the target's pool (stronger imitation).
        new = None
        if pool:
            for _ in range(40):
                cand = normalize_candidate(rng.choice(pool))
                if not cand or cand == nm or cand == nm.lstrip("$"):
                    continue
                cand2 = ("$" + cand) if (lang == "php" and nm.startswith("$")) else cand
                if cand2 in used or cand2 in mapping.values():
                    continue
                new = cand2
                break
        if new is None:
            # Fallback: style-normalize the existing name; if unchanged, still rename to a fresh synthetic name.
            new = convert(nm)
            if new == nm:
                base = "tmp_var" if target.ident_style == "snake" else "tmpVar" if target.ident_style == "camel" else "tmp"
                new = ("$" + base) if (lang == "php" and nm.startswith("$")) else base
        if lang == "go" and (nm[:1].isupper() or new[:1].isupper()):
            # Avoid exported/type-like names.
            continue
        mapping[nm] = fresh(new)

    return mapping


def _apply_mapping(text: str, *, mapping: dict[str, str], spans: list[tuple[int, int]], lang: str) -> str:
    if not mapping:
        return text

    if lang == "php":
        pat = _PHP_VAR
    else:
        pat = _C_LIKE_IDENT

    out = []
    last = 0
    for m in pat.finditer(text):
        s, e = m.start(), m.end()
        if _in_spans(spans, s):
            continue
        tok = m.group(0)
        if tok not in mapping:
            continue
        # Avoid renaming member accesses (obj.prop) and package selectors.
        prev = text[s - 1] if s - 1 >= 0 else ""
        if prev in {".", ":"}:
            continue
        nxt = text[e] if e < len(text) else ""
        # Avoid JS object keys like "foo:" and labels.
        if lang == "js":
            j = e
            while j < len(text) and text[j] in " \t":
                j += 1
            if j < len(text) and text[j] == ":":
                continue

        out.append(text[last:s])
        out.append(mapping[tok])
        last = e
    out.append(text[last:])
    return "".join(out)


@dataclass(frozen=True)
class TokenRenameReport:
    mapping: dict[str, str]
    declared: list[str]


def token_aware_rename(
    text: str,
    *,
    language: str,
    target_profile: StyleProfile,
    max_renames: int,
    rng: random.Random | None = None,
) -> tuple[str, TokenRenameReport] | None:
    lang = _norm_lang(language)
    if lang not in {"go", "java", "js", "php"}:
        return None
    rng = rng or random.Random(0)

    spans = _compute_skip_spans(text, lang=lang)
    # Build masked version by replacing spans with spaces to keep indices stable-ish for regex.
    masked = list(text)
    for s, e in spans:
        for i in range(s, e):
            masked[i] = " "
    masked_s = "".join(masked)

    declared = _collect_declared_identifiers(masked_s, lang=lang)
    if not declared:
        return None

    mapping = _make_mapping(declared, target=target_profile, lang=lang, max_renames=max_renames, rng=rng)
    if not mapping:
        return None

    out = _apply_mapping(text, mapping=mapping, spans=spans, lang=lang)
    if out == text:
        return None
    return out, TokenRenameReport(mapping=mapping, declared=declared)
