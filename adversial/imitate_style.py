from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable

from adversial.style_profile import StyleProfile
from adversial.ast_transforms import apply_python_ast_attack
from adversial.token_transforms import token_aware_rename


_KEYWORDS = ("if", "for", "while", "switch", "catch")


def _convert_indentation(text: str, *, target: StyleProfile) -> str:
    if not text:
        return text
    out_lines = []
    for ln in text.splitlines(True):
        m = re.match(r"^(\s+)(.*)$", ln, flags=re.DOTALL)
        if not m:
            out_lines.append(ln)
            continue
        ws, rest = m.group(1), m.group(2)
        # Only treat pure-leading indentation (spaces/tabs). Mixed indentation is left as-is.
        if not (ws.strip() == ""):
            out_lines.append(ln)
            continue

        if target.indent_kind == "tabs":
            # Replace leading spaces with tabs as much as possible.
            spaces = ws.replace("\t", " " * target.indent_size)
            n_tabs = len(spaces) // target.indent_size
            rem = len(spaces) % target.indent_size
            out_lines.append(("\t" * n_tabs) + (" " * rem) + rest)
        else:
            # Replace tabs with spaces.
            expanded = ws.replace("\t", " " * target.indent_size)
            out_lines.append(expanded + rest)
    return "".join(out_lines)


def _keyword_spacing(text: str, *, target: StyleProfile) -> str:
    if target.space_after_keyword:
        for kw in _KEYWORDS:
            text = re.sub(rf"\b{kw}\(", f"{kw} (", text)
    else:
        for kw in _KEYWORDS:
            text = re.sub(rf"\b{kw}\s+\(", f"{kw}(", text)
    return text


def _brace_style(text: str, *, target: StyleProfile) -> str:
    # Very lightweight, regex-only. Works for brace-heavy langs; mostly a no-op for Python.
    if target.brace_style == "allman":
        text = re.sub(r"\)\s*\{", ")\n{", text)
        text = re.sub(r"\b(else|try|finally|do)\s*\{", r"\1\n{", text)
    else:
        # Merge lines where a lone "{" follows a control line.
        text = re.sub(r"\)\s*\n\s*\{", ") {", text)
        text = re.sub(r"\b(else|try|finally|do)\s*\n\s*\{", r"\1 {", text)
    return text


def _convert_quotes(text: str, *, target: StyleProfile) -> str:
    # Conservative conversion: only flip quotes when the target quote does not appear inside.
    if not text:
        return text
    out = []
    last = 0
    for m in re.finditer(r"(\"([^\"\\\\]|\\\\.)*\"|'([^'\\\\]|\\\\.)*')", text):
        lit = m.group(0)
        out.append(text[last : m.start()])
        last = m.end()
        if target.quote_preference == "single" and lit.startswith('"'):
            inner = lit[1:-1]
            if "'" not in inner:
                out.append("'" + inner.replace("\\\"", "\"") + "'")
            else:
                out.append(lit)
        elif target.quote_preference == "double" and lit.startswith("'"):
            inner = lit[1:-1]
            if '"' not in inner:
                out.append('"' + inner.replace("\\'", "'") + '"')
            else:
                out.append(lit)
        else:
            out.append(lit)
    out.append(text[last:])
    return "".join(out)


def _to_snake(name: str) -> str:
    if "_" in name:
        return name.lower()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return s


def _to_camel(name: str) -> str:
    if "_" not in name:
        return name[0].lower() + name[1:] if name else name
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    first = parts[0].lower()
    rest = "".join(p[:1].upper() + p[1:].lower() for p in parts[1:])
    return first + rest


def _rename_identifiers(text: str, *, target: StyleProfile, max_renames: int = 6) -> str:
    if target.ident_style not in {"snake", "camel"}:
        return text

    # Find candidate LHS identifiers for simple assignments/declarations.
    candidates = []
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(=|:=)", text):
        ident = m.group(1)
        if ident in _KEYWORDS or ident in {"return", "class", "def", "import", "from", "var", "let", "const", "func"}:
            continue
        candidates.append(ident)

    # Stable order, keep only frequent candidates.
    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    uniq = uniq[:max_renames]

    mapping = {}
    for ident in uniq:
        if target.ident_style == "snake":
            new = _to_snake(ident)
        else:
            new = _to_camel(ident)
        if new != ident and new not in mapping.values() and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", new):
            mapping[ident] = new

    if not mapping:
        return text

    # Avoid renaming attribute accesses like obj.ident or pkg::ident.
    def repl(m: re.Match) -> str:
        name = m.group(0)
        return mapping.get(name, name)

    # Only rename standalone identifiers.
    pattern = r"(?<![\\.\\:])\\b(" + "|".join(re.escape(k) for k in mapping.keys()) + r")\\b(?!\\s*\\.)"
    return re.sub(pattern, repl, text)


def imitate_text(
    text: str,
    *,
    target_profile: StyleProfile,
    language: str | None = None,
    use_ast: bool = True,
) -> str:
    # Order matters:
    # 1) AST/UDC-level (when available) to do semantics-aware edits.
    # 2) Surface-level normalization to align with target profile.
    text = text.replace("\r\n", "\n")

    if use_ast and (language or "").lower() == "python":
        res = apply_python_ast_attack(text, target_profile=target_profile)
        if res is not None:
            text, _report = res

    text = _convert_indentation(text, target=target_profile)
    text = _brace_style(text, target=target_profile)
    text = _keyword_spacing(text, target=target_profile)
    text = _convert_quotes(text, target=target_profile)
    text = _rename_identifiers(text, target=target_profile)
    return text


@dataclass(frozen=True)
class AttackResult:
    attacked_text: str
    target_label: str
    meta: dict = field(default_factory=dict)


def targeted_attack(
    *,
    text: str,
    target_label: str,
    target_profile: StyleProfile,
    language: str | None = None,
    use_ast: bool = True,
    token_rename_max: int = 0,
    rng: random.Random | None = None,
) -> AttackResult:
    meta: dict = {}
    attacked = text.replace("\r\n", "\n")
    if use_ast and (language or "").lower() == "python":
        res = apply_python_ast_attack(attacked, target_profile=target_profile)
        if res is not None:
            attacked, report = res
            meta["ast_applied"] = report.applied
            if report.rename_map:
                meta["ast_rename_map"] = report.rename_map

    if token_rename_max and token_rename_max > 0:
        res2 = token_aware_rename(
            attacked,
            language=(language or ""),
            target_profile=target_profile,
            max_renames=int(token_rename_max),
            rng=rng,
        )
        if res2 is not None:
            attacked, rep = res2
            meta["token_rename_declared"] = rep.declared
            meta["token_rename_map"] = rep.mapping

    attacked = _convert_indentation(attacked, target=target_profile)
    attacked = _brace_style(attacked, target=target_profile)
    attacked = _keyword_spacing(attacked, target=target_profile)
    attacked = _convert_quotes(attacked, target=target_profile)
    attacked = _rename_identifiers(attacked, target=target_profile)
    return AttackResult(attacked_text=attacked, target_label=target_label, meta=meta)
