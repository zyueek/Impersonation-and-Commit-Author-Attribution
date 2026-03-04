from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass
from typing import Iterable

from adversial.style_profile import StyleProfile


class AstTransformError(Exception):
    pass


def _collect_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _fresh_name(used: set[str], base: str) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        cand = f"{base}{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


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


def _is_safe_ident(name: str) -> bool:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return False
    if keyword.iskeyword(name):
        return False
    return True


@dataclass(frozen=True)
class DefUseChains:
    defs: dict[str, int]
    uses: dict[str, int]


def compute_udc(tree: ast.AST) -> DefUseChains:
    defs: dict[str, int] = {}
    uses: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Param)):
                defs[node.id] = defs.get(node.id, 0) + 1
            elif isinstance(node.ctx, ast.Load):
                uses[node.id] = uses.get(node.id, 0) + 1
    return DefUseChains(defs=defs, uses=uses)


class _RenameLocals(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Do not rename the function itself here; only its args/body.
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        return node


def rename_locals_with_udc(text: str, *, target: StyleProfile, max_renames: int = 10) -> tuple[str, dict[str, str]] | None:
    if target.ident_style not in {"snake", "camel"}:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    used = _collect_names(tree)
    udc = compute_udc(tree)

    # Candidate locals: defined and used, avoid dunder and common placeholders.
    candidates = []
    for name, dcnt in udc.defs.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in {"_", "self", "cls"}:
            continue
        ucnt = udc.uses.get(name, 0)
        if ucnt <= 0:
            continue
        candidates.append((name, dcnt + ucnt))
    candidates.sort(key=lambda x: (-x[1], x[0]))

    mapping: dict[str, str] = {}
    for name, _score in candidates[: max_renames * 3]:
        if name in mapping:
            continue
        new = _to_snake(name) if target.ident_style == "snake" else _to_camel(name)
        if new == name:
            continue
        if not _is_safe_ident(new):
            continue
        if new in used or new in mapping.values():
            new = _fresh_name(used, new)
        mapping[name] = new
        if len(mapping) >= max_renames:
            break

    if not mapping:
        return None

    new_tree = _RenameLocals(mapping).visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        out = ast.unparse(new_tree)
    except Exception as e:
        raise AstTransformError(str(e)) from e
    return out, mapping


class _ForToWhile(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        if self.changed:
            return node
        if node.orelse:
            return node

        # for <target> in <iter>: <body>
        iter_name = ast.Name(id="__style_iter", ctx=ast.Store())
        iter_value = ast.Call(func=ast.Name(id="iter", ctx=ast.Load()), args=[node.iter], keywords=[])
        assign_iter = ast.Assign(targets=[iter_name], value=iter_value)

        next_call = ast.Call(func=ast.Name(id="next", ctx=ast.Load()), args=[ast.Name(id="__style_iter", ctx=ast.Load())], keywords=[])
        assign_target = ast.Assign(targets=[node.target], value=next_call)

        try_body: list[ast.stmt] = [assign_target] + node.body

        except_handler = ast.ExceptHandler(
            type=ast.Name(id="StopIteration", ctx=ast.Load()),
            name=None,
            body=[ast.Break()],
        )

        while_node = ast.While(
            test=ast.Constant(value=True),
            body=[
                ast.Try(
                    body=try_body,
                    handlers=[except_handler],
                    orelse=[],
                    finalbody=[],
                )
            ],
            orelse=[],
        )

        self.changed = 1
        return [assign_iter, while_node]


def for_to_while(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    used = _collect_names(tree)
    # Ensure our helper name doesn't collide.
    helper = "__style_iter"
    if helper in used:
        helper = _fresh_name(used, helper)

    tr = _ForToWhile()
    new_tree = tr.visit(tree)
    ast.fix_missing_locations(new_tree)
    if not tr.changed:
        return None

    # Patch the helper name if we had to choose a different one.
    if helper != "__style_iter":
        rename = _RenameLocals({"__style_iter": helper})
        new_tree = rename.visit(new_tree)
        ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


def _cfg_add_linear_edges(edges: dict[int, set[int]], n: int) -> None:
    for i in range(n - 1):
        edges.setdefault(i, set()).add(i + 1)


def compute_cfg_edges(body: list[ast.stmt]) -> dict[int, set[int]]:
    """
    Very small statement-level CFG approximation for a single block.
    Nodes are statement indices (0..len(body)-1).
    """
    edges: dict[int, set[int]] = {}
    n = len(body)
    if n <= 0:
        return edges

    _cfg_add_linear_edges(edges, n)

    for i, st in enumerate(body):
        # Terminal statements don't fall through.
        if isinstance(st, (ast.Return, ast.Raise)):
            edges[i] = set()
            continue

        # Branching.
        if isinstance(st, ast.If):
            edges.setdefault(i, set()).discard(i + 1) if (i + 1) in edges.get(i, set()) else None
            # If branch
            if st.body:
                edges.setdefault(i, set()).add(i + 1)  # enter body (approx: next stmt)
            else:
                edges.setdefault(i, set()).add(i + 1 if i + 1 < n else i)
            # Else branch
            if st.orelse:
                edges.setdefault(i, set()).add(i + 1)  # same approximation
            else:
                edges.setdefault(i, set()).add(i + 1 if i + 1 < n else i)

        # Loops: model as potentially skipping and potentially looping.
        if isinstance(st, (ast.For, ast.While)):
            if st.body:
                edges.setdefault(i, set()).add(i + 1)  # enter body (approx)
            # loop back edge
            edges.setdefault(i, set()).add(i)
            # also can exit to i+1 (already in linear edges)

        # Break/continue are conservatively terminal within this block.
        if isinstance(st, (ast.Break, ast.Continue)):
            edges[i] = set()

    return edges


class _InsertNoopCfg(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def _insert_into(self, body: list[ast.stmt]) -> list[ast.stmt]:
        if self.changed or not body:
            return body
        edges = compute_cfg_edges(body)
        # Pick a simple node with single successor; avoid control statements.
        for i, st in enumerate(body):
            if self.changed:
                break
            if isinstance(st, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Return, ast.Raise, ast.Break, ast.Continue)):
                continue
            if len(edges.get(i, set())) != 1:
                continue
            body = body[: i + 1] + [ast.Pass()] + body[i + 1 :]
            self.changed = 1
            break
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:
        self.generic_visit(node)
        node.body = self._insert_into(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._insert_into(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._insert_into(node.body)
        return node


def cfg_insert_noop(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    tr = _InsertNoopCfg()
    new_tree = tr.visit(tree)
    ast.fix_missing_locations(new_tree)
    if not tr.changed:
        return None
    return ast.unparse(new_tree)


@dataclass(frozen=True)
class AstAttackReport:
    applied: list[str]
    rename_map: dict[str, str]


def apply_python_ast_attack(
    text: str,
    *,
    target_profile: StyleProfile,
) -> tuple[str, AstAttackReport] | None:
    applied: list[str] = []
    rename_map: dict[str, str] = {}

    # Control transformation (like adv.pdf): for -> while, only when target prefers while.
    if getattr(target_profile, "loop_preference", None) == "while":
        out = for_to_while(text)
        if out is not None and out != text:
            text = out
            applied.append("for_to_while")

    # CFG-guided transformation: insert a harmless no-op statement into a simple basic block.
    out = cfg_insert_noop(text)
    if out is not None and out != text:
        text = out
        applied.append("cfg_insert_noop")

    # UDC-guided declaration transformation: rename locals consistently.
    renamed = rename_locals_with_udc(text, target=target_profile, max_renames=10)
    if renamed is not None:
        text, rename_map = renamed
        applied.append("udc_rename_locals")

    if not applied:
        return None

    return text, AstAttackReport(applied=applied, rename_map=rename_map)
