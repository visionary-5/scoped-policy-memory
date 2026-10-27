"""Scoring of one-line tool-call outputs.

Primary metric (``score_call``): strip one complete code fence and surrounding whitespace, take the
first non-empty line, parse it with ``ast`` as ``name(kw=literal, ...)`` calls separated by ``;``, and
compare with the gold calls. Truncated outputs are always wrong.

Expression-aware re-check (``score_call_expr``): untrained models often write unevaluated arithmetic
such as ``max(83, 89)`` as an argument value. ``parse_calls_eval`` additionally accepts values given
by a whitelisted, side-effect-free expression (``+ - * / // %``, unary minus, and the functions
``max min ord abs len int round sum``). Function names, argument names and argument counts must still
match the gold exactly. Report both metrics side by side; the primary metric is never replaced.
"""
from __future__ import annotations

import ast
import re

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.S)


def normalize_lenient(text: str) -> str:
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    for line in t.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _parse(line: str, value_fn, errors=(ValueError, SyntaxError)):
    try:
        mod = ast.parse(line, mode="exec")
    except SyntaxError:
        return None
    calls = []
    for stmt in mod.body:
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            return None
        c = stmt.value
        if not isinstance(c.func, ast.Name) or c.args:
            return None
        kw = {}
        for k in c.keywords:
            if k.arg is None:
                return None
            try:
                kw[k.arg] = value_fn(k.value)
            except errors:
                return None
        calls.append([c.func.id, kw])
    return calls or None


def parse_calls(line: str):
    """'a(x=1); b(y="z")' -> [['a', {'x': 1}], ['b', {'y': 'z'}]]; None if invalid."""
    return _parse(line, ast.literal_eval)


def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-9
    return a == b


def calls_equal(pred, gold) -> bool:
    if pred is None or len(pred) != len(gold):
        return False
    for (pn, pk), (gn, gk) in zip(pred, gold):
        if pn != gn or set(pk) != set(gk) or not all(_eq(pk[k], gk[k]) for k in gk):
            return False
    return True


def score_call(output: str, gold: list, *, truncated: bool = False) -> dict:
    if truncated:
        return {"correct": False, "strict": False, "parsed": False, "truncated": True}
    strict_ok = len(output.strip().splitlines()) == 1 and calls_equal(parse_calls(output.strip()), gold)
    pred = parse_calls(normalize_lenient(output))
    return {"correct": calls_equal(pred, gold), "strict": strict_ok, "parsed": pred is not None, "truncated": False}


def score_recall(output: str, letter: str, *, truncated: bool = False) -> dict:
    t = normalize_lenient(output).strip().strip("().*: ").upper()
    ok = (not truncated) and t[:1] == letter and (len(t) == 1 or not t[1].isalpha())
    return {"correct": ok, "strict": ok and output.strip() == letter, "parsed": bool(t), "truncated": truncated}


def render_calls(calls: list) -> str:
    """Canonical rendering of gold calls (training labels and worked examples)."""
    def v(x):
        return repr(x) if not isinstance(x, str) else '"' + x.replace('"', '\\"') + '"'
    return "; ".join(f"{n}({', '.join(f'{k}={v(x)}' for k, x in kw.items())})" for n, kw in calls)


# ---------- expression-aware re-check ----------
_BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b, ast.Div: lambda a, b: a / b}
_FN = {"max": max, "min": min, "ord": ord, "abs": abs, "len": len, "int": int, "round": round, "sum": sum}


def safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [safe_eval(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = safe_eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FN and not node.keywords:
        return _FN[node.func.id](*[safe_eval(a) for a in node.args])
    raise ValueError("unsafe expression")


def parse_calls_eval(line: str):
    return _parse(line, safe_eval, Exception)


def correct_expr(output: str, gold: list, truncated: bool, orig_correct: bool) -> bool:
    """Correct under the primary metric, or (not truncated and) correct after whitelisted evaluation."""
    new = (not truncated) and calls_equal(parse_calls_eval(normalize_lenient(output)), gold)
    return bool(new or orig_correct)


def score_call_expr(output: str, gold: list, *, truncated: bool = False) -> dict:
    s = score_call(output, gold, truncated=truncated)
    return {**s, "correct_expr": correct_expr(output, gold, truncated, s["correct"])}


def check_scores(rows) -> None:
    """Re-score per-item records and fail on any mismatch with the stored verdict."""
    for row in rows:
        if type(row["correct"]) is not bool or type(row["truncated"]) is not bool:
            raise ValueError("non-boolean verdict")
        if score_call(row["output"], row["answer"], truncated=row["truncated"])["correct"] != row["correct"]:
            raise ValueError("re-scoring disagrees with stored verdict")
