"""Chat messages for items. The system prompt is fixed; policy context is appended to it."""
from __future__ import annotations

import random

from ..eval.score import render_calls
from .world import World, make_item

PROMPT_VERSION = "tool-format-v3"
# System prompt of the synthetic (mixed) world.
SYSTEM = (
    "You are an airline operations agent. Output only one line of Python-style tool calls "
    "with keyword arguments. After each exact tool name, put arguments in parentheses, "
    'e.g. user="U0000", code=3. Copy actual names and values from the request and Rules. '
    "Quote strings, keep numbers unquoted (code/window integers), and use True/False for booleans. "
    "Separate calls with semicolons. Apply Rules, including tool renames. "
    "No JSON, Markdown or explanations."
)
SYSTEM_RECALL = "You are an airline operations agent. Answer the question about the rules."
# System prompt of CC-Gen-ST; the example call follows the official policy text.
SYSTEM_CCGEN = (
    "You are a helpful agent that completes tasks by following the Policy. Output only one line of Python-style "
    "tool calls with keyword arguments, e.g. finish_task_1(attributes=[25, 150, 42]) or "
    'tool_name(user="U0000", code=3). Compute every argument from the selected profile values and the Policy, '
    "in the order the Policy specifies. Quote strings and keep numbers unquoted. No JSON, Markdown or explanations."
)
SYSTEM_RECALL_CCGEN = "You are a helpful agent. Answer the question about the Policy."


def _is_ccgen(rule_type) -> bool:
    return str(rule_type).startswith("ccgen_")


def recall_system(rule_type: str) -> str:
    return SYSTEM_RECALL_CCGEN if _is_ccgen(rule_type) else SYSTEM_RECALL


def rules_block(rules: list[dict], shots: int = 0, world: World | None = None, K: int | None = None) -> str:
    """Rule text, optionally with ``shots`` train-split worked examples per rule."""
    if rules and rules[0]["kind"].startswith("ccgen_"):
        from .ccgen import policy_block
        return policy_block(rules, shots, world, K)
    lines = ["Rules:"]
    for r in rules:
        lines.append(f"- {r['text']}")
        if shots:
            rng = random.Random(f"shot-{r['id']}-{world.hash()}")
            for _ in range(shots):
                ex = make_item(world, r, "train", "exec", rng, K)
                lines.append(f"  Example: {ex['request']} -> {render_calls(ex['answer'])}")
    return "\n".join(lines)


def user_text(item: dict) -> str:
    if item["kind"] == "recall":
        return item["request"]
    return "Tools:\n" + "\n".join(f"- {t}" for t in item["tools"]) + f"\n\nRequest: {item['request']}"


def build_messages(item: dict, rules_text: str | None = None) -> list[dict]:
    if _is_ccgen(item.get("rule_type")):
        sys = SYSTEM_RECALL_CCGEN if item["kind"] == "recall" else SYSTEM_CCGEN
    else:
        sys = SYSTEM_RECALL if item["kind"] == "recall" else SYSTEM
    if rules_text:
        sys = sys + "\n\n" + rules_text
    return [{"role": "system", "content": sys}, {"role": "user", "content": user_text(item)}]


def combined_block(pinned: list[dict], retrieved: list[dict], world: World, K: int, shots: int = 1) -> str | None:
    """Pinned rules (no examples) + retrieved rules (``shots`` examples each)."""
    lines = rules_block(pinned, 0).splitlines() if pinned else ["Rules:"]
    if retrieved:
        lines += rules_block(retrieved, shots, world, K).splitlines()[1:]
    return "\n".join(lines) if len(lines) > 1 else None
