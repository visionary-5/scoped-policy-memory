"""Runtime context construction: explicit scope and dependency closure.

Context views for one request (``context_for``):
  full     the entire current policy (prompting baseline / fallback);
  matched  oracle section retrieval: shared constants + the request's own current task section
           (strong prompting baseline; same dependency information as ``closed``);
  raw      only the changed task section after the CAP-CPT "previously learnt" anchor (delta only);
  closed   dependency closure: shared constants + the changed section, and only when the request's
           task is the changed one; otherwise no context (the adapter's memory answers);
  typed    ``closed`` with a one-sentence header saying which part is updated.

The request task is parsed from the explicit ``task_type_k`` mention; update scope comes from the
store's version metadata. No answers or labels are used. An ambiguous request (zero or several task
mentions) falls back to the full current policy.
"""
from __future__ import annotations

import re

from ..data.ccgen import policy_block

SPEC = "\n\n## Task Specifications\n\n"
SEC_RE = re.compile(r"(?=^### Task_Type_\d+\s*$)", re.M)
TASK_RE = re.compile(r"task_type_(\d+)")
DELTA_HEAD = ("Based on the policy document #P{pid:05d} you previously learnt about, note that the following "
              "parts of the Policy has been changed:\n\n## Task Specifications\n\n")
TYPED_HEAD = ("Use the current task specification and shared constants below. "
              "Only the listed task is updated; other tasks retain their learned specifications.\n\n")
VIEWS = ("raw", "closed", "typed", "full", "matched")


def request_task(request):
    """The single task id mentioned as ``task_type_k`` (case-insensitive), or None if ambiguous."""
    tasks = set(int(k) for k in re.findall(r'\btask_type_(\d+)\b', request, re.I))
    return next(iter(tasks)) if len(tasks) == 1 else None


def closed_patch(world, patch, request, typed=False):
    """Keep only the queried changed section plus the unchanged shared constants.

    ``patch`` is text containing one or more ``### Task_Type_k`` sections (e.g. a raw delta).
    Returns None when the request's task is not in the patch (memory answers)."""
    k = request_task(request)
    if k is None:
        raise ValueError("ambiguous task; caller must supply the full current policy")
    sections = re.split(r'(?=^### Task_Type_\d+\s*$)', patch, flags=re.M)
    chosen = [s.strip() for s in sections if s.strip() and re.match(rf'^### Task_Type_{k}\s*$', s.splitlines()[0])]
    if not chosen:
        return None
    if len(chosen) != 1:
        raise ValueError("duplicate task section in patch")
    const = next(r for r in world.rules if r['kind'] == 'ccgen_const')
    text = (f"# Agent Policy Document #P{const['pid']:05d}\n\n" + const['text']
            + '\n\n## Task Specifications\n\n' + chosen[0])
    if typed:
        text = TYPED_HEAD + text
    return text


def raw_delta(world, sections: list[str]) -> str:
    return DELTA_HEAD.format(pid=world.extra["policy_id"]) + "\n\n".join(s.strip() for s in sections)


def context_for(world, current_world, request, changed_task, mode):
    """Context text for one request under view ``mode`` (see module docstring)."""
    k = request_task(request)
    if k is None:
        return policy_block(current_world.rules)
    if mode == 'full':
        return policy_block(current_world.rules)
    if mode == 'matched':
        return policy_block([r for r in current_world.rules
                             if r['kind'] == 'ccgen_const' or r.get('task') == k])
    if changed_task is None:
        return None
    rule = next(r for r in current_world.rules if r.get('task') == changed_task)
    patch = rule['text']
    if mode == 'raw':
        return (f"Based on the policy document #P{world.extra['policy_id']:05d} you previously learnt about, "
                "note that the following parts of the Policy has been changed:\n\n## Task Specifications\n\n" + patch)
    if mode in ('closed', 'typed'):
        return closed_patch(world, patch, request, mode == 'typed')
    raise ValueError(mode)


def split_doc(doc):
    """Full policy text -> (head = title + shared constants, {task: section text})."""
    head, body = doc.split(SPEC, 1)
    secs = {}
    for s in SEC_RE.split(body):
        s = s.strip()
        if not s:
            continue
        k = int(re.match(r'^### Task_Type_(\d+)\s*$', s.splitlines()[0]).group(1))
        assert k not in secs
        secs[k] = s
    return head, secs


def closure(head, secs, tasks):
    """Dependency closure over several sections (policy order). Equals ``closed_patch`` for one section."""
    return head + SPEC + '\n\n'.join(secs[k] for k in sorted(tasks))
