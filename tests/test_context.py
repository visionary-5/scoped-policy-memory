"""Dependency closure and context views."""
import pytest

from spm.data.ccgen import policy_block
from spm.data.variants import override_world
from spm.data.world import build_world
from spm.runtime.context import TYPED_HEAD, closed_patch, closure, context_for, raw_delta, request_task, split_doc

W = build_world("ccgen_t12a3s1_lite", 0)
REQ = "Hi. Please complete task_type_9 with the selected profiles below.\nSelected profiles:\n..."


def test_request_task():
    assert request_task(REQ) == 9
    assert request_task("TASK_TYPE_3 and task_type_3") == 3
    assert request_task("task_type_3 or task_type_4") is None
    assert request_task("no task here") is None


def test_closed_contains_constants_and_only_changed_section():
    vw = override_world(W, 9, 123)
    ctx = context_for(W, vw, REQ, 9, "closed")
    const = next(r for r in W.rules if r["kind"] == "ccgen_const")
    assert const["text"] in ctx                                   # shared constants (dependency)
    new_sec = next(r for r in vw.rules if r.get("task") == 9)["text"]
    assert ctx.endswith(new_sec)                                  # the changed section, new version
    assert ctx.count("### Task_Type_") == 1                       # no unrelated sections
    old_sec = next(r for r in W.rules if r.get("task") == 9)["text"]
    assert old_sec not in ctx
    # closure equals the multi-section helper and the matched view of the current policy
    head, secs = split_doc(policy_block(vw.rules))
    assert ctx == closure(head, secs, [9])
    assert ctx == context_for(W, vw, REQ, 9, "matched")


def test_typed_and_raw_and_unrelated():
    vw = override_world(W, 9, 123)
    assert context_for(W, vw, REQ, 9, "typed") == TYPED_HEAD + context_for(W, vw, REQ, 9, "closed")
    raw = context_for(W, vw, REQ, 9, "raw")
    assert raw.startswith("Based on the policy document #P") and "Global_Attribute_Value1" not in raw  # no dependency
    assert raw == raw_delta(W, [next(r for r in vw.rules if r.get("task") == 9)["text"]])
    other = override_world(W, 4, 123)                             # some other task changed
    assert context_for(W, other, REQ, 4, "closed") is None        # memory answers
    assert context_for(W, W, REQ, None, "closed") is None         # nothing changed


def test_ambiguous_request_falls_back_to_full_policy():
    vw = override_world(W, 9, 123)
    amb = "do task_type_9 and task_type_2"
    for mode in ("closed", "typed", "raw", "matched", "full"):
        assert context_for(W, vw, amb, 9, mode) == policy_block(vw.rules)
    with pytest.raises(ValueError):
        closed_patch(W, "### Task_Type_9\nx", amb)


def test_closed_patch_duplicate_section():
    with pytest.raises(ValueError):
        closed_patch(W, "### Task_Type_9\na\n\n### Task_Type_9\nb", REQ)
