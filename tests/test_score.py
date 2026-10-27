from spm.eval.score import (check_scores, normalize_lenient, parse_calls, parse_calls_eval, render_calls, score_call,
                            score_call_expr, score_recall)

GOLD = [["finish_task_3", {"attributes": [83, "ab", 7]}]]


def test_render_roundtrip():
    s = render_calls(GOLD)
    assert s == "finish_task_3(attributes=[83, 'ab', 7])"
    assert score_call(s, GOLD) == {"correct": True, "strict": True, "parsed": True, "truncated": False}


def test_lenient_normalisation():
    out = "```python\nfinish_task_3(attributes=[83, 'ab', 7])\n```"
    assert normalize_lenient(out) == "finish_task_3(attributes=[83, 'ab', 7])"
    r = score_call(out, GOLD)
    assert r["correct"] and not r["strict"]
    assert score_call("\n\nfinish_task_3(attributes=[83, 'ab', 7])\nextra text", GOLD)["correct"]


def test_wrong_and_truncated():
    assert not score_call("finish_task_3(attributes=[83, 'ab', 8])", GOLD)["correct"]
    assert not score_call("finish_task_4(attributes=[83, 'ab', 7])", GOLD)["correct"]
    assert not score_call("finish_task_3(attrs=[83, 'ab', 7])", GOLD)["correct"]
    assert score_call(render_calls(GOLD), GOLD, truncated=True)["correct"] is False
    assert parse_calls("finish_task_3([1])") is None and parse_calls("x = 1") is None


def test_bool_and_float_equality():
    gold = [["t", {"flag": True, "x": 1}]]
    assert score_call("t(flag=True, x=1.0)", gold)["correct"]
    assert not score_call("t(flag=1, x=1)", gold)["correct"]


def test_expression_rescoring():
    out = "finish_task_3(attributes=[max(83, 12), 'ab', 3 + 4])"
    assert not score_call(out, GOLD)["correct"]
    r = score_call_expr(out, GOLD)
    assert r["correct_expr"] and not r["correct"]
    assert parse_calls_eval("f(a=ord('a') - 90, b=min(3, sum([1, 2])))") == [["f", {"a": 7, "b": 3}]]
    assert parse_calls_eval("f(a=__import__('os'))") is None   # non-whitelisted call
    assert parse_calls_eval("f(a=max(1, key=abs))") is None    # keywords are rejected
    assert not score_call_expr("finish_task_3(attributes=[max(83, 12), 'ab', 3 + 4])", GOLD, truncated=True)["correct_expr"]


def test_recall_and_check_scores():
    assert score_recall("B) because", "B")["correct"] and not score_recall("Because", "B")["correct"]
    rows = [{"output": render_calls(GOLD), "answer": GOLD, "truncated": False, "correct": True}]
    check_scores(rows)
    rows[0]["correct"] = False
    try:
        check_scores(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatch not detected")
