"""Synthetic-world analysis data (gate-v1 V2 and exposure test set) against frozen SHA-256 (seed 0)."""
from analysis import exposure, gate_v2
from spm.io import sha256_file, write_jsonl


def test_gate_v2_and_exposure_test(tmp_path, golden):
    exp = golden["analysis"]
    R, arms = gate_v2.build()
    assert sha256_file(write_jsonl(tmp_path / "R.jsonl", R)) == exp["gate/data/R-s0.jsonl"]
    for arm in ("conf_k05", "cons_u05"):
        assert sha256_file(write_jsonl(tmp_path / f"{arm}.jsonl", arms[arm])) == exp[f"gate/data/{arm}-s0.jsonl"], arm
    rows = exposure.test_rows(0, exposure.new_world_rows())
    assert sha256_file(write_jsonl(tmp_path / "test-s0.jsonl", rows)) == exp["data/test-s0.jsonl"]

    # pairing: the two arms differ only in the practice rows' context and label
    conf = [d for d in arms["conf_k05"] if d.get("source", "").startswith("conf_")]
    cons = {d["pair_uid"]: d for d in arms["cons_u05"] if d.get("source", "").startswith("cons_")}
    assert len(conf) == len(cons) == 24
    for c in conf:
        k = cons[c["pair_uid"]]
        assert k["item"]["fields"]["fare"] == c["item"]["fields"]["fare"]
        assert "Policy updates" in c["messages"][0]["content"] and "Policy updates" not in k["messages"][0]["content"]
