"""Versioned store and the write -> update -> serve CLI (mock backend, CPU)."""
from spm.runtime.cli import main
from spm.runtime.store import Workspace


def test_store_versions(tmp_path):
    w = Workspace(tmp_path / "ws").init(domain="ccgen", rule_type="ccgen_t12a3s1_lite", seed=0, backend="mock", model=None)
    w.commit("write", "adapters/a", None, {}, upto_version=0)
    w.add_edit(3, 11); w.add_edit(5, 12); w.add_edit(3, 13)
    assert w.version == 3 and w.head_edits() == {3: 13, 5: 12}
    assert w.edits_at(1) == {3: 11} and w.pending() == [3, 5]
    w.commit("repair", "adapters/b", None, {}, upto_version=0)   # repair does not internalise edits
    assert w.pending() == [3, 5] and str(w.adapter()).endswith("adapters/b")


def test_cli_end_to_end(tmp_path):
    ws = str(tmp_path / "ws")
    main(["init", "--ws", ws]); main(["write", "--ws", ws, "--steps", "800"]); main(["repair", "--ws", ws])
    main(["update", "--ws", ws, "--task", "9"])
    res = main(["serve", "--ws", ws, "--n", "24"])
    assert res["pending_tasks"] == [9] and res["fallback_rate"] == 0.0
    full = main(["serve", "--ws", ws, "--n", "24", "--context-policy", "full"])
    assert res["mean_context_chars"] < 0.2 * full["mean_context_chars"]
    status = Workspace(ws).status()
    assert status["version"] == 1 and status["committed_version"] == 0 and status["n_commits"] == 2


def test_cli_training_data_is_the_paper_data(tmp_path, data_root, golden):
    """`spm write` trains on exactly the prec file and `spm repair` on exactly the frozen repair file."""
    from spm.io import sha256_file
    ws = tmp_path / "ws"
    main(["init", "--ws", str(ws)]); main(["write", "--ws", str(ws)]); main(["repair", "--ws", str(ws)])
    assert sha256_file(ws / "data" / "v0-write.jsonl") == sha256_file(data_root / "recipe/data/ccgen_t12a3s1_lite/prec-s0.jsonl")
    assert sha256_file(ws / "data" / "v0-repair.jsonl") == golden["ccgen"]["closeout/repair/data/repair-s0.jsonl"]
