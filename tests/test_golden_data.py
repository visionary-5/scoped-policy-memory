"""Byte-identity of every frozen data file against SHA-256 recorded by the original experiments."""
from spm.data.world import build_world
from spm.io import sha256_file


def test_world_hashes(golden):
    for key, h in golden["world_hash"].items():
        rt, seed = key.rsplit("-s", 1)
        assert build_world(rt, int(seed)).hash() == h, key


def test_ccgen_files_match_frozen_sha(data_root, golden):
    bad = [rel for rel, h in golden["ccgen"].items() if sha256_file(data_root / rel) != h]
    assert not bad, bad


def test_generation_is_deterministic(data_root, tmp_path):
    from spm.training.datasets import prepare_recipe
    prepare_recipe(tmp_path, seeds=(1,), with_cons=False)
    rel = "recipe/data/ccgen_t12a3s1_lite/prec-s1.jsonl"
    assert sha256_file(tmp_path / rel) == sha256_file(data_root / rel)
