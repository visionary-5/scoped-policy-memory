import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLDEN = json.loads((ROOT / "tests" / "golden" / "frozen_sha256.json").read_text())


@pytest.fixture(scope="session")
def golden():
    return GOLDEN


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """All CC-Gen frozen data (recipe arms, probe set, closeout views, repair data, few-shot views)."""
    from spm.experiments import main
    root = tmp_path_factory.mktemp("spm_data")
    main(["prepare", "--root", str(root)])
    return root


def reference_repo():
    """Path of the original research code, used only as a read-only behavioural reference."""
    p = Path(os.environ.get("SPM_REFERENCE_REPO", ROOT.parent / "rules2weights"))
    return p if (p / "r2w" / "world_ccgen.py").exists() else None
