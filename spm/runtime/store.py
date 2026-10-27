"""Versioned, scoped rule store (one workspace per policy).

Workspace layout:
  workspace.json     domain, rule type, seed, backend, base model
  deltas.jsonl       append-only event log:
                       edit   one rule change (version + 1): {task, vseed}
                       write  an adapter that internalises the policy up to ``upto_version``
                       repair a format-continuation adapter (same version scope as its parent)
  adapters/<name>/   adapters produced by write / repair (real backend: LoRA; mock: mock_adapter.json)
  data/<name>.jsonl  the frozen training file of each adapter

All state is replayed from ``deltas.jsonl``:
  head_edits      = policy version after all edits {task: vseed}
  committed_edits = version internalised by the latest adapter
  pending         = tasks whose head version differs from the committed one (served via context)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

COMMIT_OPS = ("write", "repair", "consolidate")


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        self.cfg_path, self.log_path = self.root / "workspace.json", self.root / "deltas.jsonl"

    def init(self, **cfg):
        if self.cfg_path.exists():
            raise FileExistsError(f"workspace already exists: {self.root}")
        (self.root / "adapters").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(exist_ok=True)
        self.cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        self.log_path.touch()
        return self

    @property
    def cfg(self):
        return json.loads(self.cfg_path.read_text())

    def events(self):
        return [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]

    def _append(self, ev):
        ev = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **ev}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    # ---------- versions ----------
    @property
    def version(self):
        return sum(e["op"] == "edit" for e in self.events())

    def edits_at(self, version: int) -> dict:
        """Policy version ``version`` as {task: vseed}; a later edit of the same task wins."""
        out, v = {}, 0
        for e in self.events():
            if e["op"] == "edit":
                v += 1
                if v > version:
                    break
                out[e["task"]] = e["vseed"]
        return out

    def head_edits(self):
        return self.edits_at(self.version)

    def last_commit(self):
        commits = [e for e in self.events() if e["op"] in COMMIT_OPS]
        return commits[-1] if commits else None

    def committed_edits(self):
        c = self.last_commit()
        return self.edits_at(c["upto_version"]) if c else {}

    def pending(self) -> list[int]:
        head, base = self.head_edits(), self.committed_edits()
        return sorted(k for k in head if base.get(k) != head[k])

    def adapter(self):
        c = self.last_commit()
        return None if c is None else self.root / c["adapter"]

    # ---------- events ----------
    def add_edit(self, task: int, vseed: int, note: str = ""):
        return self._append({"op": "edit", "version": self.version + 1, "task": task, "vseed": vseed, "note": note})

    def commit(self, op: str, adapter: str, data: str | None, info: dict, upto_version: int | None = None):
        if op not in COMMIT_OPS:
            raise ValueError(op)
        upto = self.version if upto_version is None else upto_version
        return self._append({"op": op, "upto_version": upto, "adapter": adapter, "data": data, "info": info})

    def status(self):
        c = self.last_commit()
        return {"version": self.version, "committed_version": c["upto_version"] if c else None,
                "adapter": str(self.adapter()) if c else None, "pending_tasks": self.pending(),
                "head_edits": self.head_edits(), "n_commits": sum(e["op"] in COMMIT_OPS for e in self.events())}
