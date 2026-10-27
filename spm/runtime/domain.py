"""CC-Gen domain adapter: a policy plus a sequence of edits, mapped onto worlds, items and contexts.

* original policy  = ``build_world(rule_type, seed)``;
* one edit         = an official Override of one task (``override_world``) with version seed ``vseed``;
                     editing the same task again replaces the previous edit;
* policy version   = original policy + {task: vseed};
* write data       = the conflict-aware recipe ``prec`` (memory + Override + Substitute practice);
* repair data      = ``prec`` + 96 closure-format conflict rows (120-step continuation).

Answers of any version depend only on the item's own task formula and the shared constants, so a
versioned item is labelled with a single-task ``variant`` and checked by the world oracle.
"""
from __future__ import annotations

from ..data.ccgen import policy_block, tasks
from ..data.variants import override_world
from ..data.world import World, build_world, lookup_solve, make_item
from ..runtime.context import DELTA_HEAD


class CCGenDomain:
    name = "ccgen"

    def __init__(self, rule_type: str = "ccgen_t12a3s1_lite", seed: int = 0):
        self.rt, self.seed = rule_type, seed
        self.w = build_world(rule_type, seed)
        self.T = self.w.extra["T"]
        self.pid = self.w.extra["policy_id"]

    # ---------- versions ----------
    def tasks(self):
        return list(range(1, self.T + 1))

    def task_rule(self, edits: dict, k: int) -> dict:
        src = override_world(self.w, k, edits[k]) if k in edits else self.w
        return next(t for t in tasks(src) if t["task"] == k)

    def world_at(self, edits: dict) -> World:
        rules = [self.task_rule(edits, r["task"]) if r["kind"] == "ccgen_task" else r for r in self.w.rules]
        return World(self.rt, self.seed, rules, {**self.w.extra, "policy_edits": sorted(edits.items())})

    def formula(self, edits: dict, k: int):
        return self.task_rule(edits, k)["ops"]

    def is_change(self, edits: dict, k: int, vseed: int) -> bool:
        """An edit must actually change the current formula."""
        return self.formula({**edits, k: vseed}, k) != self.formula(edits, k)

    def full_text(self, edits: dict) -> str:
        return policy_block(self.world_at(edits).rules)

    def delta_text(self, edits: dict, changed) -> str | None:
        """Raw delta: the current sections of ``changed`` tasks under the CAP-CPT Override template."""
        changed = sorted(set(changed))
        if not changed:
            return None
        return DELTA_HEAD.format(pid=self.pid) + "\n\n".join(self.task_rule(edits, k)["text"].strip() for k in changed)

    # ---------- items ----------
    def tag(self, item: dict, edits: dict) -> dict:
        k = item["fields"]["task"]
        item["fields"].pop("variant", None)
        if k in edits:
            item["fields"]["variant"] = {"kind": "override", "task": k, "vseed": edits[k]}
        item["world"] = f"{self.rt}-s{self.seed}"
        assert item["answer"] == lookup_solve(self.w, self.T, item), "versioned answer disagrees with the oracle"
        return item

    def answer(self, item: dict, edits: dict):
        it = {**item, "fields": {k: v for k, v in item["fields"].items() if k != "variant"}}
        if it["fields"]["task"] in edits:
            it["fields"]["variant"] = {"kind": "override", "task": it["fields"]["task"], "vseed": edits[it["fields"]["task"]]}
        return lookup_solve(self.w, self.T, it)

    def item(self, edits: dict, k: int, split: str, rng) -> dict:
        w = self.world_at(edits)
        it = make_item(w, next(t for t in tasks(w) if t["task"] == k), split, "exec", rng, self.T)
        return self.tag(it, edits)

    # ---------- training data ----------
    def write_rows(self) -> list[dict]:
        from ..training.datasets import build_arms
        return build_arms(self.rt, self.seed)["prec"]

    def repair_rows(self, write_rows: list[dict], arm: str = "repair") -> list[dict]:
        from ..training.conflict import repair_rows
        extra = repair_rows(self.w, write_rows)
        return sorted(write_rows + extra[arm], key=lambda d: d["uid"])
