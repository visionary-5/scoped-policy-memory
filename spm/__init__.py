"""Scoped Policy Memory (SPM).

Stable agent policies are written into LoRA weights with conflict-aware training;
versioned, scoped edits are served at runtime as dependency-closed context.

Subpackages:
  spm.data       deterministic policy worlds (CC-Gen-ST-Lite, synthetic mixed world), prompts, item recipes
  spm.training   conflict-aware data construction and the LoRA trainer
  spm.runtime    versioned rule store, scope routing, dependency closure, serving CLI
  spm.eval       scoring, probe sets, confirmation sets, generation and IFEval wrappers
  spm.baselines  plain SFT / consistent-context / mixed exposure, CAP-CPT, prompting baselines
"""

__version__ = "0.1.0"
