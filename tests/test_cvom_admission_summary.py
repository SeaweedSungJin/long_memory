"""The only permitted cross-run change is the learned writer itself."""
from copy import deepcopy

import pytest

from run_scripts.robomme import summarize_cvom_admission as summary


def fixture():
    metadata = {"seed": 1, "actor_frozen": True}
    def model(arm):
        return {"mode": "fixed_parent", "memory_checkpoint": "same", "memory_off": False,
            "write_policy": "cvom-admission", "writer_checkpoint": arm, "writer_files_sha256": {"writer": arm},
            "cvom_admission": {"source_sha256": {"core": "same"}, "manifest": {
                "arm": arm, "parent_identity": "same", "config": "same", "step": 2000, "metadata": metadata}}}
    settings = {"dataset": "val", "n_episodes": 10, "seed": 6, "tasks": list(range(16)),
                "n_action_steps": 16, "max_episode_steps": 1300}
    left = {"settings": settings, "source_sha256": {"eval": "same"}, "models": {"memory": model("single")}}
    left["models"]["fifo"] = {**deepcopy(left["models"]["memory"]), "write_policy": "fifo"}
    right = {**deepcopy(left), "models": {"memory": model("coalitional")}}
    return left, right


def test_writer_only_difference_allowed(monkeypatch):
    monkeypatch.setattr(summary, "validate_manifest_contract", lambda value: None)
    summary.validate_study(*fixture())


@pytest.mark.parametrize("change", ["source", "parent", "seed", "budget", "swapped", "read_off"])
def test_other_differences_rejected(monkeypatch, change):
    monkeypatch.setattr(summary, "validate_manifest_contract", lambda value: None)
    left, right = fixture()
    model = right["models"]["memory"]
    if change == "source":
        right["source_sha256"]["eval"] = "changed"
    elif change == "parent":
        model["memory_checkpoint"] = "other"
    elif change == "seed":
        right["settings"]["seed"] = 7
    elif change == "budget":
        model["cvom_admission"]["manifest"]["config"] = "larger"
    elif change == "swapped":
        model["cvom_admission"]["manifest"]["arm"] = "single"
    else:
        left["models"]["fifo"]["memory_off"] = True
    with pytest.raises(ValueError):
        summary.validate_study(left, right)
