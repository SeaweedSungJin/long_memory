"""CPU-only V18 evaluation contracts; never start a simulator or large model."""
from __future__ import annotations
from collections import OrderedDict
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run_scripts.robomme import eval_representation_v18 as ev
from run_scripts.robomme.baseline_reference_v18 import equivalent_v18
from run_scripts.robomme.baseline_reference_v10 import SERVERS


def args(*extra):
    result = ev.build_parser().parse_args(list(extra))
    ev.validate_options(result)
    return result


def manifest():
    baseline = {"base_model": {"path": "/original", "id": "base"}, "memory_checkpoint": None,
        "mode": "none", "write_policy": "none", "write_policy_override": "checkpoint",
        "memory_off": False, "archive_read_off": False, "server_script": ev.BASELINE_SERVER}
    common = {"base_model": baseline["base_model"], "memory_checkpoint": "/candidate", "mode": ev.VARIANT,
        "step": 100, "memory_off": False, "archive_read_off": False, "server_script": ev.SERVER,
        "write_policy": "fifo", "representation_config": {"representation": "adapted_short"},
        "checkpoint_files_sha256": {p: "a" * 64 for p in ("checkpoint.json", "model.safetensors", "expert.safetensors")},
        "training_metadata": {"payload_sha256": {"model.safetensors": "a" * 64}}}
    result = {"format_version": 1, "trainer_variant": ev.VARIANT,
        "models": {"baseline": baseline, "memory": common, "memory-off": dict(common, memory_off=True)},
        "settings": {"tasks": ["BinFill"], "n_episodes": 1, "dataset": "val", "seed": 6,
            "n_action_steps": 16, "max_episode_steps": 1300, "save_videos": False, "device": "cuda:0"}}
    result["evaluation_id"] = ev.identity_digest(result)
    return result


def rehash(m):
    m["evaluation_id"] = ev.identity_digest(m)
    return m


def test_fast160_defaults_and_candidate_only_reference():
    parsed = args("--checkpoint", "/candidate", "--models", "memory", "--baseline-reference", "/reference")
    assert len(parsed.tasks) * parsed.n_episodes == 160
    assert parsed.dataset == "val" and parsed.seed == 6
    assert parsed.models == ["baseline", "memory"]
    assert args("--models", "baseline").checkpoint is None
    with unittest.TestCase().assertRaisesRegex(ValueError, "require --checkpoint"):
        args("--models", "baseline", "memory")
    with unittest.TestCase().assertRaisesRegex(ValueError, "same-capacity"):
        args("--models", "baseline", "fifo", "--checkpoint", "/candidate")


def test_same_actor_off_and_unmodified_baseline_enforced():
    m = manifest()
    ev.validate_manifest_contract(m)
    broken = copy.deepcopy(m)
    broken["models"]["memory-off"]["memory_checkpoint"] = "/different_ae"
    with unittest.TestCase().assertRaisesRegex(ValueError, "SAME"):
        ev.validate_manifest_contract(rehash(broken))
    broken = copy.deepcopy(m)
    broken["models"]["baseline"]["memory_checkpoint"] = "/adapter"
    with unittest.TestCase().assertRaisesRegex(ValueError, "ORIGINAL"):
        ev.validate_manifest_contract(rehash(broken))


def test_server_dispatch_is_truthful():
    m = manifest()
    parsed = args("--models", "baseline")
    baseline = ev.server_command(parsed, m["models"]["baseline"], 40000)
    assert str(ev.REPO_ROOT / ev.BASELINE_SERVER) in baseline
    assert "--checkpoint" not in baseline and "--memory-off" not in baseline
    off = ev.server_command(parsed, m["models"]["memory-off"], 40000)
    assert str(ev.REPO_ROOT / ev.SERVER) in off
    assert "--memory-off" in off and off[off.index("--checkpoint") + 1] == "/candidate"


def reference_manifest():
    m = manifest()
    m["models"] = {"baseline": m["models"]["baseline"]}
    source = {p: "x" for p in ("gr00t/long_memory/online_policy_v7.py", "gr00t/long_memory/online_policy.py",
        "gr00t/eval/sim/robomme/run_long_memory_rollout.py", "gr00t/model/gr00t_n1d6/gr00t_n1d6.py",
        "run_scripts/robomme/eval_long_memory_comparison.py")}
    source[SERVERS["archive_projector_v10"][0]] = SERVERS["archive_projector_v10"][1]
    m["source_sha256"] = source
    for k in ("base_file_sha256", "policy_package_versions", "benchmark", "server_python", "robomme_python"):
        m[k] = {"identity": k}
    return rehash(m)


def test_fresh_v18_reference_cannot_spoof_or_change_baseline():
    ref = reference_manifest()
    assert equivalent_v18(ref, ref) == ev.BASELINE_SERVER
    changed = copy.deepcopy(ref)
    changed["settings"]["n_episodes"] = 10
    with unittest.TestCase().assertRaisesRegex(ValueError, "settings"):
        equivalent_v18(ref, changed)
    changed = copy.deepcopy(ref)
    changed["source_sha256"]["gr00t/long_memory/online_policy_v7.py"] = "changed"
    with unittest.TestCase().assertRaisesRegex(ValueError, "source closure"):
        equivalent_v18(ref, changed)
    changed = copy.deepcopy(ref)
    changed["models"]["baseline"]["server_script"] = ev.SERVER
    with unittest.TestCase().assertRaisesRegex(ValueError, "unchanged reviewed"):
        equivalent_v18(ref, changed)


def test_preflight_does_not_launch_or_create_output():
    with tempfile.TemporaryDirectory() as tmp, patch.object(ev, "check_dependencies"), \
            patch.object(ev, "build_identity", return_value=manifest()), \
            patch.object(ev, "run_evaluation", side_effect=AssertionError("preflight launched evaluation")):
        destination = Path(tmp) / "absent"
        assert ev.main(["--models", "baseline", "--preflight-only", "--output-dir", str(destination)]) == 0
        assert not destination.exists()


def test_missing_diagnostics_are_not_evidence_of_off():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "memory-off" / "BinFill"
        folder.mkdir(parents=True)
        (folder / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,6,1\n")
        result = ev.completed_read_diagnostics(Path(tmp), "memory-off", manifest())
        assert result["missing_sessions"] == 1 and not result["complete_evidence"]


def test_completed_off_diagnostics_reject_active_read_or_writer_mismatch():
    m = manifest()
    role = "memory-off"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / role / "BinFill"
        folder.mkdir(parents=True)
        (folder / "simulation_results.csv").write_text("episode_idx,episode_seed,success\n0,6,1\n")
        info = {"checkpoint_variant": ev.VARIANT, "checkpoint_step": 100,
            "representation": "adapted_short", "memory_off": True, "writer_sha256": None,
            "payload_sha256": m["models"][role]["training_metadata"]["payload_sha256"],
            "long_memory": {"policy": "fifo", "memory_read_enabled": False,
                "read": {"ae_conditioning_delta_norm": 0.0}}}
        call = {"kind": "policy_call", "session_id": "one", "episode_idx": 0, "episode_seed": 6,
            "passive": False, "info": info}
        end = {"kind": "episode_complete", "session_id": "one", "episode_idx": 0, "episode_seed": 6, "success": 1}

        def write():
            (folder / "memory_diagnostics.jsonl").write_text(json.dumps(call) + "\n" + json.dumps(end) + "\n")

        write()
        assert ev.completed_read_diagnostics(Path(tmp), role, m)["complete_evidence"]
        info["long_memory"]["memory_read_enabled"] = True
        write()
        with unittest.TestCase().assertRaisesRegex(ValueError, "READ-off actually"):
            ev.completed_read_diagnostics(Path(tmp), role, m)
        info["long_memory"]["memory_read_enabled"] = False
        info["writer_sha256"] = "not_same_writer"
        write()
        with unittest.TestCase().assertRaisesRegex(ValueError, "identity differs"):
            ev.completed_read_diagnostics(Path(tmp), role, m)


def test_online_off_preserves_adapted_short_and_captures_pre_hamlet_moment():
    from run_scripts.robomme.policy_representation_v18 import RepresentationPolicyV18

    class Processor:
        def __call__(self, records):
            return {"state": np.zeros(3, dtype=np.float32)}

        def decode_action(self, prediction, embodiment, states):
            return {"joint_position": prediction[:, :, :2]}

    class Head:
        action_horizon, action_dim = 2, 2
        _memory_cache = None

        def vlln(self, raw):
            return raw * 2

        def process_backbone_output(self, raw, action_inputs_B):
            self._memory_cache = torch.ones(1, 4, 8)
            return {"backbone_features": raw["backbone_features"] * 2 + 3}

        def state_encoder(self, state, embodiment):
            return state

        def get_action_with_features(self, features, *a):
            self.conditioning = features.clone()
            return {"action_pred": torch.zeros(1, 2, 2)}

        def reset_memory(self):
            self._memory_cache = None

    class Core:
        config = SimpleNamespace(representation="adapted_short")

        def step(self, short, moment, *a, read_enabled, **kw):
            self.moment = moment
            adapted = short + 5
            return {"fused": adapted + (2 if read_enabled else 0), "short": adapted,
                "metrics": {"write_rate": torch.tensor(1.)}, "bank": torch.zeros(1, 4, 8), "moment_history": None}

    p = RepresentationPolicyV18.__new__(RepresentationPolicyV18)
    p.sessions, p.session_cap = OrderedDict(), 4
    head = Head()
    p.model = SimpleNamespace(device=torch.device("cpu"), action_head=head,
        prepare_input=lambda batch: (None, SimpleNamespace(state=torch.zeros(1, 3), embodiment_id=torch.tensor([0]))),
        backbone=lambda batch: {"backbone_features": torch.ones(1, 6, 8)})
    p.processor, p.representation = Processor(), Core()
    p.n_q, p.stride, p.memory_off, p.stage, p.mode, p.write_policy = 2, 16, True, 1, ev.VARIANT, "fifo"
    p.payload_sha256, p.writer_sha256, p.writer_callback, p.checkpoint_step = {}, None, None, 100
    p.embodiment_tag = "dummy"
    p.modality_configs = {"state": SimpleNamespace(modality_keys=["joints"])}
    p._unbatch_observation = lambda obs: [obs]
    p._to_vla_step_data = lambda obs: SimpleNamespace(states={"joints": np.zeros(2, dtype=np.float32)})
    p.collate_fn = lambda processed: {"inputs": {}}
    _, info = p._get_action({}, {"session_ids": ["one"], "frame_index": 0, "episode_seed": 6})
    assert torch.equal(p.representation.moment, torch.full((1, 2, 8), 2.))
    assert torch.equal(head.conditioning[:, -2:], torch.full((1, 2, 8), 10.))
    assert info["long_memory"]["read"]["ae_conditioning_delta_norm"] == 0
    assert info["long_memory"]["read"]["short_adaptation_delta_norm"] > 0
    p.memory_off = False
    _, info = p._get_action({}, {"session_ids": ["one"], "frame_index": 16, "episode_seed": 6,
        "executed_actions": np.zeros((16, 8), dtype=np.float32)})
    assert info["long_memory"]["memory_read_enabled"] is True
    assert torch.equal(head.conditioning[:, -2:], torch.full((1, 2, 8), 12.))


def test_online_constructor_preserves_frozen_bf16_short_copy():
    from run_scripts.robomme import policy_representation_v18 as policy_module

    class Core(torch.nn.Module):
        def __init__(self, config, base_memory_transformer):
            super().__init__()
            self.config = config
            self.short_transformer = copy.deepcopy(base_memory_transformer)
            self.reader = torch.nn.Linear(8, 8, dtype=torch.float32)

    def base_init(self, *a, **kw):
        head = torch.nn.Module()
        head.memory_transformer = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
        self.model = SimpleNamespace(action_head=head, config=SimpleNamespace(memory_window=4))
        self.processor = SimpleNamespace(max_state_dim=3)
        self.n_q = 2

    info = {"step": 1, "metadata": {"payload_sha256": {}}, "config": {
        "expert": {"rank": 8, "alpha": 16.0}, "expert_targets": [],
        "representation": {"feature_dim": 8, "state_dim": 3, "num_short_tokens": 2,
            "representation": "adapted_short"}}}
    with patch.object(policy_module, "checkpoint_info_v18", return_value=info), \
            patch.object(policy_module.LongMemoryPolicy, "__init__", base_init), \
            patch.object(policy_module, "install_expert_lora"), \
            patch.object(policy_module, "set_expert_trainable"), \
            patch.object(policy_module, "load_checkpoint_v18"), \
            patch.object(policy_module, "RepresentationMemoryV18", Core):
        policy = policy_module.RepresentationPolicyV18("/base", "/checkpoint", device="cpu")
    assert policy.representation.short_transformer.weight.dtype == torch.bfloat16
    assert policy.representation.reader.weight.dtype == torch.float32


class RepresentationEvalTests(unittest.TestCase):
    """Register small pure checks with the repository's dependency-free runner."""


for _name, _function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        setattr(RepresentationEvalTests, _name, lambda self, check=_function: check())
del _name, _function


if __name__ == "__main__":
    unittest.main()
