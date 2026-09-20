"""CPU regressions for Expert adaptation, flow gradients and immutable bundles."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from gr00t.long_memory.checkpoint_v4 import (file_sha256, load_checkpoint_v4,
    reader_state_sha256, save_checkpoint_v4, v4_checkpoint_info)
from gr00t.long_memory.core_v3 import ActionValueMemory, MemoryV3Config
from gr00t.long_memory.expert_v4 import (ExpertLoRALinear, LoRAConfig,
    adapter_disabled, expert_episode_flow_loss, expert_flow_loss, expert_parameters,
    expert_state_dict, expert_state_sha256, install_expert_lora,
    load_expert_state_dict, set_expert_trainable)
from gr00t.long_memory.hamlet import checkpoint_identity, flow_loss


class _Projection(nn.Linear):
    def forward(self, x, *unused):
        return super().forward(x)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(.5)])

    def forward(self, x, features):
        value = F.scaled_dot_product_attention(self.to_q(x), self.to_k(features), self.to_v(features))
        return self.to_out[1](self.to_out[0](value))


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Module() for _ in range(2)])
        for block in self.transformer_blocks:
            block.attn1 = _Attention()

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        for block in self.transformer_blocks:
            hidden_states = hidden_states + block.attn1(hidden_states, encoder_hidden_states)
        return hidden_states, []


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Expert()
        self.state_encoder = _Projection(4, 8)
        self.action_encoder = _Projection(3, 8)
        self.action_decoder = _Projection(8, 3)
        self.config = SimpleNamespace(add_pos_embed=False, use_alternate_vl_dit=False)
        self.num_timestep_buckets = 10

    def sample_time(self, n, device, dtype):
        return torch.rand(n, device=device, dtype=dtype)


def flow_inputs(requires_grad=True):
    return dict(features=torch.randn(1, 6, 8, requires_grad=requires_grad),
        state=torch.randn(1, 1, 4), target=torch.randn(1, 2, 3),
        target_mask=torch.ones(1, 2, 3), attention_mask=torch.ones(1, 6, dtype=torch.bool),
        image_mask=torch.zeros(1, 6, dtype=torch.bool), embodiment_id=torch.zeros(1, dtype=torch.long),
        noise=torch.randn(1, 2, 3), time=torch.tensor([[[.3]]]))


class ExpertV4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(100)
        self.head = FakeHead().eval().requires_grad_(False)
        self.cfg = LoRAConfig(rank=2, alpha=4)

    def test_config_rejects_invalid_values(self):
        for options in ({"rank": 0}, {"rank": True}, {"rank": 1.5},
                        {"alpha": float("nan")}, {"alpha": 0}, {"alpha": True}):
            with self.assertRaises(ValueError):
                LoRAConfig(**options)

    def test_exact_sorted_targets_and_no_partial_or_double_install(self):
        targets = install_expert_lora(self.head, self.cfg)
        self.assertEqual(targets, sorted(targets))
        self.assertEqual(len(targets), 8)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_expert_lora(self.head, self.cfg)
        with self.assertRaisesRegex(ValueError, "target list"):
            install_expert_lora(FakeHead(), self.cfg, targets[:-1])

    def test_zero_adapter_matches_original_flow_loss_and_predictions_exactly(self):
        inputs = flow_inputs()
        original = flow_loss(self.head, **inputs)
        install_expert_lora(self.head, self.cfg)
        adapted = expert_flow_loss(self.head, **inputs)
        for name in ("loss", "prediction", "velocity_mae"):
            self.assertTrue(torch.equal(original[name], adapted[name]), name)
        # The legacy assertion remains strict; trainable adapters cannot leak in.
        with self.assertRaisesRegex(ValueError, "requires_grad"):
            flow_loss(self.head, **inputs)

    def test_joint_memory_and_expert_gradient_base_parameters_unchanged(self):
        original = {name: value.clone() for name, value in self.head.state_dict().items()}
        install_expert_lora(self.head, self.cfg)
        inputs = flow_inputs()
        optimizer = torch.optim.AdamW(expert_parameters(self.head), lr=.01)
        expert_flow_loss(self.head, **inputs)["loss"].backward()
        self.assertGreater(inputs["features"].grad.abs().sum().item(), 0)
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in expert_parameters(self.head)))
        adapter_ids = {id(parameter) for parameter in expert_parameters(self.head)}
        self.assertTrue(all(parameter.grad is None for parameter in self.head.parameters() if id(parameter) not in adapter_ids))
        optimizer.step()
        for name, value in original.items():
            new_name = name
            for target, module in self.head.named_modules():
                if isinstance(module, ExpertLoRALinear) and name.startswith(target + "."):
                    new_name = target + ".base." + name[len(target) + 1:]
                    break
            self.assertTrue(torch.equal(value, self.head.state_dict()[new_name]), name)

    def test_activation_checkpoint_with_detached_inputs_still_trains_expert(self):
        install_expert_lora(self.head, self.cfg)
        inputs = flow_inputs(False)
        from torch.utils.checkpoint import checkpoint
        with patch("torch.utils.checkpoint.checkpoint", wraps=checkpoint) as run:
            result = expert_flow_loss(self.head, **inputs, activation_checkpointing=True)
            result["loss"].backward()
        run.assert_called_once()
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in expert_parameters(self.head)))

    def test_frozen_adapters_keep_input_gradient_and_no_parameter_gradient(self):
        install_expert_lora(self.head, self.cfg)
        set_expert_trainable(self.head, False)
        inputs = flow_inputs()
        expert_flow_loss(self.head, **inputs, activation_checkpointing=True)["loss"].backward()
        self.assertGreater(inputs["features"].grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.head.parameters()))
        set_expert_trainable(self.head, True)
        self.assertEqual(sum(p.requires_grad for p in self.head.parameters()), 16)

    def test_disable_restores_baseline_and_nested_context_on_exception(self):
        original = copy.deepcopy(self.head)
        install_expert_lora(self.head, self.cfg)
        with torch.no_grad():
            for name, parameter in self.head.named_parameters():
                if name.endswith("lora_B"):
                    parameter.fill_(.1)
        inputs = flow_inputs()
        base = flow_loss(original, **inputs)["prediction"]
        changed = expert_flow_loss(self.head, **inputs)["prediction"]
        self.assertFalse(torch.equal(base, changed))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with adapter_disabled(self.head):
                with adapter_disabled(self.head):
                    self.assertTrue(torch.equal(base, expert_flow_loss(self.head, **inputs)["prediction"]))
                raise RuntimeError("intentional")
        self.assertTrue(torch.equal(changed, expert_flow_loss(self.head, **inputs)["prediction"]))

    def test_fp32_adapters_bf16_base_and_episode_bridge(self):
        self.head.to(dtype=torch.bfloat16)
        install_expert_lora(self.head, self.cfg)
        self.assertTrue(all(p.dtype == torch.float32 for p in expert_parameters(self.head)))
        episode = {"features": [torch.randn(6, 8)], "state": torch.randn(1, 4),
            "targets": torch.randn(1, 2, 3), "target_mask": torch.ones(1, 2, 3),
            "attention_masks": [torch.ones(6)], "image_masks": [torch.zeros(6)], "embodiment_id": 0}
        fused = torch.randn(1, 2, 8, requires_grad=True)
        result = expert_episode_flow_loss(self.head, episode, 0, fused, seed=5, activation_checkpointing=True)
        self.assertEqual(result["prediction"].dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertGreater(fused.grad.abs().sum().item(), 0)

    def test_adapter_load_is_strict_atomic_finite_and_hash_sensitive(self):
        install_expert_lora(self.head, self.cfg)
        state = expert_state_dict(self.head)
        original_hash = expert_state_sha256(self.head)
        key = next(iter(state))
        incomplete = dict(state)
        incomplete.pop(key)
        with self.assertRaisesRegex(ValueError, "tensor names"):
            load_expert_state_dict(self.head, incomplete)
        state[key].flatten()[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            load_expert_state_dict(self.head, state)
        self.assertEqual(original_hash, expert_state_sha256(self.head))
        state[key].flatten()[0] = 100
        load_expert_state_dict(self.head, state)
        self.assertNotEqual(original_hash, expert_state_sha256(self.head))

    def test_thawed_base_or_training_mode_rejected(self):
        install_expert_lora(self.head, self.cfg)
        self.head.train()
        with self.assertRaisesRegex(ValueError, "only LoRA"):
            expert_flow_loss(self.head, **flow_inputs())
        self.head.eval()
        self.head.action_decoder.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "only LoRA"):
            expert_flow_loss(self.head, **flow_inputs())


class BundleV4Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(100)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.head = FakeHead().eval().requires_grad_(False)
        base_state = {"action_head." + key: tensor for key, tensor in self.head.state_dict().items()}
        save_file(base_state, str(self.base / "model.safetensors"))
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: "model.safetensors" for key in base_state}}))
        (self.base / "config.json").write_text(json.dumps({"hamlet_mode": "finetune", "mem_cond_type": "cross_attn",
            "memory_type": "moment_token", "n_moment_tokens": 4, "memory_stride": 16, "backbone_embedding_dim": 8}))
        (self.base / "processor_config.json").write_text(json.dumps({"processor_kwargs": {"max_state_dim": 4, "max_action_dim": 3}}))
        for name in ("statistics.json", "embodiment_id.json"):
            (self.base / name).write_text("{}")
        self.cfg = MemoryV3Config(feature_dim=8, state_dim=4, action_dim=3, hidden_dim=8,
                                  num_heads=2, capacity=3, min_fill=1)
        self.memory = ActionValueMemory(self.cfg)
        self.lora = LoRAConfig(rank=2, alpha=4)
        targets = install_expert_lora(self.head, self.lora)
        self.config = {"trainer_variant": "action_expert_v4", "stage": 1, "memory": asdict(self.cfg),
                       "expert": asdict(self.lora), "expert_targets": targets, "train": {"reader_mode": "memory"}}
        self.metadata = {"base_model": checkpoint_identity(self.base), "cache_fingerprint": "cache-A"}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name="run", step=1, optimizer=None):
        return save_checkpoint_v4(self.root / name, step, self.memory, self.head, optimizer, self.config, self.metadata)

    def test_bundle_roundtrip_immutable_and_readonly_preserves_rng(self):
        original = {p.name: p.read_bytes() for p in self.base.iterdir()}
        path = self.save()
        self.assertEqual({p.name for p in path.iterdir()}, {"model.safetensors", "expert.safetensors", "checkpoint.json", "training_state.pt"})
        before_rng = torch.get_rng_state().clone()
        info = v4_checkpoint_info(self.base, path, expected_stage=1)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(info["metadata"]["expert_sha256"], file_sha256(path / "expert.safetensors"))
        with self.assertRaises(FileExistsError):
            self.save()
        with torch.no_grad():
            next(self.memory.parameters()).add_(10)
            next(expert_parameters(self.head)).add_(10)
        load_checkpoint_v4(path, self.memory, self.head)
        for key, value in expert_state_dict(self.head).items():
            self.assertTrue(torch.equal(value, load_file(str(path / "expert.safetensors"))[key]))
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.base.iterdir()})

    def test_optimizer_and_rng_resume(self):
        optimizer = torch.optim.AdamW(expert_parameters(self.head), lr=.01)
        expert_flow_loss(self.head, **flow_inputs(False))["loss"].backward()
        optimizer.step()
        path = self.save(optimizer=optimizer)
        expected = torch.randn(5)
        optimizer.param_groups[0]["lr"] = .5
        torch.manual_seed(99)
        load_checkpoint_v4(path, self.memory, self.head, optimizer)
        self.assertTrue(torch.equal(expected, torch.randn(5)))
        self.assertEqual(optimizer.param_groups[0]["lr"], .01)

    def test_missing_adapter_nonfinite_and_wrong_targets_rejected(self):
        path = self.save()
        state = load_file(str(path / "expert.safetensors"))
        first = next(iter(state))
        broken = dict(state)
        broken.pop(first)
        save_file(broken, str(path / "expert.safetensors"))
        with self.assertRaisesRegex(ValueError, "tensor names"):
            v4_checkpoint_info(self.base, path)
        state[first].flatten()[0] = float("nan")
        save_file(state, str(path / "expert.safetensors"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            v4_checkpoint_info(self.base, path)

    def test_hash_and_wrong_alpha_rejected(self):
        path = self.save()
        info = json.loads((path / "checkpoint.json").read_text())
        info["metadata"]["expert_sha256"] = "0" * 64
        (path / "checkpoint.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "expert_sha256"):
            v4_checkpoint_info(self.base, path)
        self.config["expert"]["alpha"] = 8
        with self.assertRaisesRegex(ValueError, "rank/alpha"):
            self.save("wrong-alpha")

    def test_stage2_frozen_fingerprints_and_selfcontained_parent(self):
        self.config["stage"] = 2
        self.metadata.update(stage1_parent={"path": "/parent/no/longer/mounted", "checkpoint_sha256": "1" * 64,
            "memory_sha256": "2" * 64, "expert_sha256": "3" * 64},
            frozen_reader_sha256=reader_state_sha256(self.memory), frozen_expert_sha256=expert_state_sha256(self.head))
        path = self.save()
        self.assertEqual(v4_checkpoint_info(self.base, path)["config"]["stage"], 2)
        with torch.no_grad():
            next(self.memory.writer_parameters()).add_(1)
        self.save("writer-changed")  # Only writer changes are permitted.
        with torch.no_grad():
            next(self.memory.reader_parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, "frozen reader"):
            self.save("reader-changed")

    def test_no_memory_control_and_stage2_rejection(self):
        self.config["train"]["reader_mode"] = "none"
        path = self.save()
        self.assertEqual(v4_checkpoint_info(self.base, path)["config"]["train"]["reader_mode"], "none")
        info = json.loads((path / "checkpoint.json").read_text())
        info["config"]["stage"] = 2
        (path / "checkpoint.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "Stage 2 requires memory"):
            v4_checkpoint_info(self.base, path)

    def test_conflicting_reader_mode_is_rejected(self):
        self.config["reader_mode"] = "none"
        path = self.save()
        with self.assertRaisesRegex(ValueError, "conflicting reader_mode"):
            v4_checkpoint_info(self.base, path)

    def test_wrong_base_missing_cache_and_legacy_rejected(self):
        path = self.save()
        original = (path / "checkpoint.json").read_text()
        for mutate, message in ((lambda i: i["config"].update(trainer_variant="action_value_v3"), "legacy"),
                (lambda i: i["metadata"].update(cache_fingerprint=""), "cache_fingerprint"),
                (lambda i: i["metadata"].update(base_model={}), "different/changed"),
                (lambda i: i["config"]["expert_targets"].pop(), "expert_targets")):
            info = json.loads(original)
            mutate(info)
            (path / "checkpoint.json").write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, message):
                v4_checkpoint_info(self.base, path)


if __name__ == "__main__":
    unittest.main()
