import unittest
import torch

from run_scripts.robomme.semantic_memory_answers import SemanticAnswerHead


def model():
    return SemanticAnswerHead({"input_dim": 8, "hidden_dim": 12,
        "classification_sizes": {"direction": 3}, "regression_sizes": {"past_target_anchor_yx": 2}})


def test_only_actual_read_input_and_backward():
    torch.manual_seed(7)
    head = model()
    read = torch.randn(2, 4, 8, requires_grad=True)
    target = [{"classification": {"direction": 2}, "regression": {}},
              {"classification": {}, "regression": {"past_target_anchor_yx": [.2, .8]}}]
    loss, metrics = head.loss(read, target)
    assert torch.isfinite(loss) and metrics["answer_active_heads"] == 2
    assert metrics["answer_direction_count"] == 1
    assert metrics["answer_past_target_anchor_yx_count"] == 1
    loss.backward()
    assert read.grad is not None and read.grad.abs().sum() > 0
    assert head.classifiers["direction"].weight.grad is not None
    assert head.regressors["past_target_anchor_yx"].weight.grad is not None


def test_no_targets_are_differentiable_zero_not_fake_class():
    head = model()
    read = torch.randn(2, 4, 8, requires_grad=True)
    loss, metrics = head.loss(read, None)
    assert loss.item() == 0 and metrics["answer_active_heads"] == 0
    loss.backward()
    assert torch.equal(read.grad, torch.zeros_like(read))


def test_serializable_config_and_weight_roundtrip():
    head = model()
    replica = SemanticAnswerHead(head.config_dict())
    replica.load_state_dict(head.state_dict())
    read = torch.randn(1, 4, 8)
    assert torch.equal(head(read)["classification"]["direction"], replica(read)["classification"]["direction"])


def test_invalid_shapes_and_labels_rejected():
    head = model()
    with unittest.TestCase().assertRaisesRegex(ValueError, "Expected actual"):
        head(torch.randn(1, 8))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Invalid class"):
        head.loss(torch.randn(1, 4, 8), {"classification": {"direction": -1}})
    with unittest.TestCase().assertRaisesRegex(ValueError, "normalized"):
        head.loss(torch.randn(1, 4, 8), {"regression": {"past_target_anchor_yx": [float("nan"), .5]}})


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in globals().items() if name.startswith("test_"))


if __name__ == "__main__":
    unittest.main()
