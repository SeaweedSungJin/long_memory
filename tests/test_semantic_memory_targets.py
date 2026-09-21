import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gr00t.long_memory.cache import decision_frames
from run_scripts.robomme.semantic_memory_targets import (
    ANCHOR_NAME, CONFIG, KIND, SemanticTargets, compositional_labels, digest,
    episode_rows, fit_vocabulary, role_anchor, selected_demo_anchor,
)


def fixture_episode():
    n, demo = 160, 64
    raw = {"frame_index": list(range(n)), "episode_index": [7] * n,
           "is_demo": [True] * demo + [False] * (n - demo)}
    labels = ["pick up the cube"] * 16 + ["drop the cube onto target"] * 32 + ["static"] * 16
    labels += ["pick up the cube"] * 16 + ["place the cube onto the correct target"] * 80
    grounded = ["pick up the cube at <12, 14>"] * 16 + ["drop the cube onto target at <80, 120>"] * 32 + ["static"] * 16
    grounded += ["pick up the cube at <20, 21>"] * 16 + ["place the cube onto the correct target at <82, 123>"] * 80
    raw.update(simple_subgoal=labels, simple_subgoal_online=labels[:],
               grounded_subgoal=grounded, grounded_subgoal_online=grounded[:])
    frames = decision_frames(np.asarray(raw["is_demo"]), 16).tolist()
    layout = {"frames": frames, "is_demo": [raw["is_demo"][f] for f in frames],
              "decision_mask": [not raw["is_demo"][f] for f in frames[:-1]]}
    goal = "watch the video carefully, then place the blue cube on the first target it was previously placed on"
    return raw, layout, goal


def test_compositional_labels_no_generic_or_first_coordinate_labels():
    assert compositional_labels("move forward-right") == {"direction": "forward-right"}
    assert compositional_labels("pick up the red cube for the second time") == {"ordinal": "second"}
    assert compositional_labels("pick up the peg by grasping the far end") == {"grasp_end": "far"}
    assert compositional_labels("insert the peg from the right side of the box") == {"insert_side": "right"}
    for text in ["no record", "static", "remain static", "press the button", "pick up the cube"]:
        assert compositional_labels(text) == {}
    assert role_anchor("pick up the cube at <10, 20> then drop it at <30, 40>") is None
    assert role_anchor("drop the cube onto target at <80, 120>") == [80, 120]
    assert role_anchor("drop the cube onto target at <280, 120>") is None


def test_past_anchor_is_past_onset_not_current_tracking():
    raw, layout, goal = fixture_episode()
    answer, reason = selected_demo_anchor(goal, raw, layout["frames"])
    assert reason is None and answer["point_yx"] == [80, 120]
    assert answer["source_interval"] == [16, 48]
    rows, excluded = episode_rows(raw, layout, 7, "train", goal)
    supervised = [r for r in rows if r["regression"]]
    assert supervised
    for row in supervised:
        assert row["regression"][ANCHOR_NAME] == [80 / 255, 120 / 255]
        assert max(row["target_provenance"][ANCHOR_NAME]["observed_source_frames"]) < row["query_frame"]
    # The current anchor [82,123] was only a consistency check, never the answer.
    assert all(r["regression"][ANCHOR_NAME] != [82 / 255, 123 / 255] for r in supervised)


def test_ambiguous_anchor_and_alignment_fail_closed():
    raw, layout, goal = fixture_episode()
    raw["grounded_subgoal"][20] = "drop the cube onto target at <79, 120>"
    assert selected_demo_anchor(goal, raw, layout["frames"])[1] == "anchor_past_grounding_missing_or_changes_within_run"
    raw, layout, goal = fixture_episode()
    raw["grounded_subgoal"][96] = "place the cube onto the correct target at <10, 10>"
    assert selected_demo_anchor(goal, raw, layout["frames"])[0] is None
    raw, layout, goal = fixture_episode()
    layout["frames"][1] += 1
    with unittest.TestCase().assertRaisesRegex(ValueError, "cadence"):
        episode_rows(raw, layout, 7, "train", goal)


def test_button_relation_uses_correct_neighbor_not_first_target():
    raw, layout, _ = fixture_episode()
    # Put a button in the demo between the first placement and the execution.
    for k in ("simple_subgoal", "simple_subgoal_online"):
        raw[k][48:64] = ["press the button"] * 16
    for k in ("grounded_subgoal", "grounded_subgoal_online"):
        raw[k][48:64] = ["press the button at <10, 20>"] * 16
    before = "watch the video carefully, then place the red cube on the target right before the button was pressed"
    after = before.replace("before", "after")
    assert selected_demo_anchor(before, raw, layout["frames"])[0]["point_yx"] == [80, 120]
    assert selected_demo_anchor(after, raw, layout["frames"])[1] == "anchor_no_adjacent_placement"


def make_row(eid, split, label, anchor=None):
    return {"episode_id": eid, "decision": 1, "query_frame": 16, "split": split,
            "classification_labels": {"direction": label, "constant": "same"},
            "classification": {}, "regression": {} if anchor is None else {ANCHOR_NAME: anchor}}


def test_train_only_vocabulary_and_masked_val_unknown():
    rows = [make_row(1, "train", "left", [.1, .2]), make_row(2, "train", "right", [.3, .4]),
            make_row(3, "val", "forward", [.8, .9])]
    vocabulary, regression, disabled = fit_vocabulary(rows)
    assert vocabulary == {"direction": ["left", "right"]}
    assert rows[2]["classification"] == {}
    assert disabled == {"constant": "constant_or_absent_in_train"}
    assert regression == {ANCHOR_NAME: 2}


def check_target_reader(tmp_path):
    rows = [make_row(1, "train", "left"), make_row(2, "train", "right"), make_row(3, "val", "right")]
    vocab, regression, _ = fit_vocabulary(rows)
    cache = {"fingerprint": "cache-fixture", "splits": {"train": [1, 2], "val": [3]}}
    manifest = {"kind": KIND, "status": "complete", "config": CONFIG, "cache_fingerprint": cache["fingerprint"],
                "splits": cache["splits"], "vocabulary": vocab, "regression_sizes": regression, "rows": rows}
    manifest["fingerprint"] = digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    target = SemanticTargets(path, cache)
    assert target.get(1, 1)["classification"] == {"direction": 0}
    assert target.get(99, 1) is None
    assert target.answer_config()["classification_sizes"] == {"direction": 2}
    wrong = copy.deepcopy(cache)
    wrong["splits"]["val"] = [9]
    with unittest.TestCase().assertRaisesRegex(ValueError, "identity/splits"):
        SemanticTargets(path, wrong)
    manifest["rows"][0]["classification"]["direction"] = 999
    path.write_text(json.dumps(manifest))
    with unittest.TestCase().assertRaisesRegex(ValueError, "fingerprint"):
        SemanticTargets(path, cache)


def test_target_reader_checks_fingerprint_splits_and_values():
    with tempfile.TemporaryDirectory() as directory:
        check_target_reader(Path(directory))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in globals().items() if name.startswith("test_"))


if __name__ == "__main__":
    unittest.main()
