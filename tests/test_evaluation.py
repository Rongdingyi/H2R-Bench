import copy
import json
from pathlib import Path

import pytest
from evaluation import contact, embodiment, goal_action, run
from evaluation.aggregate import aggregate, core
from evaluation.common import ROOT, read_jsonl
from evaluation.prepare import prepare


def example(tmp_path):
    records = prepare(tmp_path, tmp_path, "test-model", ROOT / "benchmark/annotations.jsonl", True)
    return next(r for r in records if run.frame_counts(r)["M1_M2"] == 24)


def goal_response(record):
    a = record["annotation"]
    return {"goal_checks": [dict(id=g["id"], score_0_to_4=4, reason="visible") for g in a["goal_checks"]],
            "action_events": [dict(id=e["id"], score_0_to_4=4, first_frame=1, reason="visible") for e in a["action_events"]]}


def contact_response():
    return {"source_grounding_hard_failure": {"triggered": False},
        "h2r_affordance_contact_transfer": {d: dict(score_0_to_4=4, reason="visible", evidence_source_frames=[1], evidence_generated_frames=[1]) for d in contact.DIMENSIONS}}


def embodiment_response(target):
    return {"embodiment_diagnosis": {"observed_end_effector_type": "rigid_two_jaw_gripper" if target == "mechanical_arm" else "robotic_multi_finger_dexterous_hand"},
        "hard_failure": False,
        "embodiment_correctness": {d: dict(score_0_to_4=4, reason="visible", evidence_generated_frames=[1]) for d in embodiment.DIMENSIONS}}


def test_completion_not_order_and_goal_weights():
    a = {"goal_checks": [{"id": "g1", "weight": 2}, {"id": "g2", "weight": 1}],
         "action_events": [{"id": "e1", "weight": 1}, {"id": "e2", "weight": 1}], "action_order": [["e1", "e2"]]}
    out = {"goal_checks": [{"id": "g1", "score_0_to_4": 4}, {"id": "g2", "score_0_to_4": 0}],
           "action_events": [{"id": "e1", "score_0_to_4": 4, "first_frame": 8}, {"id": "e2", "score_0_to_4": 4, "first_frame": 1}]}
    assert goal_action.compute_goal_score(a, out)["score"] == pytest.approx(2 / 3)
    result = goal_action.compute_action_score(a, out)
    assert result["event_completion_score"] == 1
    assert result["order_score"] == 0


def test_contact_gate_and_embodiment_cap(tmp_path):
    record = example(tmp_path)
    obj = contact_response()
    assert run.validate_and_score("M3", obj, record)[1]["M3"] == 1
    obj["source_grounding_hard_failure"]["triggered"] = True
    assert run.validate_and_score("M3", obj, record)[1]["M3"] == 0
    record["target_embodiment"] = "mechanical_arm"
    obj = embodiment_response("dexterous_hand")
    assert run.validate_and_score("M4", obj, record)[1]["M4"] == pytest.approx(.875)
    obj["hard_failure"] = True
    assert run.validate_and_score("M4", obj, record)[1]["M4"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "frame", "bool_score"])
def test_invalid_checklists_are_not_scored_as_zero(tmp_path, fault):
    record = example(tmp_path)
    obj = goal_response(record)
    if fault == "missing": obj["goal_checks"].pop()
    if fault == "duplicate": obj["goal_checks"].append(obj["goal_checks"][0])
    if fault == "extra": obj["goal_checks"].append(dict(id="unexpected", score_0_to_4=4))
    if fault == "frame": obj["action_events"][0]["first_frame"] = 25
    if fault == "bool_score": obj["goal_checks"][0]["score_0_to_4"] = True
    with pytest.raises(ValueError): run.validate_and_score("M1_M2", obj, record)


def test_aggregation_missing_m5_and_exact_video_join():
    r = dict(id="c", model="m", target_embodiment="mechanical_arm", generated_video_sha256="a", M1=1., M2=1., M3=1., M4=1.)
    out, summary = aggregate([r], [])
    assert out[0]["H2RCore"] is None and summary[0]["n_M5"] == 0
    out, summary = aggregate([r], [dict(id="c", m5=.5, video_sha256="a")])
    assert out[0]["H2RCore"] == pytest.approx(95)
    with pytest.raises(ValueError): aggregate([r], [dict(id="c", m5=.5, video_sha256="different")])
    assert core(dict(r, M5=1)) == pytest.approx(100)


def test_video_sampling_cache_and_provider_changes(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6, (32, 32))
    assert writer.isOpened()
    for i in range(6): writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
    writer.release()
    indices, frames = run.sample_frames(video, 24)
    assert len(frames) == 24 and indices[0] == 0 and indices[-1] == 5
    record = example(tmp_path)
    record.update(source_video=str(video), generated_video=str(video))
    providers = [run.provider_config("qwen")]
    calls = []

    def fake(config, prompt, sequences, timeout):
        calls.append(prompt)
        if "M3:" in prompt:
            assert len(sequences) == 2 and all(len(images) == 24 for _, images in sequences)
            return json.dumps(contact_response())
        if prompt.startswith("Evaluate only Embodiment"):
            assert len(sequences[0][1]) == 12
            return json.dumps(embodiment_response(record["target_embodiment"]))
        assert len(sequences[0][1]) == 24
        return json.dumps(goal_response(record))

    monkeypatch.setattr(run, "call_judge", fake)
    result = run.judge_record(record, providers, tmp_path / "run")
    assert result["M1"] == result["M2"] == result["M3"] == result["M4"] == 1
    run.judge_record(record, providers, tmp_path / "run")
    assert len(calls) == 3
    providers[0]["model"] += "-new-version"
    run.judge_record(record, providers, tmp_path / "run")
    assert len(calls) == 6
    record["annotation"]["goal_checks"][0]["description"] += " revised"
    run.judge_record(record, providers, tmp_path / "run")
    assert len(calls) == 9


def test_full_offline_manifest_and_annotation_frame_indices(tmp_path):
    records = prepare(tmp_path, tmp_path, "example", ROOT / "benchmark/annotations.jsonl", True)
    assert len(records) == 240
    for record in records:
        prompts = run.make_prompts(record)
        assert "source_evidence_frames" not in prompts["M3"]
        expected_window = "frames 19--24" if run.frame_counts(record)["M1_M2"] == 24 else "frames 25--32"
        assert expected_window in prompts["M1_M2"]
        assert run.validate_and_score("M1_M2", goal_response(record), record)[1] == {"M1": 1., "M2": 1.}


def test_main_annotations_choose_historical_frame_windows(tmp_path):
    records = prepare(tmp_path, tmp_path, "example", ROOT / "benchmark/annotations.jsonl", True)
    assert sum(run.frame_counts(r)["M1_M2"] == 32 for r in records) == 216
    assert sum(run.frame_counts(r)["M1_M2"] == 24 for r in records) == 24
    original = next(r for r in records if run.frame_counts(r)["M1_M2"] == 32)
    prompt = run.make_prompts(original)["M1_M2"]
    assert "frames 25--32" in prompt and "{case_annotation_json}" not in prompt
    response = goal_response(original)
    response["action_events"][0]["first_frame"] = 32
    assert run.validate_and_score("M1_M2", response, original)[1]["M2"] == 1
