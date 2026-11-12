import copy
import json
import re
from pathlib import Path

import pytest
from evaluation import contact, embodiment, goal_action, run
from evaluation.aggregate import aggregate, core
from evaluation.common import COVERAGE_METRICS, ROOT, benchmark_family, read_jsonl
from evaluation.prepare import prepare
from evaluation.video_quality import clip_quality_components, quality_mean


def score_record(**values):
    for key in COVERAGE_METRICS:
        values.setdefault(key, 1.)
    values.update(protocol=run.PROTOCOL, frame_counts=run.FRAME_COUNTS.copy(),
                  judge_models=run.DEFAULT_MODELS.copy(), temperature=0)
    values["providers"] = {name: {m: values[m] for m in ("M1", "M2", "M3", "M4", *COVERAGE_METRICS)}
                           for name in run.DEFAULT_MODELS}
    return values


def quality_record(**values):
    return dict(run_signature="test-m5-config", **values)


def example(tmp_path):
    records = prepare(tmp_path, tmp_path, "test-model", ROOT / "benchmark/annotations.jsonl", True)
    return records[0]


def goal_response(record):
    a = goal_action.normalize_annotation(record["annotation"])
    return {"goal_checks": [dict(id=g["id"], score_0_to_4=4, evidence_generated_frames=[25], reason="visible") for g in a["goal_checks"]],
            "action_events": [dict(id=e["id"], score_0_to_4=4, first_frame=1, evidence_generated_frames=[1], reason="visible") for e in a["action_events"]]}


def contact_response():
    return {"source_grounding_hard_failure": {"triggered": False, "reason": "grounding decision", "evidence_generated_frames": [1]},
        "h2r_affordance_contact_transfer": {d: dict(score_0_to_4=4, reason="visible", evidence_source_frames=[1], evidence_generated_frames=[1]) for d in contact.DIMENSIONS}}


def embodiment_response(target):
    return {"embodiment_diagnosis": {"observed_end_effector_type": "rigid_two_jaw_gripper" if target == "mechanical_arm" else "robotic_multi_finger_dexterous_hand"},
        "hard_failure": False, "hard_failure_reason": "manipulator decision",
        "embodiment_correctness": {d: dict(score_0_to_4=4, reason="visible", evidence_generated_frames=[1]) for d in embodiment.DIMENSIONS}}


def test_completion_not_order_and_goal_weights():
    a = {"goal_checks": [{"id": "g1", "weight": 2}, {"id": "g2", "weight": 1}],
         "action_events": [{"id": "e1", "weight": 1}, {"id": "e2", "weight": 1}], "action_order": [["e1", "e2"]]}
    out = {"goal_checks": [{"id": "g1", "score_0_to_4": 4}, {"id": "g2", "score_0_to_4": 0}],
           "action_events": [{"id": "e1", "score_0_to_4": 4, "first_frame": 8}, {"id": "e2", "score_0_to_4": 4, "first_frame": 1}]}
    assert goal_action.compute_goal_score(a, out)["score"] == pytest.approx(2 / 3)
    result = goal_action.compute_action_score(a, out)
    assert result["event_completion_score"] == 1
    assert result["score"] == 1
    assert "order_score" not in result


def test_contact_gate_and_embodiment_formula(tmp_path):
    record = example(tmp_path)
    obj = contact_response()
    assert run.validate_and_score("M3", obj, record)[1]["M3"] == 1
    obj["source_grounding_hard_failure"]["triggered"] = True
    assert run.validate_and_score("M3", obj, record)[1]["M3"] == 0
    record["target_embodiment"] = "mechanical_arm"
    obj = embodiment_response("dexterous_hand")
    assert run.validate_and_score("M4", obj, record)[1]["M4"] == pytest.approx(1.0)
    obj["hard_failure"] = True
    assert run.validate_and_score("M4", obj, record)[1]["M4"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "frame", "bool_score"])
def test_invalid_checklists_are_not_scored_as_zero(tmp_path, fault):
    record = example(tmp_path)
    obj = goal_response(record)
    if fault == "missing": obj["goal_checks"].pop()
    if fault == "duplicate": obj["goal_checks"].append(obj["goal_checks"][0])
    if fault == "extra": obj["goal_checks"].append(dict(id="unexpected", score_0_to_4=4))
    if fault == "frame": obj["action_events"][0]["first_frame"] = 26
    if fault == "bool_score": obj["goal_checks"][0]["score_0_to_4"] = True
    with pytest.raises(ValueError): run.validate_and_score("M1_M2", obj, record)


def test_aggregation_missing_m5_and_exact_video_join():
    r = score_record(id="c", model="m", target_embodiment="mechanical_arm", generated_video_sha256="a", M1=1., M2=1., M3=1., M4=1.)
    out, summary = aggregate([r], [])
    assert out[0]["H2RCore"] is None and summary[0]["n_M5"] == 0
    out, summary = aggregate([r], [quality_record(id="c", m5=.5, video_sha256="a")])
    assert out[0]["H2RCore"] == pytest.approx(95)
    with pytest.raises(ValueError): aggregate([r], [quality_record(id="c", m5=.5, video_sha256="different")])
    assert core(dict(r, M5=1)) == pytest.approx(100)


def test_video_sampling_cache_and_provider_changes(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6, (32, 32))
    assert writer.isOpened()
    for i in range(49): writer.write(np.full((32, 32, 3), i * 5, dtype=np.uint8))
    writer.release()
    indices, frames = run.sample_frames(video, 25)
    assert len(frames) == 25 and indices == list(range(0, 49, 2))
    record = example(tmp_path)
    record.update(source_video=str(video), generated_video=str(video))
    providers = [run.provider_config("qwen")]
    calls = []

    def fake(config, prompt, sequences, timeout):
        calls.append(prompt)
        if "M3:" in prompt:
            assert len(sequences) == 2 and all(len(images) == 25 for _, images in sequences)
            return json.dumps(contact_response())
        if prompt.startswith("Evaluate only Embodiment"):
            assert len(sequences[0][1]) == 25
            return json.dumps(embodiment_response(record["target_embodiment"]))
        assert len(sequences[0][1]) == 25
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
    protocol = run.PROTOCOL
    monkeypatch.setattr(run, "PROTOCOL", "release-mixed-spec-v1")
    run.judge_record(record, providers, tmp_path / "run")
    assert len(calls) == 12
    monkeypatch.setattr(run, "PROTOCOL", protocol)
    run.judge_record(record, providers, tmp_path / "run")
    assert len(calls) == 15
    cache = tmp_path / "run/judgments" / record["id"] / "qwen/M3.json"
    saved = json.loads(cache.read_text())
    assert saved["protocol"] == protocol
    assert saved["sampled_frame_indices"] == {
        "source_video": indices, "generated_video": indices}


def test_full_offline_manifest_and_annotation_frame_indices(tmp_path):
    records = prepare(tmp_path, tmp_path, "example", ROOT / "benchmark/annotations.jsonl", True)
    assert len(records) == 240
    for record in records:
        prompts = run.make_prompts(record)
        assert "source_evidence_frames" not in prompts["M3"]
        assert "approx_source_frames" not in prompts["M3"]
        payload = contact.build_case_payload(record["contact"], record)
        assert not re.search(r"\bframes?\s+\d", json.dumps(payload), re.I)
        assert "25 uniformly sampled frames" in prompts["M1_M2"]
        assert "source_25_frames" in prompts["M3"] and "generated_25_frames" in prompts["M3"]
        assert "25 uniformly sampled generated frames" in prompts["M4"]
        assert "end of the sequence" in prompts["M1_M2"]
        assert all(value == 1. for value in run.validate_and_score("M1_M2", goal_response(record), record)[1].values())


def test_all_annotations_use_the_same_evaluation_budget(tmp_path):
    records = prepare(tmp_path, tmp_path, "example", ROOT / "benchmark/annotations.jsonl", True)
    assert {r["annotation"]["annotation_version"] for r in records} == {
        "original-source32", "corrected-source32-20260721"}
    for record in records:
        assert run.frame_counts(record) == {"M1_M2": 25, "M3": 25, "M4": 25}


@pytest.mark.parametrize("frame", [1, 25, 0, 26, 32])
@pytest.mark.parametrize("field", ["M1_M2", "M3_source", "M3_generated", "M3_gate", "M4"])
def test_evidence_frame_boundaries(tmp_path, frame, field):
    record = example(tmp_path)
    if field == "M1_M2":
        metric, obj = "M1_M2", goal_response(record)
        obj["action_events"][0]["first_frame"] = frame
    elif field.startswith("M3"):
        metric, obj = "M3", contact_response()
        if field == "M3_gate":
            obj["source_grounding_hard_failure"]["evidence_generated_frames"] = [frame]
        else:
            key = "evidence_source_frames" if field == "M3_source" else "evidence_generated_frames"
            obj["h2r_affordance_contact_transfer"][contact.DIMENSIONS[0]][key] = [frame]
    else:
        metric, obj = "M4", embodiment_response(record["target_embodiment"])
        obj["embodiment_correctness"][embodiment.DIMENSIONS[0]]["evidence_generated_frames"] = [frame]
    if 1 <= frame <= 25:
        run.validate_and_score(metric, obj, record)
    else:
        with pytest.raises(ValueError):
            run.validate_and_score(metric, obj, record)


def test_contact_sanitization_preserves_semantics_and_original_annotation(tmp_path):
    record = example(tmp_path)
    spec = record["contact"]["contact_spec"]
    spec["notes_for_m3_judge"] = (
        "Hold the glasses frame (frames 26-28 and 31-32). "
        "Press (frame 19+), then release (around frame 28). "
        "Act around frames 11-18. The annotation labels frame 30 as contact.")
    before = copy.deepcopy(record)
    payload = contact.build_case_payload(record["contact"], record)
    assert payload["notes_for_m3_judge"] == (
        "Hold the glasses frame. Press, then release. "
        "Act during the source demonstration. The annotation labels the source observation as contact.")
    assert record == before
    assert payload["functional_contact_regions"]
    assert payload["temporal_contact_sequence"]


def test_optional_events_do_not_affect_completion_or_coverage():
    annotation = {"case_id": "example", "goal_checks": [
        {"id": "g1", "weight": 2}, {"id": "g2", "weight": 1}],
        "action_events": [{"id": "e1", "weight": 2, "required": True},
                          {"id": "e2", "weight": 1, "required": True},
                          {"id": "optional", "weight": 100, "required": False}],
        "action_order": [["e1", "e2"]]}
    response = {"goal_checks": [{"id": "g1", "score_0_to_4": 3}, {"id": "g2", "score_0_to_4": 4}],
                "action_events": [{"id": "e1", "score_0_to_4": 3, "first_frame": 20},
                                  {"id": "e2", "score_0_to_4": 4, "first_frame": 1},
                                  {"id": "optional", "score_0_to_4": 0, "first_frame": None}]}
    goal = goal_action.compute_goal_score(annotation, response)
    action = goal_action.compute_action_score(annotation, response)
    assert goal["score"] == action["score"] == pytest.approx(5 / 6)
    assert goal["goal_coverage_at_3"] == action["event_coverage_at_3"] == 1
    assert goal["goal_coverage_at_4"] == action["event_coverage_at_4"] == pytest.approx(1 / 3)
    normalized = goal_action.normalize_annotation(annotation)
    assert [e["id"] for e in normalized["action_events"]] == ["e1", "e2"]
    payload = goal_action.build_case_annotation_for_prompt(annotation)
    assert "optional" not in json.dumps(payload) and "action_order" not in payload


def test_real_cases_use_required_events_and_canonical_families(tmp_path):
    original = read_jsonl(ROOT / "benchmark/annotations.jsonl")
    records = prepare(tmp_path, tmp_path, "example", ROOT / "benchmark/annotations.jsonl", True)
    for record in records:
        normalized = goal_action.normalize_annotation(record["annotation"])
        goal_payload = goal_action.build_case_annotation_for_prompt(normalized)
        contact_payload = contact.build_case_payload(record["contact"], record)
        assert goal_payload["family"] == contact_payload["family"] == benchmark_family(record)
        assert contact_payload["bimanual_or_support_roles"]
        response = goal_response(record)
        values = run.validate_and_score("M1_M2", response, record)[1]
        assert values["M2"] == values["M2_coverage_at_3"] == values["M2_coverage_at_4"] == 1
        optional_ids = {e["id"] for e in record["annotation"]["action_events"] if not e.get("required", True)}
        assert not optional_ids & {e["id"] for e in goal_payload["action_events"]}
    assert sum(not e.get("required", True) for a in original for e in a["action_events"]) == 52
    assert original == read_jsonl(ROOT / "benchmark/annotations.jsonl")


def test_contact_applicability_comes_from_annotation(tmp_path):
    record = example(tmp_path)
    record["contact"]["contact_spec"]["m3_applicable_subdimensions"] = ["affordance_region", "object_response"]
    dimensions = contact.applicable_dimensions(record["contact"])
    assert dimensions == ["contact_region_transfer", "object_response_causality"]
    response = contact_response()
    response["h2r_affordance_contact_transfer"]["object_response_causality"]["score_0_to_4"] = 2
    for dim in set(contact.DIMENSIONS) - set(dimensions):
        # Even a judge-supplied score for an excluded dimension cannot affect M3.
        response["h2r_affordance_contact_transfer"][dim]["score_0_to_4"] = 0
    clean, values = run.validate_and_score("M3", response, record)
    assert values["M3"] == .75
    assert set(clean["h2r_affordance_contact_transfer"]) == set(dimensions)
    for dim in set(contact.DIMENSIONS) - set(dimensions):
        del response["h2r_affordance_contact_transfer"][dim]
    assert run.validate_and_score("M3", response, record)[1]["M3"] == .75
    prompt = run.make_prompts(record)["M3"]
    schema = json.loads(prompt.split("Return JSON in this exact schema:\n")[1])
    assert set(schema["h2r_affordance_contact_transfer"]) == set(dimensions)
    response["source_grounding_hard_failure"]["triggered"] = True
    assert run.validate_and_score("M3", response, record)[1]["M3"] == 0
    del response["h2r_affordance_contact_transfer"][dimensions[0]]
    with pytest.raises(ValueError, match="missing dimension"):
        run.validate_and_score("M3", response, record)


@pytest.mark.parametrize("dimensions", [None, [], ["unknown"], ["object_response", "object_response_causality"]])
def test_invalid_contact_applicability_is_rejected(tmp_path, dimensions):
    record = example(tmp_path)
    record["contact"]["contact_spec"]["m3_applicable_subdimensions"] = dimensions
    with pytest.raises(ValueError):
        run.make_prompts(record)


def test_robot_subtype_does_not_change_presence_or_override_judge_scores(tmp_path):
    record = example(tmp_path)
    record["target_embodiment"] = "mechanical_arm"
    response = embodiment_response("dexterous_hand")
    response["embodiment_correctness"]["embodiment_category_match"]["score_0_to_4"] = 0
    response["embodiment_correctness"]["end_effector_correctness"]["score_0_to_4"] = 0
    assert run.validate_and_score("M4", response, record)[1]["M4"] == pytest.approx(.55)
    response["embodiment_correctness"]["end_effector_correctness"]["score_0_to_4"] = 3
    assert run.validate_and_score("M4", response, record)[1]["M4"] == pytest.approx(.7375)
    prompt = run.make_prompts(record)["M4"]
    assert "irrespective of subtype" in prompt
    assert "wrong robot subtype alone is not a hard failure" in prompt
    response["hard_failure"] = True
    assert run.validate_and_score("M4", response, record)[1]["M4"] == 0


def test_m5_clips_components_before_averaging():
    assert clip_quality_components([1.2, -.1, .6, .8]) == [1., 0., .6, .8]
    assert quality_mean([1.2, -.1, .6, .8]) == pytest.approx(.6)
    for invalid in ([float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0], [0, 0, 0]):
        with pytest.raises(ValueError):
            quality_mean(invalid)


def test_aggregate_requires_matching_populations_and_uses_component_means():
    rows = [score_record(id=str(i), model="m", target_embodiment="mechanical_arm",
                 generated_video_sha256=str(i), **{f"M{k}": float(i) for k in range(1, 5)}) for i in range(2)]
    m5 = [quality_record(id=str(i), video_sha256=str(i), m5=.8) for i in range(2)]
    with pytest.raises(ValueError, match="same video population"):
        aggregate(rows, m5[1:])
    out, summary = aggregate(rows, m5)
    assert summary[0]["H2RCore"] == pytest.approx(53)
    assert summary[0]["H2RCore"] == pytest.approx(core(summary[0]))
    assert all(summary[0]["n_" + m] == 2 for m in ["M1", "M2", "M3", "M4", "M5", "H2RCore"])
    assert sum(r["H2RCore"] for r in out) / 2 == pytest.approx(53)


def test_coverage_is_averaged_saved_and_aggregated(tmp_path, monkeypatch):
    record = example(tmp_path)
    video = tmp_path / "mock-video.mp4"
    video.write_bytes(b"mock media")
    record.update(source_video=str(video), generated_video=str(video))
    monkeypatch.setattr(run, "sample_frames", lambda path, n: (list(range(n)), ["mock-image"] * n))

    def fake(config, prompt, sequences, timeout):
        if "M3:" in prompt:
            return json.dumps(contact_response())
        if prompt.startswith("Evaluate only Embodiment"):
            return json.dumps(embodiment_response(record["target_embodiment"]))
        response = goal_response(record)
        if config["name"] == "gemini":
            for key in ("goal_checks", "action_events"):
                for item in response[key]:
                    item["score_0_to_4"] = 0
                    if key == "action_events": item["first_frame"] = None
        elif config["name"] == "gpt":
            for key in ("goal_checks", "action_events"):
                for item in response[key]:
                    item["score_0_to_4"] = 3
        return json.dumps(response)

    monkeypatch.setattr(run, "call_judge", fake)
    result = run.judge_record(record, [run.provider_config(n) for n in ("qwen", "gemini", "gpt")], tmp_path)
    assert result["M1"] == result["M2"] == pytest.approx(7 / 12)
    expected_coverage = {key: (2 / 3 if key.endswith("_3") else 1 / 3) for key in COVERAGE_METRICS}
    assert all(result[key] == pytest.approx(value) for key, value in expected_coverage.items())
    saved = json.loads((tmp_path / "judgments" / record["id"] / "qwen/M1_M2.json").read_text())
    assert all(saved["scores"][key] == 1 for key in COVERAGE_METRICS)
    _, summary = aggregate([result], [])
    assert all(summary[0][key] == pytest.approx(value) for key, value in expected_coverage.items())


@pytest.mark.parametrize("metric", ["M1", "M2", "M3_source", "M3_generated", "M4"])
@pytest.mark.parametrize("fault", ["missing_frames", "null_frames", "empty_positive", "out_of_range", "bool_frame", "missing_reason", "blank_reason"])
def test_evidence_and_rationale_contract(tmp_path, metric, fault):
    record = example(tmp_path)
    field = "evidence_generated_frames"
    if metric in {"M1", "M2"}:
        response, group = goal_response(record), "M1_M2"
        item = response["goal_checks" if metric == "M1" else "action_events"][0]
    elif metric.startswith("M3"):
        response, group = contact_response(), "M3"
        item = response["h2r_affordance_contact_transfer"][contact.DIMENSIONS[0]]
        if metric == "M3_source": field = "evidence_source_frames"
    else:
        response, group = embodiment_response(record["target_embodiment"]), "M4"
        item = response["embodiment_correctness"][embodiment.DIMENSIONS[0]]
    if fault == "missing_frames": item.pop(field)
    if fault == "null_frames": item[field] = None
    if fault == "empty_positive": item[field] = []
    if fault == "out_of_range": item[field] = [26]
    if fault == "bool_frame": item[field] = [True]
    if fault == "missing_reason": item.pop("reason")
    if fault == "blank_reason": item["reason"] = "  "
    with pytest.raises(ValueError):
        run.validate_and_score(group, response, record)


def test_zero_scores_allow_explicit_absence_and_evidence_is_preserved(tmp_path):
    record = example(tmp_path)
    response = goal_response(record)
    response["goal_checks"][0]["evidence_generated_frames"] = [23, 25]
    clean, _ = run.validate_and_score("M1_M2", response, record)
    assert clean["goal_checks"][0]["evidence_generated_frames"] == [23, 25]
    for key in ("goal_checks", "action_events"):
        for item in response[key]:
            item.update(score_0_to_4=0, evidence_generated_frames=[], reason="Required evidence is absent")
            if key == "action_events": item["first_frame"] = None
    assert run.validate_and_score("M1_M2", response, record)[1]["M1"] == 0
    for metric, response, key in [
        ("M3", contact_response(), "h2r_affordance_contact_transfer"),
        ("M4", embodiment_response(record["target_embodiment"]), "embodiment_correctness")]:
        for item in response[key].values():
            item.update(score_0_to_4=0, evidence_generated_frames=[], reason="Required evidence is absent")
            if metric == "M3": item["evidence_source_frames"] = []
        assert run.validate_and_score(metric, response, record)[1][metric] == 0


@pytest.mark.parametrize("fault", ["missing", "null", "nan", "bool", "above_one", "missing_coverage", "provider_missing", "wrong_mean"])
def test_aggregate_rejects_incomplete_or_invalid_scores(fault):
    row = score_record(id="a", model="m", target_embodiment="mechanical_arm",
                       generated_video_sha256="hash", M1=.5, M2=.5, M3=.5, M4=.5)
    if fault == "missing": row.pop("M1")
    if fault == "null": row["M1"] = None
    if fault == "nan": row["M1"] = float("nan")
    if fault == "bool": row["M1"] = True
    if fault == "above_one": row["M1"] = 1.1
    if fault == "missing_coverage": row.pop("M1_coverage_at_3")
    if fault == "provider_missing": row["providers"]["qwen"].pop("M1")
    if fault == "wrong_mean": row["M1"] = .9
    with pytest.raises(ValueError): aggregate([row], [])


@pytest.mark.parametrize("fault", ["old_protocol", "wrong_frames", "missing_models", "different_model", "different_judges", "temperature"])
def test_aggregate_rejects_mixed_profiles(fault):
    first = score_record(id="a", model="m", target_embodiment="mechanical_arm",
                         generated_video_sha256="hash", M1=.5, M2=.5, M3=.5, M4=.5)
    second = copy.deepcopy(first)
    second["id"] = "b"
    if fault == "old_protocol": second["protocol"] = "release-mixed-spec-v1"
    if fault == "wrong_frames": second["frame_counts"]["M4"] = 12
    if fault == "missing_models": second.pop("judge_models")
    if fault == "different_model": second["judge_models"]["qwen"] = "different-model"
    if fault == "different_judges":
        second["judge_models"].pop("gpt")
        second["providers"].pop("gpt")
    if fault == "temperature": second["temperature"] = 1
    with pytest.raises(ValueError): aggregate([first, second], [])


def test_custom_judges_are_explicit_and_m5_configs_cannot_mix():
    rows = [score_record(id=str(i), model="m", target_embodiment="mechanical_arm",
                         generated_video_sha256=str(i), M1=.5, M2=.5, M3=.5, M4=.5) for i in range(2)]
    for row in rows:
        row["judge_models"] = {"qwen": run.DEFAULT_MODELS["qwen"]}
        row["providers"] = {"qwen": row["providers"]["qwen"]}
    _, summary = aggregate(rows, [])
    assert summary[0]["paper_judges"] is False
    assert json.loads(summary[0]["judge_models"]) == rows[0]["judge_models"]
    quality = [quality_record(id=str(i), video_sha256=str(i), m5=.8) for i in range(2)]
    quality[1]["run_signature"] = "different-weights"
    with pytest.raises(ValueError, match="M5 configurations"): aggregate(rows, quality)


def test_command_pipeline_preserves_evidence_and_paper_aggregation(tmp_path, monkeypatch):
    import csv
    import sys
    from evaluation import aggregate as aggregation
    from evaluation.common import write_jsonl
    records = prepare(tmp_path, tmp_path, "pipeline-model", ROOT / "benchmark/annotations.jsonl", True)[:2]
    for record in records:
        for field in ("source_video", "generated_video"):
            path = Path(record[field])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mock media")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, records)
    monkeypatch.setattr(run, "sample_frames", lambda path, n: (list(range(n)), ["mock-image"] * n))
    monkeypatch.setattr(run, "environment_info", lambda: {})
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    injected_invalid_response = False
    for name in run.DEFAULT_MODELS:
        monkeypatch.setenv(f"H2R_{name.upper()}_API_KEY", "test-key")
        monkeypatch.setenv(f"H2R_{name.upper()}_BASE_URL", "https://example.invalid")
        monkeypatch.delenv(f"H2R_{name.upper()}_MODEL", raising=False)

    def fake(config, prompt, sequences, timeout):
        nonlocal injected_invalid_response
        score = {"qwen": 4, "gemini": 2, "gpt": 0}[config["name"]]
        assert all(len(frames) == 25 for _, frames in sequences)
        if "M3:" in prompt:
            response = contact_response()
            items = response["h2r_affordance_contact_transfer"].values()
        elif prompt.startswith("Evaluate only Embodiment"):
            response = embodiment_response("mechanical_arm")
            items = response["embodiment_correctness"].values()
        else:
            response = goal_response(records[0])
            items = response["goal_checks"] + response["action_events"]
        for item in items:
            item["score_0_to_4"] = score
            if "first_frame" in item and score == 0: item["first_frame"] = None
        if "goal_checks" in response and config["name"] == "qwen" and not injected_invalid_response:
            response["goal_checks"][0].pop("evidence_generated_frames")
            injected_invalid_response = True
        return json.dumps(response)

    monkeypatch.setattr(run, "call_judge", fake)
    monkeypatch.setattr(sys, "argv", ["run", "--manifest", str(manifest), "--output-dir", str(tmp_path / "judges")])
    run.main()
    scores = read_jsonl(tmp_path / "judges/scores.jsonl")
    assert len(scores) == 2
    quality = [quality_record(id=r["id"], video_sha256=r["generated_video_sha256"], m5=.8) for r in scores]
    write_jsonl(tmp_path / "m5.jsonl", quality)
    monkeypatch.setattr(sys, "argv", ["aggregate", "--scores", str(tmp_path / "judges/scores.jsonl"),
        "--m5", str(tmp_path / "m5.jsonl"), "--output-dir", str(tmp_path / "summary")])
    aggregation.main()
    with (tmp_path / "summary/summary.csv").open() as handle:
        summaries = list(csv.DictReader(handle))
    assert all(float(r["H2RCore"]) == pytest.approx(53) and r["paper_judges"] == "True" for r in summaries)
    cache = tmp_path / "judges/judgments" / records[0]["id"] / "qwen/M1_M2.json"
    saved = json.loads(cache.read_text())
    assert saved["judgment"]["goal_checks"][0]["evidence_generated_frames"] == [25]
    assert saved["attempt"] == 2
    assert cache.with_name("M1_M2.attempt1.failed.json").is_file()
