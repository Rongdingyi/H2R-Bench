import json
from collections import Counter

from evaluation.common import ROOT, read_jsonl
from evaluation import contact, embodiment, goal_action


def test_scene_annotation_and_prompt_alignment():
    scenes = read_jsonl(ROOT / "benchmark/annotations.jsonl")
    assert len(scenes) == len({r["case_id"] for r in scenes}) == 120
    assert Counter(r["scene_id"].split("_")[0] for r in scenes) == {f"F{i}": 20 for i in range(1, 7)}
    cases = {r["case_id"] for r in scenes}
    ids = (ROOT / "benchmark/scene_ids.txt").read_text().splitlines()
    assert len(ids) == 120 and set(ids) == {r["scene_id"] for r in scenes}
    for r in scenes:
        assert r["contact_spec"]["case_id"] == r["case_id"]
        assert r["goal_checks"] and r["action_events"]
        assert not {"source_video", "egodex_source_id", "clip_start_seconds", "clip_end_seconds"} & r.keys()
    prompts = read_jsonl(ROOT / "prompts/generation.jsonl")
    assert len(prompts) == len({r["id"] for r in prompts}) == 240
    assert Counter(r["case_id"] for r in prompts) == {c: 2 for c in cases}
    assert Counter(r["target_embodiment"] for r in prompts) == {"mechanical_arm": 120, "dexterous_hand": 120}
    assert sum(r["annotation_version"] == "corrected-source32-20260721" for r in scenes) == 12


def test_prompt_snapshots_match_preserved_builders():
    p = ROOT / "prompts/judging"
    ann = goal_action.normalize_annotation({'case_id': 'EXAMPLE', 'goal_checks': [], 'action_events': [], 'entities': {}})
    payload = json.dumps(goal_action.build_case_annotation_for_prompt(ann), ensure_ascii=False, indent=2)
    assert (p / "M1_M2_24frames.txt").read_text().replace("{{CASE_ANNOTATION_JSON}}", payload) == goal_action.build_combined_goal_action_prompt(ann)
    assert (p / "M3_24frames.txt").read_text() == contact.build_h2r_contact_prompt("{{CONTACT_SPEC_JSON}}", 24)
    for target in ("mechanical_arm", "dexterous_hand"):
        assert (p / f"M4_{target}.txt").read_text() == embodiment.build_embodiment_prompt({"target_embodiment": target})
