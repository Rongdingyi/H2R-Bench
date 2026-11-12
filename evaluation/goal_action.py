"""Goal and action evaluation adapted from the research evaluator."""
from __future__ import annotations
import json
import re
from typing import Any, Literal
from evaluation.common import EVALUATION_FRAME_COUNT, benchmark_family, require_evidence, require_reason

MetricName = Literal["goal", "action"]


PROMPT_VERSION = "goal_action_v8_evidence_25frames"


VALID_SCORE_VALUES = {0, 1, 2, 3, 4}


def required_action_events(annotation):
    return [event for event in annotation.get("action_events", [])
            if event.get("required", True)]


def normalize_annotation(record: dict[str, Any]) -> dict[str, Any]:
    goal_checks = record.get("goal_checks")
    if goal_checks is None:
        goal_checks = record.get("goal_predicates", [])

    action_order = record.get("action_order")
    if action_order is None:
        action_order = record.get("precedence_edges", [])

    entities = record.get("entities") or {}
    normalized = dict(record)
    normalized["case_id"] = str(record["case_id"])
    normalized["family"] = benchmark_family(record)
    normalized["task_description"] = (
        record.get("task_description")
        or record.get("source_description")
        or record.get("task_name")
        or ""
    )
    normalized["entities"] = {
        "controlled_entity": entities.get("controlled_entity"),
        "target_entity": entities.get("target_entity"),
    }
    normalized["goal_checks"] = [
        {
            "id": str(item.get("id", f"g{i + 1}")),
            "description": str(item.get("description", "")),
            "weight": float(item.get("weight", 1.0)),
        }
        for i, item in enumerate(goal_checks)
    ]
    normalized["action_events"] = [
        {
            "id": str(item.get("id", f"e{i + 1}")),
            "description": str(item.get("description", "")),
            "required": bool(item.get("required", True)),
            "weight": float(item.get("weight", 1.0)),
        }
        for i, item in enumerate(record.get("action_events", []))
        if item.get("required", True)
    ]
    normalized["action_order"] = [
        [str(pair[0]), str(pair[1])]
        for pair in action_order
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    return normalized


def build_case_annotation_for_prompt(annotation: dict[str, Any], metric: MetricName | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "case_id": annotation["case_id"],
        "family": benchmark_family(annotation),
        "task_name": annotation.get("task_name"),
        "task_description": annotation.get("task_description", ""),
        "entities": annotation.get("entities", {}),
    }
    if metric in {None, "goal"}:
        payload["goal_checks"] = annotation.get("goal_checks", [])
    if metric in {None, "action"}:
        payload["action_events"] = required_action_events(annotation)
    return payload


def build_judge_prompt(annotation: dict[str, Any], metric: MetricName = "goal") -> str:
    case_annotation_json = json.dumps(
        build_case_annotation_for_prompt(annotation, metric),
        ensure_ascii=False,
        indent=2,
    )
    if metric == "goal":
        frame_notice = f"You will receive {EVALUATION_FRAME_COUNT} uniformly sampled frames from the generated video, including the first and last frame. Frame indices are from 1 to {EVALUATION_FRAME_COUNT}. Judge the final state using the end of the sequence, with earlier frames providing context."
        id_notice = "You must evaluate all goal_check ids from the case annotation."
    else:
        frame_notice = f"You will receive {EVALUATION_FRAME_COUNT} uniformly sampled frames from the generated video. Frame indices are from 1 to {EVALUATION_FRAME_COUNT}, including the first and last frame."
        id_notice = "You must evaluate all action_event ids from the case annotation."
    shared = f"""You are evaluating a generated robot manipulation video for H2R-Bench.

{frame_notice}

You will also receive a case annotation derived from the source human demonstration. The annotation defines the task goal and required action structure. Your job is to evaluate the generated video only. Do not evaluate the source video.

Important rules:

1. Output strict JSON only.
2. Do not output Markdown or explanations outside JSON. Every item must include a non-empty reason and evidence_generated_frames with integer indices in 1..{EVALUATION_FRAME_COUNT}. Cite at least one frame for positive scores; for zero scores, an empty list is allowed if the reason explains absent evidence.
3. For Goal-State Completion, judge the state at the end of the sequence; use preceding frames as context to determine whether it was reached and preserved.
4. For Action-Event Completion, use all frames 1--{EVALUATION_FRAME_COUNT}.
5. Goal-State Completion evaluates only annotation-defined final-state predicates, not the process.
6. Action-Event Completion evaluates only annotation-defined action event completion, not whether the final state is achieved.
7. Do not evaluate affordance/contact transfer quality here; that belongs to M3 unless missing contact makes an annotated action event impossible to judge.
8. Do not evaluate embodiment correctness here; wrong robot type, human-hand residue, or robot morphology errors belong to M4 unless they directly make the goal state or action event impossible to judge.
9. Do not evaluate visual fidelity, blur, flicker, or aesthetic quality here; that belongs to M5 unless the video is impossible to judge.
10. The word "actor" refers to the visible manipulating agent in the generated video, usually the target robot.
11. {id_notice}
12. Do not add new ids. Do not omit existing ids.
13. Do not output aggregate metric names such as goal_predicate_score, event_completion_score, or coverage. These are computed outside this prompt from your per-id scores.

Scoring rule:

For each goal_check and each action_event, use an integer score from 0 to 4.

0 = No visible evidence or completely failed.
The required state or event is absent, clearly wrong, or contradicted by the video.

1 = Weak evidence, mostly not completed.
There is a slight visual hint related to the required state or event, but the core requirement is not achieved.

2 = Partially completed.
The required state or event is partially achieved, roughly halfway completed, or visible but missing important conditions.

3 = Mostly completed.
The required state or event is largely achieved, but there are minor defects, instability, occlusion, ambiguity, or small visual inconsistencies.

4 = Clearly and fully completed.
The required state or event is clearly visible, complete, stable, and satisfies the requirement without obvious problems.

Only output integers in {{0, 1, 2, 3, 4}}. Do not output decimal scores.
Do not use 2 or 3 as a default compromise. Choose the score according to visible evidence.
"""
    if metric == "goal":
        return f"""{shared}
Goal-State Completion:

For each goal_check, judge whether the final state is achieved at the end of the sequence, using preceding frames as context to determine whether it was reached and preserved.

A goal_check score means:

0 = The final goal state is not achieved at all, or the result is clearly wrong.
1 = The final state shows only a weak hint toward the goal, but the core goal is not achieved.
2 = The final state is partially achieved, but important goal conditions are missing or highly uncertain.
3 = The final state is mostly achieved, but there are minor issues such as slight instability, partial occlusion, unclear placement, small spill, slight deformation, or visual ambiguity.
4 = The final state is clearly achieved and visually stable, with no obvious failure.

Score the final state, using earlier frames only as supporting context.

Case annotation:
{case_annotation_json}

Return JSON in this exact schema:

{{
  "goal_checks": [
    {{
      "id": "g1",
      "score_0_to_4": 0,
      "evidence_generated_frames": [],
      "reason": ""
    }}
  ]
}}
"""
    return f"""{shared}
Action-Event Completion:

For each action_event, judge whether the event occurs using all frames 1--{EVALUATION_FRAME_COUNT}.

An action_event score means:

0 = The action event does not occur, with no visible evidence.
1 = There is weak evidence or preparation for the event, but the core action does not really happen.
2 = The event partially occurs, but it is incomplete, unstable, interrupted, or missing an important condition such as clear contact, control, alignment, transfer, insertion, release, or surface change.
3 = The event mostly occurs, but there are minor defects, partial occlusion, temporal discontinuity, instability, or slight ambiguity.
4 = The event clearly and completely occurs.

For each action_event, also output first_frame:
- if score_0_to_4 > 0, first_frame must be an integer from 1 to {EVALUATION_FRAME_COUNT} indicating the earliest sampled frame where the event starts to be visible;
- if score_0_to_4 = 0, first_frame must be null.

Case annotation:
{case_annotation_json}

Return JSON in this exact schema:

{{
  "action_events": [
    {{
      "id": "e1",
      "score_0_to_4": 0,
      "evidence_generated_frames": [],
      "first_frame": null,
      "reason": ""
    }}
  ]
}}
"""


def build_combined_goal_action_prompt(annotation: dict[str, Any]) -> str:
    case_annotation_json = json.dumps(
        build_case_annotation_for_prompt(annotation, None),
        ensure_ascii=False,
        indent=2,
    )
    return f"""You are evaluating a generated robot manipulation video for H2R-Bench.

You will receive {EVALUATION_FRAME_COUNT} uniformly sampled frames from the generated video. Frame indices are from 1 to {EVALUATION_FRAME_COUNT}, including the first and last frame.

You will also receive a case annotation derived from the source human demonstration. The annotation defines the task goal and required action structure. Your job is to evaluate the generated video only. Do not evaluate the source video.

Important rules:

1. Output strict JSON only.
2. Do not output Markdown or explanations outside JSON. Every item must include a non-empty reason and evidence_generated_frames with integer indices in 1..{EVALUATION_FRAME_COUNT}. Cite at least one frame for positive scores; for zero scores, an empty list is allowed if the reason explains absent evidence.
3. For Goal-State Completion, judge the state at the end of the sequence; use preceding frames as context to determine whether it was reached and preserved.
4. For Action-Event Completion, use all frames 1--{EVALUATION_FRAME_COUNT}.
5. Goal-State Completion evaluates only annotation-defined final-state predicates, not the process.
6. Action-Event Completion evaluates only annotation-defined action event completion, not whether the final state is achieved.
7. Do not evaluate affordance/contact transfer quality here; that belongs to M3 unless missing contact makes an annotated action event impossible to judge.
8. Do not evaluate embodiment correctness here; wrong robot type, human-hand residue, or robot morphology errors belong to M4 unless they directly make the goal state or action event impossible to judge.
9. Do not evaluate visual fidelity, blur, flicker, or aesthetic quality here; that belongs to M5 unless the video is impossible to judge.
10. The word "actor" refers to the visible manipulating agent in the generated video, usually the target robot.
11. You must evaluate all goal_check ids and all action_event ids from the case annotation.
12. Do not add new ids. Do not omit existing ids.
13. Do not output aggregate metric names such as goal_predicate_score, event_completion_score, or coverage. These are computed outside this prompt from your per-id scores.

Scoring rule:

For each goal_check and each action_event, use an integer score from 0 to 4.

0 = No visible evidence or completely failed.
The required state or event is absent, clearly wrong, or contradicted by the video.

1 = Weak evidence, mostly not completed.
There is a slight visual hint related to the required state or event, but the core requirement is not achieved.

2 = Partially completed.
The required state or event is partially achieved, roughly halfway completed, or visible but missing important conditions.

3 = Mostly completed.
The required state or event is largely achieved, but there are minor defects, instability, occlusion, ambiguity, or small visual inconsistencies.

4 = Clearly and fully completed.
The required state or event is clearly visible, complete, stable, and satisfies the requirement without obvious problems.

Only output integers in {{0, 1, 2, 3, 4}}. Do not output decimal scores.
Do not use 2 or 3 as a default compromise. Choose the score according to visible evidence.

Goal-State Completion:

For each goal_check, judge whether the final state predicate is achieved at the end of the sequence, using preceding frames as context to determine whether it was reached and preserved.
Score the final state, using earlier frames only as supporting context.
If the final state is partially occluded but still inferable, score according to visible evidence and mention the ambiguity in reason.

Action-Event Completion:

For each action_event, judge whether the annotated event occurs using all frames 1--{EVALUATION_FRAME_COUNT}.
Score event occurrence only; do not require final success.
For each action_event, also output first_frame:
- if score_0_to_4 > 0, first_frame must be an integer from 1 to {EVALUATION_FRAME_COUNT} indicating the earliest sampled frame where the event starts to be visible;
- if score_0_to_4 = 0, first_frame must be null.

Case annotation:
{case_annotation_json}

Return JSON in this exact schema:

{{
  "goal_checks": [
    {{
      "id": "g1",
      "score_0_to_4": 0,
      "evidence_generated_frames": [],
      "reason": ""
    }}
  ],
  "action_events": [
    {{
      "id": "e1",
      "score_0_to_4": 0,
      "evidence_generated_frames": [],
      "first_frame": null,
      "reason": ""
    }}
  ]
}}
"""


def _coerce_score_0_to_4(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"score must be integer 0..4, got {value!r}")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
    else:
        raise ValueError(f"score must be integer 0..4, got {value!r}")
    if score not in VALID_SCORE_VALUES:
        raise ValueError(f"score must be one of {sorted(VALID_SCORE_VALUES)}, got {score}")
    return score


def clean_judge_output(annotation: dict[str, Any], output: dict[str, Any], metric: MetricName = "goal") -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("judge output must be a JSON object")
    field = "goal_checks" if metric == "goal" else "action_events"
    expected = annotation.get("goal_checks", []) if metric == "goal" else required_action_events(annotation)
    items = output.get(field)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"Missing or invalid {field}")
    ids = [item.get("id") for item in items]
    if not all(isinstance(id_, str) for id_ in ids) or len(ids) != len(set(ids)) or set(ids) != {item["id"] for item in expected}:
        raise ValueError(f"Missing, duplicate or extra {field} IDs")
    by_id = {item["id"]: item for item in items}
    cleaned = []
    for definition in expected:
        item = by_id[definition["id"]]
        score = _coerce_score_0_to_4(item.get("score_0_to_4"))
        frames = require_evidence(item, "evidence_generated_frames", positive=score > 0)
        result = dict(id=definition["id"], score_0_to_4=score,
                      evidence_generated_frames=frames, reason=require_reason(item))
        if metric == "action":
            first = item.get("first_frame")
            if score > 0:
                if type(first) is not int or not 1 <= first <= EVALUATION_FRAME_COUNT:
                    raise ValueError(f"first_frame must be integer 1..{EVALUATION_FRAME_COUNT}")
            elif first is not None:
                raise ValueError("first_frame must be null for a zero score")
            result["first_frame"] = first
        cleaned.append(result)
    return {field: cleaned}


def clean_combined_judge_output(annotation: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    cleaned_goal = clean_judge_output(annotation, output, "goal")
    cleaned_action = clean_judge_output(annotation, output, "action")
    return {**cleaned_goal, **cleaned_action}


def weighted_mean(items: list[dict[str, Any]], values_by_id: dict[str, float]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for item in items:
        weight = float(item.get("weight", 1.0))
        if weight <= 0:
            continue
        numerator += weight * float(values_by_id.get(item["id"], 0.0))
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def compute_goal_score(annotation: dict[str, Any], judge_output: dict[str, Any]) -> dict[str, Any]:
    output_by_id = {x["id"]: x for x in judge_output.get("goal_checks", [])}
    raw_by_id = {k: int(v.get("score_0_to_4", 0)) for k, v in output_by_id.items()}
    norm_by_id = {k: v / 4.0 for k, v in raw_by_id.items()}
    coverage_at_3_by_id = {k: 1.0 if v >= 3 else 0.0 for k, v in raw_by_id.items()}
    coverage_at_4_by_id = {k: 1.0 if v == 4 else 0.0 for k, v in raw_by_id.items()}
    per_check = []
    for item in annotation.get("goal_checks", []):
        goal_id = item["id"]
        raw = int(raw_by_id.get(goal_id, 0))
        out = output_by_id.get(goal_id, {})
        per_check.append(
            {
                "id": goal_id,
                "weight": float(item.get("weight", 1.0)),
                "score_0_to_4": raw,
                "normalized_score": raw / 4.0,
                "reason": str(out.get("reason", "")),
                "evidence_generated_frames": out.get("evidence_generated_frames", []),
            }
        )
    goal_predicate_score = weighted_mean(annotation.get("goal_checks", []), norm_by_id)
    goal_coverage_at_3 = weighted_mean(annotation.get("goal_checks", []), coverage_at_3_by_id)
    goal_coverage_at_4 = weighted_mean(annotation.get("goal_checks", []), coverage_at_4_by_id)
    return {
        "score": goal_predicate_score,
        "goal_predicate_score": goal_predicate_score,
        "goal_coverage_at_3": goal_coverage_at_3,
        "goal_coverage_at_4": goal_coverage_at_4,
        "per_check": per_check,
        "per_goal_check_scores": per_check,
    }


def compute_action_score(annotation: dict[str, Any], judge_output: dict[str, Any]) -> dict[str, Any]:
    events = required_action_events(annotation)
    event_outputs = {x["id"]: x for x in judge_output.get("action_events", [])}
    raw_by_id = {k: int(v.get("score_0_to_4", 0)) for k, v in event_outputs.items()}
    completion_by_id = {k: v / 4.0 for k, v in raw_by_id.items()}
    coverage_at_3_by_id = {k: 1.0 if v >= 3 else 0.0 for k, v in raw_by_id.items()}
    coverage_at_4_by_id = {k: 1.0 if v == 4 else 0.0 for k, v in raw_by_id.items()}
    first_frame_by_id = {k: v.get("first_frame") for k, v in event_outputs.items()}
    per_event = []
    for item in events:
        event_id = item["id"]
        raw = int(raw_by_id.get(event_id, 0))
        out = event_outputs.get(event_id, {})
        per_event.append(
            {
                "id": event_id,
                "weight": float(item.get("weight", 1.0)),
                "score_0_to_4": raw,
                "completion": raw / 4.0,
                "first_frame": first_frame_by_id.get(event_id),
                "reason": str(out.get("reason", "")),
                "evidence_generated_frames": out.get("evidence_generated_frames", []),
            }
        )

    event_completion = weighted_mean(events, completion_by_id)
    if event_completion is None:
        event_completion = 0.0
    event_coverage_at_3 = weighted_mean(events, coverage_at_3_by_id)
    if event_coverage_at_3 is None:
        event_coverage_at_3 = 0.0
    event_coverage_at_4 = weighted_mean(events, coverage_at_4_by_id)
    if event_coverage_at_4 is None:
        event_coverage_at_4 = 0.0

    return {
        "score": event_completion,
        "event_completion": event_completion,
        "event_completion_score": event_completion,
        "event_coverage_at_3": event_coverage_at_3,
        "event_coverage_at_4": event_coverage_at_4,
        "per_event": per_event,
        "per_action_event_scores": per_event,
    }
