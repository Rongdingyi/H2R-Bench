"""Preserved metric/prompt functions from h2r-bench-eval/scripts/evaluate_h2r_contact_v2.py."""
from __future__ import annotations
import json
import re
from typing import Any, Literal

PROMPT_VERSION = "h2r_contact_v2_5dims_contactspec_24source_24generated"


DIMENSIONS = [
    "contact_region_transfer",
    "contact_establishment",
    "manipulation_mode_transfer",
    "object_response_causality",
    "embodiment_compatible_contact",
]


VALID_SCORE_VALUES = {0, 1, 2, 3, 4}


TARGET_EMBODIMENT_DESCRIPTIONS = {
    "mechanical_arm": "robot arm(s) with rigid two-jaw parallel gripper end-effector(s) replacing human hands",
    "dual_arm": "robot arm(s) with rigid two-jaw parallel gripper end-effector(s) replacing human hands",
    "dexterous_hand": "robotic dexterous hand(s) with multiple articulated fingers replacing human hands",
}


def build_case_payload(spec_record: dict[str, Any], gen: dict[str, Any]) -> dict[str, Any]:
    target = str(gen.get("target_embodiment") or "unknown")
    spec = spec_record.get("contact_spec") or {}
    return {
        "case_id": spec_record.get("case_id") or gen.get("case_id"),
        "family": spec_record.get("family") or gen.get("family"),
        "scene": spec_record.get("scene") or gen.get("scene"),
        "target_embodiment": target,
        "target_embodiment_description": TARGET_EMBODIMENT_DESCRIPTIONS.get(target, target),
        "task_summary": spec.get("task_summary", ""),
        "manipulated_entities": spec.get("manipulated_entities", []),
        "functional_contact_regions": spec.get("functional_contact_regions", []),
        "manipulation_modes": spec.get("manipulation_modes", []),
        "object_responses": spec.get("object_responses", []),
        "temporal_contact_sequence": spec.get("temporal_contact_sequence", []),
        "embodiment_contact_constraints": spec.get("embodiment_contact_constraints", {}),
        "notes_for_m3_judge": spec.get("notes_for_m3_judge", ""),
    }


def build_h2r_contact_prompt(case_payload_json: str, frame_count: int) -> str:
    return f"""You are evaluating M3: Human-to-Robot Affordance-Contact Transfer for H2R-Bench.

You will receive two videos represented by uniformly sampled frames:
1. source_{frame_count}_frames: the original human-hand egocentric manipulation video.
2. generated_{frame_count}_frames: the generated robot manipulation video.

You will also receive a contact specification derived from the source video. It describes the manipulated entities, functional contact regions, manipulation modes, object responses, temporal contact sequence, and target-embodiment contact constraints.

Your task is to judge whether the generated robot video transfers the source human contact semantics into a robot-compatible execution.

Important rules:
1. Output strict JSON only. Do not output Markdown.
2. Use only integer frame ids from 1 to {frame_count}. Do not write strings such as "frame_01".
3. Frame ids are local to each video: source_frame_01..source_frame_{frame_count:02d} and generated_frame_01..generated_frame_{frame_count:02d}. Never count the source and generated images together as 1..{frame_count * 2}.
4. Before scoring contact, verify source grounding: the generated video must retain the source scene/camera context and the task-relevant objects, tools, and containers. If these are substantially substituted or absent, trigger source_grounding_hard_failure and assign zero to every contact dimension. Do not credit generic-looking contact in a different scene.
5. Focus on contact transfer, not generic video quality.
6. Do not mainly judge final task success; that is M1.
7. Do not mainly judge action ordering; that is M2.
8. Do not mainly judge robot appearance correctness; that is M4.
9. If a human hand, human arm, or non-robot actor performs the required contact in the generated video, score contact transfer low even if the object moves correctly.
10. If the object changes state without visible robot contact/control, contact_establishment and object_response_causality should be low.
11. The robot does not need to copy the exact human trajectory. It only needs functionally equivalent contact using the target embodiment.

Evaluate exactly these five dimensions:

A. contact_region_transfer
Does the robot contact the same or functionally equivalent object/tool/surface region as the source human demonstration?

B. contact_establishment
Is there visible, sustained, physically plausible contact/control between the robot end-effector or held tool and the relevant object/surface? Penalize hovering, empty-space contact, disconnected motion, and object motion without contact.

C. manipulation_mode_transfer
Is the functional manipulation mode transferred, such as grasp, pinch, support, push, pull, press, rotate, insert, connect, detach, wipe, pour, fold, spread, scoop, or stabilize?

D. object_response_causality
Does the object/tool/surface respond in a way consistent with the source contact and caused by robot contact/control, rather than teleporting, floating, duplicating, penetrating, disappearing, or changing by itself?

E. embodiment_compatible_contact
Is the contact strategy feasible for the target embodiment, using the constraints in the contact spec? For example, a parallel-jaw gripper should grasp, press, or push using gripper-like contacts; a dexterous hand may use articulated fingers. Do not require perfect visual morphology, but penalize contact behavior that only a human hand could perform.

Scoring:
0 = no transfer or completely wrong.
1 = weak hint, mostly wrong.
2 = partial transfer with important missing/unstable/incorrect conditions.
3 = mostly successful transfer with minor ambiguity or physical/contact issues.
4 = clear successful contact transfer with visible functionally equivalent contact and causal object response.

Do not use 2 or 3 as a default compromise. Choose based on visible evidence.
If the generated video is too blurry to verify contact, assign a lower score and explain the ambiguity.

Input contact specification:
{case_payload_json}

Return JSON in this exact schema:
{{
  "source_grounding_hard_failure": {{
    "triggered": false,
    "reason": "",
    "evidence_generated_frames": []
  }},
  "h2r_affordance_contact_transfer": {{
    "contact_region_transfer": {{
      "score_0_to_4": 0,
      "evidence_source_frames": [],
      "evidence_generated_frames": [],
      "reason": ""
    }},
    "contact_establishment": {{
      "score_0_to_4": 0,
      "evidence_source_frames": [],
      "evidence_generated_frames": [],
      "reason": ""
    }},
    "manipulation_mode_transfer": {{
      "score_0_to_4": 0,
      "evidence_source_frames": [],
      "evidence_generated_frames": [],
      "reason": ""
    }},
    "object_response_causality": {{
      "score_0_to_4": 0,
      "evidence_source_frames": [],
      "evidence_generated_frames": [],
      "reason": ""
    }},
    "embodiment_compatible_contact": {{
      "score_0_to_4": 0,
      "evidence_source_frames": [],
      "evidence_generated_frames": [],
      "reason": ""
    }}
  }},
  "overall_reason": ""
}}
"""


def _coerce_score(value: Any) -> int:
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


def _clean_frame_list(value: Any, field: str, max_frame: int) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{field} contains invalid frame id {item!r}")
        if isinstance(item, int):
            frame = item
        elif isinstance(item, float) and item.is_integer():
            frame = int(item)
        elif isinstance(item, str) and item.strip().isdigit():
            frame = int(item.strip())
        else:
            raise ValueError(f"{field} contains invalid frame id {item!r}")
        if frame < 1 or frame > max_frame:
            raise ValueError(f"{field} frame id out of range 1..{max_frame}: {frame}")
        out.append(frame)
    return out


def validate_h2r_contact_output(obj: dict[str, Any], max_frame: int) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("judge output must be a JSON object")
    root = obj.get("h2r_affordance_contact_transfer")
    if not isinstance(root, dict):
        raise ValueError("missing h2r_affordance_contact_transfer object")
    clean_root: dict[str, Any] = {}
    for dim in DIMENSIONS:
        item = root.get(dim)
        if not isinstance(item, dict):
            raise ValueError(f"missing dimension object: {dim}")
        clean_root[dim] = {
            **item,
            "score_0_to_4": _coerce_score(item.get("score_0_to_4")),
            "evidence_source_frames": _clean_frame_list(
                item.get("evidence_source_frames", []), f"{dim}.evidence_source_frames", max_frame
            ),
            "evidence_generated_frames": _clean_frame_list(
                item.get("evidence_generated_frames", []), f"{dim}.evidence_generated_frames", max_frame
            ),
            "reason": str(item.get("reason", "")),
        }
    hard_failure = obj.get("source_grounding_hard_failure", {})
    if not isinstance(hard_failure, dict):
        raise ValueError("source_grounding_hard_failure must be an object")
    triggered = hard_failure.get("triggered", False)
    if not isinstance(triggered, bool):
        raise ValueError("source_grounding_hard_failure.triggered must be boolean")
    return {
        "source_grounding_hard_failure": {
            "triggered": triggered,
            "reason": str(hard_failure.get("reason", "")),
            "evidence_generated_frames": _clean_frame_list(
                hard_failure.get("evidence_generated_frames", []),
                "source_grounding_hard_failure.evidence_generated_frames",
                max_frame,
            ),
        },
        "h2r_affordance_contact_transfer": clean_root,
        "overall_reason": str(obj.get("overall_reason", "")),
    }


def compute_provider_h2r_score(clean_obj: dict[str, Any]) -> dict[str, Any]:
    root = clean_obj["h2r_affordance_contact_transfer"]
    hard_failure = clean_obj["source_grounding_hard_failure"]
    per_dimension: dict[str, Any] = {}
    normalized_scores: list[float] = []
    for dim in DIMENSIONS:
        item = root[dim]
        score = 0 if hard_failure["triggered"] else int(item["score_0_to_4"])
        norm = score / 4.0
        normalized_scores.append(norm)
        per_dimension[dim] = {
            "score_0_to_4": score,
            "normalized_score": norm,
            "evidence_source_frames": item.get("evidence_source_frames", []),
            "evidence_generated_frames": item.get("evidence_generated_frames", []),
            "reason": item.get("reason", ""),
        }
    return {
        "score": sum(normalized_scores) / len(normalized_scores),
        "source_grounding_hard_failure": hard_failure,
        "valid_dimensions": list(DIMENSIONS),
        "per_dimension": per_dimension,
    }
