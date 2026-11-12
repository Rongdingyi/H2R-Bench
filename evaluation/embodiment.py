"""Embodiment evaluation adapted from the research evaluator."""
from __future__ import annotations
import json
import re
from typing import Any, Literal
from evaluation.common import EVALUATION_FRAME_COUNT, require_evidence, require_reason

PROMPT_VERSION = "embodiment_correctness_v7_evidence_25frames"


DIMENSIONS = [
    "target_robot_presence",
    "human_absence",
    "embodiment_category_match",
    "end_effector_correctness",
    "structural_consistency",
]


DIMENSION_WEIGHTS = {
    "target_robot_presence": 0.20,
    "human_absence": 0.20,
    "embodiment_category_match": 0.20,
    "end_effector_correctness": 0.25,
    "structural_consistency": 0.15,
}


VALID_SCORE_VALUES = {0, 1, 2, 3, 4}


END_EFFECTOR_TYPES = {"rigid_two_jaw_gripper", "robotic_multi_finger_dexterous_hand", "human_hand_or_skin", "generic_robot_claw_or_tool", "ambiguous_or_occluded", "no_robot_visible"}


DEFAULT_TARGET_DESCRIPTIONS = {
    "mechanical_arm": "two robotic arms with rigid two-jaw parallel grippers replacing the human hands. Each end-effector must be a rigid two-finger gripper used for grasping, not a human hand and not a multi-finger dexterous hand.",
    "dual_arm": "two robotic arms with rigid two-jaw parallel grippers replacing the human hands. Each end-effector must be a rigid two-finger gripper used for grasping, not a human hand and not a multi-finger dexterous hand.",
    "two_finger_gripper": "a robotic arm with a rigid two-finger gripper replacing the human hand. The end-effector should have two opposing gripper fingers, not five human-like fingers.",
    "dexterous_hand": "robotic dexterous hand(s) with multiple articulated fingers replacing the human hands. The target should look like a robotic multi-finger hand, not a simple two-finger gripper and not a human skin hand.",
    "robot_arm": "a robotic arm replacing the human hand.",
}


def get_target_embodiment_description(gen: dict[str, Any]) -> tuple[str, bool]:
    explicit = gen.get("target_embodiment_description")
    if explicit:
        return str(explicit), False
    target = str(gen.get("target_embodiment", "unknown"))
    return DEFAULT_TARGET_DESCRIPTIONS.get(target, target), True


def build_embodiment_prompt(gen: dict[str, Any]) -> str:
    target = str(gen.get("target_embodiment", "unknown"))
    description, _ = get_target_embodiment_description(gen)
    if target in {"mechanical_arm", "dual_arm", "two_finger_gripper"}:
        target_rule = "Target is robot arm(s) with rigid two-jaw parallel gripper(s). Not dexterous hand, not human hand."
        category_rule = "category=robot arm/gripper manipulator; end_effector=rigid two-jaw parallel gripper."
    elif target == "dexterous_hand":
        target_rule = "Target is robotic dexterous hand(s) with articulated multi-finger end-effector(s). Not two-jaw gripper, not human hand."
        category_rule = "category=robotic dexterous hand; end_effector=articulated multi-finger robotic hand."
    else:
        target_rule = f"Target: {description}"
        category_rule = "Use the target description."
    ref_note = "Robot reference image may be provided; use it only for appearance." if gen.get("robot_reference_image") else "No robot reference image."
    return f"""Evaluate only Embodiment Correctness from sampled generated frames. Ignore task success/action order/contact except robot structural stability.

You will receive {EVALUATION_FRAME_COUNT} uniformly sampled generated frames, including the first and last frame. Use integer evidence indices from 1 to {EVALUATION_FRAME_COUNT}.

Target={target}. {description}
Rule: {target_rule} {category_rule}
{ref_note}

Output strict JSON only. Scores are integers 0-4: 0 wrong/absent, 1 weak, 2 partial, 3 mostly correct, 4 clearly correct.
Each dimension must include a non-empty reason and evidence_generated_frames. Positive scores require at least one cited frame; zero scores may use an empty list with a rationale. If hard_failure is true, explain it in hard_failure_reason.
Do not treat two-jaw grippers and multi-finger dexterous hands as interchangeable.

Diagnosis observed_end_effector_type must be one of: rigid_two_jaw_gripper, robotic_multi_finger_dexterous_hand, human_hand_or_skin, generic_robot_claw_or_tool, ambiguous_or_occluded, no_robot_visible.

Score dimensions:
- target_robot_presence: a robotic actor is visibly present, irrespective of subtype. A visible dexterous hand receives presence credit even when a parallel-jaw gripper is requested; assess type mismatch separately below.
- human_absence: no human hand/skin/arm remains.
- embodiment_category_match: correct broad category.
- end_effector_correctness: correct end-effector subtype.
- structural_consistency: robot body/end-effector stay stable over time.

Hard failure: if human hands perform the principal manipulation or no robotic actor is visible. A wrong robot subtype alone is not a hard failure.

Return exactly:
{{
  "embodiment_diagnosis": {{
    "visible_manipulator": "",
    "observed_end_effector_type": "ambiguous_or_occluded",
    "two_jaw_gripper_visible": false,
    "multi_finger_dexterous_hand_visible": false,
    "human_hand_visible": false,
    "diagnosis_reason": ""
  }},
  "embodiment_correctness": {{
    "target_robot_presence": {{"score_0_to_4": 0, "evidence_generated_frames": [], "reason": ""}},
    "human_absence": {{"score_0_to_4": 0, "evidence_generated_frames": [], "reason": ""}},
    "embodiment_category_match": {{"score_0_to_4": 0, "evidence_generated_frames": [], "reason": ""}},
    "end_effector_correctness": {{"score_0_to_4": 0, "evidence_generated_frames": [], "reason": ""}},
    "structural_consistency": {{"score_0_to_4": 0, "evidence_generated_frames": [], "reason": ""}}
  }},
  "hard_failure": false,
  "hard_failure_reason": "",
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


def _clean_frame_list(value: Any, field: str) -> list[int]:
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
        elif isinstance(item, str):
            text = item.strip()
            if text.isdigit():
                frame = int(text)
            else:
                match = re.fullmatch(r"frame[_\s-]*(\d+)", text, flags=re.IGNORECASE)
                if not match:
                    raise ValueError(f"{field} contains invalid frame id {item!r}")
                frame = int(match.group(1))
        else:
            raise ValueError(f"{field} contains invalid frame id {item!r}")
        if frame < 1 or frame > EVALUATION_FRAME_COUNT:
            raise ValueError(f"{field} frame id out of range 1..{EVALUATION_FRAME_COUNT}: {frame}")
        out.append(frame)
    return out


def _infer_diagnosis_from_text(root: dict[str, Any]) -> dict[str, Any]:
    reason_text = " ".join(
        str(item.get("reason", ""))
        for item in root.values()
        if isinstance(item, dict)
    ).lower()
    observed_type = "ambiguous_or_occluded"
    if any(k in reason_text for k in ["human hand", "human arm", "skin", "bare hand"]):
        observed_type = "human_hand_or_skin"
    elif any(k in reason_text for k in ["multi-finger", "multifinger", "dexterous", "articulated finger", "five-finger", "five finger"]):
        observed_type = "robotic_multi_finger_dexterous_hand"
    elif any(k in reason_text for k in ["two-jaw", "two jaw", "parallel gripper", "two-finger", "two finger", "opposing jaws"]):
        observed_type = "rigid_two_jaw_gripper"
    elif any(k in reason_text for k in ["robot claw", "robotic claw", "generic claw", "tool"]):
        observed_type = "generic_robot_claw_or_tool"
    return {
        "visible_manipulator": "",
        "observed_end_effector_type": observed_type,
        "two_jaw_gripper_visible": observed_type == "rigid_two_jaw_gripper",
        "multi_finger_dexterous_hand_visible": observed_type == "robotic_multi_finger_dexterous_hand",
        "human_hand_visible": observed_type == "human_hand_or_skin",
        "diagnosis_reason": "Inferred from dimension reasons because embodiment_diagnosis was missing or invalid.",
        "diagnosis_inferred": True,
    }


def validate_embodiment_output(obj: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("judge output must be a JSON object")
    root = obj.get("embodiment_correctness")
    if not isinstance(root, dict):
        raise ValueError("missing embodiment_correctness object")
    clean_root: dict[str, Any] = {}
    for dim in DIMENSIONS:
        item = root.get(dim)
        if not isinstance(item, dict):
            raise ValueError(f"missing dimension object: {dim}")
        score = _coerce_score(item.get("score_0_to_4"))
        clean_root[dim] = {
            "score_0_to_4": score,
            "evidence_generated_frames": require_evidence(item, "evidence_generated_frames", positive=score > 0),
            "reason": require_reason(item),
        }
    diagnosis = obj.get("embodiment_diagnosis")
    if not isinstance(diagnosis, dict):
        diagnosis = _infer_diagnosis_from_text(clean_root)
    observed_type = str(diagnosis.get("observed_end_effector_type", "ambiguous_or_occluded"))
    if observed_type not in END_EFFECTOR_TYPES:
        diagnosis = _infer_diagnosis_from_text(clean_root)
        observed_type = str(diagnosis.get("observed_end_effector_type", "ambiguous_or_occluded"))
    clean_diagnosis = {
        "visible_manipulator": str(diagnosis.get("visible_manipulator", "")),
        "observed_end_effector_type": observed_type,
        "two_jaw_gripper_visible": bool(diagnosis.get("two_jaw_gripper_visible", False)),
        "multi_finger_dexterous_hand_visible": bool(diagnosis.get("multi_finger_dexterous_hand_visible", False)),
        "human_hand_visible": bool(diagnosis.get("human_hand_visible", False)),
        "diagnosis_reason": str(diagnosis.get("diagnosis_reason", "")),
        "diagnosis_inferred": bool(diagnosis.get("diagnosis_inferred", False)),
    }
    hard_failure = obj.get("hard_failure")
    if not isinstance(hard_failure, bool):
        raise ValueError("hard_failure must be boolean")
    return {"embodiment_diagnosis": clean_diagnosis, "embodiment_correctness": clean_root, "hard_failure": hard_failure, "hard_failure_reason": require_reason(obj, "hard_failure_reason", allow_empty=not hard_failure), "overall_reason": str(obj.get("overall_reason", ""))}


def compute_provider_embodiment_score(clean_obj: dict[str, Any], target_embodiment: str | None = None) -> dict[str, Any]:
    root = clean_obj["embodiment_correctness"]
    diagnosis = clean_obj.get("embodiment_diagnosis", {})
    per_dimension: dict[str, Any] = {}
    normalized = []
    for dim in DIMENSIONS:
        item = root[dim]
        raw_score = int(item["score_0_to_4"])
        norm = raw_score / 4.0
        normalized.append(norm)
        per_dimension[dim] = {"score_0_to_4": raw_score, "normalized_score": norm, "evidence_generated_frames": item.get("evidence_generated_frames", []), "reason": item.get("reason", "")}
    if clean_obj.get("hard_failure"):
        score = 0.0
    else:
        total_weight = sum(DIMENSION_WEIGHTS[dim] for dim in DIMENSIONS)
        score = sum(
            DIMENSION_WEIGHTS[dim] * value
            for dim, value in zip(DIMENSIONS, normalized)
        ) / total_weight
    return {"score": score, "hard_failure": bool(clean_obj.get("hard_failure")), "hard_failure_reason": str(clean_obj.get("hard_failure_reason", "")), "embodiment_diagnosis": diagnosis, "per_dimension": per_dimension}
