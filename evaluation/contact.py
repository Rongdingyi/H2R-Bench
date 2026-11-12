"""Functional-contact evaluation adapted from the research evaluator."""
from __future__ import annotations
import json
import re
from typing import Any, Literal
from evaluation.common import benchmark_family, require_evidence, require_reason

PROMPT_VERSION = "h2r_contact_v5_evidence_25frames"

# Source annotations use a different sampling grid. Keep their semantics, but
# let the judge locate evidence in the actual source frames sent with the request.
_FRAME_REFERENCE = r"\bframes?\s+\d+(?:\s*[-–—]\s*\d+)?\+?(?:\s*(?:,|and)\s*\d+(?:\s*[-–—]\s*\d+)?\+?)*"


def without_annotation_frame_ids(value):
    if isinstance(value, dict):
        return {k: without_annotation_frame_ids(v) for k, v in value.items()
                if k not in {"source_evidence_frames", "approx_source_frames"}}
    if isinstance(value, list):
        return [without_annotation_frame_ids(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"\s*\((?:around\s+)?" + _FRAME_REFERENCE + r"\)", "", value, flags=re.I)
        value = re.sub(r"\baround\s+" + _FRAME_REFERENCE, "during the source demonstration", value, flags=re.I)
        return re.sub(_FRAME_REFERENCE, "the source observation", value, flags=re.I)
    return value


DIMENSIONS = [
    "contact_region_transfer",
    "contact_establishment",
    "manipulation_mode_transfer",
    "object_response_causality",
    "embodiment_compatible_contact",
]

ANNOTATION_DIMENSIONS = {
    "affordance_region": "contact_region_transfer",
    "manipulation_mode": "manipulation_mode_transfer",
    "object_response": "object_response_causality",
}


def normalize_dimensions(dimensions):
    if not isinstance(dimensions, (list, tuple)) or not dimensions:
        raise ValueError("M3 requires a non-empty annotation-defined dimension list")
    if not all(isinstance(dim, str) for dim in dimensions):
        raise ValueError("M3 dimension names must be strings")
    mapped = [ANNOTATION_DIMENSIONS.get(dim, dim) for dim in dimensions]
    if len(set(mapped)) != len(mapped) or set(mapped) - set(DIMENSIONS):
        raise ValueError("Unknown or duplicate M3 dimensions")
    return [dim for dim in DIMENSIONS if dim in mapped]


def applicable_dimensions(spec_record):
    return normalize_dimensions(spec_record["contact_spec"].get("m3_applicable_subdimensions"))


VALID_SCORE_VALUES = {0, 1, 2, 3, 4}


TARGET_EMBODIMENT_DESCRIPTIONS = {
    "mechanical_arm": "robot arm(s) with rigid two-jaw parallel gripper end-effector(s) replacing human hands",
    "dual_arm": "robot arm(s) with rigid two-jaw parallel gripper end-effector(s) replacing human hands",
    "dexterous_hand": "robotic dexterous hand(s) with multiple articulated fingers replacing human hands",
}


def build_case_payload(spec_record: dict[str, Any], gen: dict[str, Any]) -> dict[str, Any]:
    target = str(gen.get("target_embodiment") or "unknown")
    spec = without_annotation_frame_ids(spec_record.get("contact_spec") or {})
    return {
        "case_id": spec_record.get("case_id") or gen.get("case_id"),
        "family": benchmark_family(spec_record),
        "scene": spec_record.get("scene") or gen.get("scene"),
        "target_embodiment": target,
        "target_embodiment_description": TARGET_EMBODIMENT_DESCRIPTIONS.get(target, target),
        "task_summary": spec.get("task_summary", ""),
        "manipulated_entities": spec.get("manipulated_entities", []),
        "functional_contact_regions": spec.get("functional_contact_regions", []),
        "manipulation_modes": spec.get("manipulation_modes", []),
        "object_responses": spec.get("object_responses", []),
        "temporal_contact_sequence": spec.get("temporal_contact_sequence", []),
        "bimanual_or_support_roles": spec.get("bimanual_or_support_roles", []),
        "m3_applicable_subdimensions": applicable_dimensions(spec_record),
        "embodiment_contact_constraints": spec.get("embodiment_contact_constraints", {}),
        "notes_for_m3_judge": spec.get("notes_for_m3_judge", ""),
    }


def build_h2r_contact_prompt(case_payload_json: str, frame_count: int,
                           dimensions=None) -> str:
    dimensions = normalize_dimensions(DIMENSIONS if dimensions is None else dimensions)
    response_schema = json.dumps({
        "source_grounding_hard_failure": {"triggered": False, "reason": "", "evidence_generated_frames": []},
        "h2r_affordance_contact_transfer": {
            dim: {"score_0_to_4": 0, "evidence_source_frames": [],
                  "evidence_generated_frames": [], "reason": ""} for dim in dimensions},
        "overall_reason": "",
    }, indent=2)
    return f"""You are evaluating M3: Human-to-Robot Affordance-Contact Transfer for H2R-Bench.

You will receive two videos represented by uniformly sampled frames:
1. source_{frame_count}_frames: the original human-hand egocentric manipulation video.
2. generated_{frame_count}_frames: the generated robot manipulation video.

You will also receive a contact specification derived from the source video. It describes the manipulated entities, functional contact regions, manipulation modes, object responses, temporal contact sequence, and target-embodiment contact constraints.
Locate source evidence in the supplied source frames; all evidence indices must refer to the frames in this request.

Your task is to judge whether the generated robot video transfers the source human contact semantics into a robot-compatible execution.

Important rules:
1. Output strict JSON only. Do not output Markdown. Each dimension must contain a non-empty reason and both evidence lists. Positive scores require source and generated evidence; zero scores may use empty lists with a rationale. The grounding decision must include a reason and evidence_generated_frames; cite frames if it fails.
2. Use only integer frame ids from 1 to {frame_count}. Do not write strings such as "frame_01".
3. Frame ids are local to each video: source_frame_01..source_frame_{frame_count:02d} and generated_frame_01..generated_frame_{frame_count:02d}. Never count the source and generated images together as 1..{frame_count * 2}.
4. Before scoring contact, verify source grounding: the generated video must retain the source scene/camera context and the task-relevant objects, tools, and containers. If these are substantially substituted or absent, trigger source_grounding_hard_failure and assign zero to every contact dimension. Do not credit generic-looking contact in a different scene.
5. Focus on contact transfer, not generic video quality.
6. Do not mainly judge final task success; that is M1.
7. Do not mainly judge action-event completion; that is M2.
8. Do not mainly judge robot appearance correctness; that is M4.
9. If a human hand, human arm, or non-robot actor performs the required contact in the generated video, score contact transfer low even if the object moves correctly.
10. If the object changes state without visible robot contact/control, contact_establishment and object_response_causality should be low.
11. The robot does not need to copy the exact human trajectory. It only needs functionally equivalent contact using the target embodiment.

The annotation fixes the applicable dimensions for all judges: {", ".join(dimensions)}.
Score only those dimensions. Omit non-applicable dimensions rather than assigning neutral or zero scores. Do not change applicability based on the generated video.
Dimension definitions:

A. contact_region_transfer
Does the robot contact the same or functionally equivalent object/tool/surface region as the source human demonstration?

B. contact_establishment
Is there visible, sustained, physically plausible contact/control between the robot end-effector or held tool and the relevant object/surface? Penalize hovering, empty-space contact, disconnected motion, and object motion without contact.

C. manipulation_mode_transfer
Is the functional manipulation mode transferred, such as grasp, pinch, support, push, pull, press, rotate, insert, connect, detach, wipe, pour, fold, spread, scoop, or stabilize?

D. object_response_causality
Does the expected object/tool/surface response visibly follow robot contact or control, consistently with the source interaction? Assess visual-temporal support, not physical causation. Penalize unsupported changes such as teleporting, floating, duplicating, penetrating, or disappearing.

E. embodiment_compatible_contact
Is the contact strategy feasible for the target embodiment, using the constraints in the contact spec? For example, a parallel-jaw gripper should grasp, press, or push using gripper-like contacts; a dexterous hand may use articulated fingers. Do not require perfect visual morphology, but penalize contact behavior that only a human hand could perform.

Scoring:
0 = no transfer or completely wrong.
1 = weak hint, mostly wrong.
2 = partial transfer with important missing/unstable/incorrect conditions.
3 = mostly successful transfer with minor ambiguity or physical/contact issues.
4 = clear successful contact transfer with visible functionally equivalent contact and a temporally supported object response.

Do not use 2 or 3 as a default compromise. Choose based on visible evidence.
If the generated video is too blurry to verify contact, assign a lower score and explain the ambiguity.

Input contact specification:
{case_payload_json}

Return JSON in this exact schema:
{response_schema}
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


def validate_h2r_contact_output(obj: dict[str, Any], max_frame: int, dimensions=None) -> dict[str, Any]:
    dimensions = normalize_dimensions(DIMENSIONS if dimensions is None else dimensions)
    if not isinstance(obj, dict):
        raise ValueError("judge output must be a JSON object")
    root = obj.get("h2r_affordance_contact_transfer")
    if not isinstance(root, dict):
        raise ValueError("missing h2r_affordance_contact_transfer object")
    clean_root: dict[str, Any] = {}
    for dim in dimensions:
        item = root.get(dim)
        if not isinstance(item, dict):
            raise ValueError(f"missing dimension object: {dim}")
        score = _coerce_score(item.get("score_0_to_4"))
        clean_root[dim] = {
            "score_0_to_4": score,
            "evidence_source_frames": require_evidence(item, "evidence_source_frames", max_frame, positive=score > 0),
            "evidence_generated_frames": require_evidence(item, "evidence_generated_frames", max_frame, positive=score > 0),
            "reason": require_reason(item),
        }
    hard_failure = obj.get("source_grounding_hard_failure")
    if not isinstance(hard_failure, dict):
        raise ValueError("source_grounding_hard_failure must be an object")
    triggered = hard_failure.get("triggered")
    if not isinstance(triggered, bool):
        raise ValueError("source_grounding_hard_failure.triggered must be boolean")
    return {
        "source_grounding_hard_failure": {
            "triggered": triggered,
            "reason": require_reason(hard_failure),
            "evidence_generated_frames": require_evidence(
                hard_failure, "evidence_generated_frames", max_frame, positive=triggered),
        },
        "h2r_affordance_contact_transfer": clean_root,
        "overall_reason": str(obj.get("overall_reason", "")),
    }


def compute_provider_h2r_score(clean_obj: dict[str, Any], dimensions=None) -> dict[str, Any]:
    dimensions = normalize_dimensions(DIMENSIONS if dimensions is None else dimensions)
    root = clean_obj["h2r_affordance_contact_transfer"]
    hard_failure = clean_obj["source_grounding_hard_failure"]
    per_dimension: dict[str, Any] = {}
    normalized_scores: list[float] = []
    for dim in dimensions:
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
        "valid_dimensions": dimensions,
        "per_dimension": per_dimension,
    }
