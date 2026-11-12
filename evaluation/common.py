"""Small IO utilities shared by the evaluation commands."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FRAME_COUNT = 25
PROTOCOL = "paper-25frames-v4"
JUDGE_MODELS = {"qwen": "qwen3.7-plus", "gemini": "gemini-3.5-flash", "gpt": "gpt-5.4"}
FRAME_COUNTS = dict.fromkeys(("M1_M2", "M3", "M4"), EVALUATION_FRAME_COUNT)
FAMILIES = {
    "F1": "F1_rigid_transport_rearrangement",
    "F2": "F2_mechanism_articulated_change",
    "F3": "F3_insertion_connection_disassembly",
    "F4": "F4_deformable_configuration",
    "F5": "F5_bulk_content_transfer_mixing",
    "F6": "F6_surface_material_transformation",
}
COVERAGE_METRICS = ("M1_coverage_at_3", "M1_coverage_at_4",
                    "M2_coverage_at_3", "M2_coverage_at_4")


def benchmark_family(record):
    """The benchmark partition is defined by the scene ID, not LLM labels."""
    scene = str(record.get("scene_id") or record.get("case_id") or "")
    return FAMILIES.get(scene.split("_", 1)[0], record.get("family"))


def require_reason(item, field="reason", *, allow_empty=False):
    reason = item.get(field)
    if not isinstance(reason, str) or (not allow_empty and not reason.strip()):
        raise ValueError(f"{field} must contain a textual rationale")
    return reason


def require_evidence(item, field, max_frame=EVALUATION_FRAME_COUNT, *, positive=False):
    frames = item.get(field)
    if not isinstance(frames, list):
        raise ValueError(f"{field} must be an explicit list of frame indices")
    if any(type(frame) is not int or not 1 <= frame <= max_frame for frame in frames):
        raise ValueError(f"{field} must contain integer indices in 1..{max_frame}")
    if positive and not frames:
        raise ValueError(f"{field} must cite evidence for a positive score")
    return list(dict.fromkeys(frames))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
