"""Run M1–M4 with 25 uniformly sampled frames per input video."""
import argparse
import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from evaluation import contact, embodiment, goal_action
from evaluation.common import COVERAGE_METRICS, FRAME_COUNTS, JUDGE_MODELS, PROTOCOL, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from evaluation.environment import environment_info

DEFAULT_MODELS = JUDGE_MODELS


def frame_counts(record):
    # Annotation provenance does not change the evaluation evidence budget.
    return FRAME_COUNTS.copy()


def provider_config(name):
    if name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown provider: {name}")
    prefix = "H2R_" + name.upper()
    return dict(name=name, model=os.getenv(prefix + "_MODEL", DEFAULT_MODELS[name]),
        base_url=os.getenv(prefix + "_BASE_URL", ""), key_env=prefix + "_API_KEY")


def make_prompts(record):
    ann = goal_action.normalize_annotation(record["annotation"])
    payload = contact.build_case_payload(record["contact"], record)
    return {
        "M1_M2": goal_action.build_combined_goal_action_prompt(ann),
        "M3": contact.build_h2r_contact_prompt(json.dumps(payload, ensure_ascii=False, indent=2),
            FRAME_COUNTS["M3"], payload["m3_applicable_subdimensions"]),
        "M4": embodiment.build_embodiment_prompt(record),
    }


def validate_and_score(metric, obj, record):
    if metric == "M1_M2":
        ann = goal_action.normalize_annotation(record["annotation"])
        clean = goal_action.clean_combined_judge_output(ann, obj)
        goal = goal_action.compute_goal_score(ann, clean)
        action = goal_action.compute_action_score(ann, clean)
        return clean, {"M1": goal["score"], "M2": action["score"],
                       "M1_coverage_at_3": goal["goal_coverage_at_3"],
                       "M1_coverage_at_4": goal["goal_coverage_at_4"],
                       "M2_coverage_at_3": action["event_coverage_at_3"],
                       "M2_coverage_at_4": action["event_coverage_at_4"]}
    if metric == "M3":
        if not isinstance(obj.get("source_grounding_hard_failure"), dict) or type(obj["source_grounding_hard_failure"].get("triggered")) is not bool:
            raise ValueError("Missing source-grounding decision")
        dimensions = contact.applicable_dimensions(record["contact"])
        clean = contact.validate_h2r_contact_output(obj, FRAME_COUNTS["M3"], dimensions)
        return clean, {"M3": contact.compute_provider_h2r_score(clean, dimensions)["score"]}
    clean = embodiment.validate_embodiment_output(obj)
    return clean, {"M4": embodiment.compute_provider_embodiment_score(clean, record["target_embodiment"])["score"]}


def sample_frames(video, count):
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or total <= 0:
            raise ValueError(f"Cannot decode {video}")
        indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
        images = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}: {video}")
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("JPEG encoding failed")
            images.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
        return indices, images
    finally:
        cap.release()


def call_judge(config, prompt, sequences, timeout):
    from openai import OpenAI
    key = os.getenv(config["key_env"])
    if not key or not config["base_url"]:
        raise ValueError(f"Set {config['key_env']} and H2R_{config['name'].upper()}_BASE_URL")
    content = [{"type": "text", "text": prompt}]
    for prefix, images in sequences:
        for i, image in enumerate(images, 1):
            content.extend([{"type": "text", "text": f"{prefix}frame_{i:02d}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}}])
    with OpenAI(api_key=key, base_url=config["base_url"], timeout=timeout, max_retries=0) as client:
        response = client.chat.completions.create(model=config["model"],
            messages=[{"role": "user", "content": content}], temperature=0)
    return response.choices[0].message.content or ""


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("Expected one JSON object")
    return obj


def judge_record(record, providers, output_dir, retries=3, timeout=180):
    prompts = make_prompts(record)
    counts = frame_counts(record)
    media_hashes = {k: sha256(record[k]) for k in ("source_video", "generated_video")}
    code_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    images = {}
    results = {}
    for config in providers:
        scores = {}
        for metric, prompt in prompts.items():
            signature = fingerprint(dict(protocol=PROTOCOL, prompt=prompt, provider=config,
                media=media_hashes, code=code_hashes, annotation=record["annotation"],
                target=record["target_embodiment"], frames=counts[metric]))
            cache = output_dir / "judgments" / record["id"] / config["name"] / f"{metric}.json"
            if cache.exists():
                cached = json.loads(cache.read_text())
                if cached.get("signature") == signature:
                    _, values = validate_and_score(metric, cached["judgment"], record)
                    scores.update(values)
                    continue
            count = counts[metric]
            kinds = ["source_video", "generated_video"] if metric == "M3" else ["generated_video"]
            sequences, indices = [], {}
            for kind in kinds:
                key = (kind, count)
                if key not in images:
                    images[key] = sample_frames(record[kind], count)
                indices[kind], encoded = images[key]
                sequences.append((("source_" if kind == "source_video" else "generated_") if metric == "M3" else "", encoded))
            for attempt in range(1, retries + 1):
                raw = ""
                try:
                    raw = call_judge(config, prompt, sequences, timeout)
                    clean, values = validate_and_score(metric, parse_response(raw), record)
                    write_json(cache, dict(signature=signature, protocol=PROTOCOL, provider=config,
                        prompt=prompt, media_sha256=media_hashes, sampled_frame_indices=indices,
                        raw_text=raw, judgment=clean, scores=values, attempt=attempt,
                        completed_at_utc=datetime.now(timezone.utc).isoformat()))
                    scores.update(values)
                    break
                except Exception as exc:
                    # Preserve failed model replies without logging API headers or credentials.
                    write_json(cache.with_name(f"{metric}.attempt{attempt}.failed.json"),
                        dict(signature=signature, raw_text=raw, error_type=type(exc).__name__, attempt=attempt))
                    if attempt == retries:
                        raise RuntimeError(f"{record['id']} / {config['name']} / {metric}: {type(exc).__name__}") from None
                    time.sleep(min(attempt * 2, 10))
        results[config["name"]] = scores
    return dict(id=record["id"], case_id=record["case_id"], model=record["model"],
        target_embodiment=record["target_embodiment"], protocol=PROTOCOL,
        generated_video_sha256=media_hashes["generated_video"],
        frame_counts=counts,
        annotation_version=record["annotation"]["annotation_version"], providers=results,
        judge_models={config["name"]: config["model"] for config in providers},
        temperature=0,
        **{m: mean(v[m] for v in results.values()) for m in ("M1", "M2", "M3", "M4", *COVERAGE_METRICS)})


def load_manifest(path):
    rows = read_jsonl(path)
    ids = [r["id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty manifest or duplicate IDs")
    for r in rows:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", r["id"]) or r["id"] in {".", ".."}:
            raise ValueError("Unsafe case ID")
        if r["target_embodiment"] not in {"mechanical_arm", "dexterous_hand"}:
            raise ValueError("Unsupported target embodiment")
        for key in ("annotation", "contact"):
            if r[key]["case_id"] != r["case_id"]:
                raise ValueError("Case/annotation mismatch")
        for field in ("source_video", "generated_video"):
            p = Path(r[field]).expanduser()
            r[field] = str((path.parent / p).resolve()) if not p.is_absolute() else str(p)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--providers", default="qwen,gemini,gpt")
    p.add_argument("--limit", type=int)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.retries < 1 or args.timeout <= 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("Retries, timeout and limit must be positive")
    names = [n.strip() for n in args.providers.split(",") if n.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("Select at least one distinct provider")
    providers = [provider_config(n) for n in names]
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    if args.dry_run:
        for r in rows:
            make_prompts(r)
        print(json.dumps(dict(cases=len(rows), protocol=PROTOCOL,
            frame_counts=FRAME_COUNTS,
            providers=providers, first_case=rows[0]["id"], prompts=make_prompts(rows[0])), indent=2))
        return
    for c in providers:
        if not c["base_url"] or not os.getenv(c["key_env"]):
            raise ValueError(f"Configure H2R_{c['name'].upper()}_BASE_URL and {c['key_env']}")
    for r in rows:
        for field in ("source_video", "generated_video"):
            if not Path(r[field]).is_file():
                raise FileNotFoundError(r[field])
    write_json(args.output_dir / "run_config.json", dict(protocol=PROTOCOL, providers=providers,
        frame_counts=FRAME_COUNTS,
        manifest_sha256=sha256(args.manifest),
        environment=environment_info(), max_attempts=args.retries, temperature=0))
    completed = []
    for r in rows:
        completed.append(judge_record(r, providers, args.output_dir, args.retries, args.timeout))
        write_jsonl(args.output_dir / "scores.jsonl", completed)
        print(f"{len(completed)}/{len(rows)} {r['id']}", flush=True)


if __name__ == "__main__":
    main()
