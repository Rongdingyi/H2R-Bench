"""Create a 240-case evaluation manifest from source and generated video folders."""
import argparse
from pathlib import Path
from evaluation.common import ROOT, benchmark_family, read_jsonl, write_jsonl


def prepare(source_dir, generated_dir, model, annotations, allow_missing=False):
    annotations = {r["case_id"]: r for r in read_jsonl(annotations)}
    output = []
    for case in read_jsonl(ROOT / "prompts/generation.jsonl"):
        source = (source_dir / f"{case['scene_id']}.mp4").resolve()
        generated = (generated_dir / case["id"] / "output.mp4").resolve()
        if not allow_missing:
            for p in (source, generated):
                if not p.is_file():
                    raise FileNotFoundError(p)
        annotation = dict(annotations[case["case_id"]])
        annotation["family"] = benchmark_family(case)
        contact_spec = annotation.pop("contact_spec")
        annotation_frames = annotation.pop("contact_annotation_frame_count")
        contact = dict(case_id=case["case_id"], scene_id=case["scene_id"],
            family=benchmark_family(case), source_video=str(source),
            annotation_frame_count=annotation_frames, contact_spec=contact_spec)
        output.append(dict(id=case["id"], case_id=case["case_id"], model=model,
            target_embodiment=case["target_embodiment"], source_video=str(source),
            generated_video=str(generated), annotation=annotation, contact=contact))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=ROOT / "benchmark/videos")
    p.add_argument("--generated-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--annotations", type=Path, default=ROOT / "benchmark/annotations.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-missing", action="store_true", help="Prepare metadata for offline inspection only.")
    args = p.parse_args()
    rows = prepare(args.source_dir, args.generated_dir, args.model, args.annotations, args.allow_missing)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} source/target cases to {args.output}")


if __name__ == "__main__":
    main()
