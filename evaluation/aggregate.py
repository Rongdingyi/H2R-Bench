"""Join M1–M4 and M5 by exact case ID and summarize each model/embodiment."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from evaluation.common import COVERAGE_METRICS, FRAME_COUNTS, JUDGE_MODELS, PROTOCOL, fingerprint, read_jsonl, write_jsonl
import json

WEIGHTS = {"M1": .15, "M2": .15, "M3": .30, "M4": .30, "M5": .10}
JUDGED_METRICS = ("M1", "M2", "M3", "M4", *COVERAGE_METRICS)


def normalized_score(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid or missing {label}: {value!r}")
    return value


def evaluation_profile(row):
    if row.get("protocol") != PROTOCOL or row.get("frame_counts") != FRAME_COUNTS:
        raise ValueError("Scores must use the current 25-frame evaluation protocol")
    models, providers = row.get("judge_models"), row.get("providers")
    if not isinstance(models, dict) or not models or not all(isinstance(v, str) and v for v in models.values()):
        raise ValueError("Missing judge model IDs")
    if not isinstance(providers, dict) or set(providers) != set(models):
        raise ValueError("Judge model IDs and provider scores must match")
    if type(row.get("temperature")) not in (int, float) or row["temperature"] != 0:
        raise ValueError("Evaluation requires temperature 0")
    for metric in JUDGED_METRICS:
        values = []
        for name, scores in providers.items():
            if not isinstance(scores, dict):
                raise ValueError(f"Invalid provider scores: {name}")
            values.append(normalized_score(scores.get(metric), f"{name}/{metric}"))
        value = normalized_score(row.get(metric), metric)
        if not math.isclose(value, mean(values), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{metric} must equal the arithmetic mean across judges")
    return dict(protocol=row["protocol"], frame_counts=row["frame_counts"],
                judge_models=models, temperature=row["temperature"])


def core(values):
    for metric in WEIGHTS:
        if values.get(metric) is not None:
            normalized_score(values[metric], metric)
    if any(values.get(m) is None for m in WEIGHTS):
        return None
    return 100 * sum(WEIGHTS[m] * values[m] for m in WEIGHTS)


def aggregate(rows, m5_rows):
    if not rows:
        raise ValueError("Empty score file")
    profiles = [evaluation_profile(row) for row in rows]
    if len({fingerprint(profile) for profile in profiles}) != 1:
        raise ValueError("Cannot mix evaluation protocols or judge configurations")
    profile = profiles[0]
    if len({r["id"] for r in rows}) != len(rows) or len({r["id"] for r in m5_rows}) != len(m5_rows):
        raise ValueError("Duplicate IDs; aggregate one generated-model manifest at a time")
    quality = {r["id"]: r for r in m5_rows}
    if set(quality) - {r["id"] for r in rows}:
        raise ValueError("M5 contains IDs outside the M1–M4 population")
    if quality and set(quality) != {r["id"] for r in rows}:
        raise ValueError("M5 must cover the same video population as M1–M4; finish M5 or omit it")
    for row in m5_rows:
        normalized_score(row.get("m5"), "M5")
        if not isinstance(row.get("run_signature"), str) or not row["run_signature"]:
            raise ValueError("M5 is missing its run signature")
    if len({row["run_signature"] for row in m5_rows}) > 1:
        raise ValueError("Cannot mix different M5 configurations")
    output, groups = [], defaultdict(list)
    for r in rows:
        q = quality.get(r["id"])
        if not isinstance(r.get("generated_video_sha256"), str) or not r["generated_video_sha256"]:
            raise ValueError("Missing generated video hash")
        if q and q.get("video_sha256") != r.get("generated_video_sha256"):
            raise ValueError(f"M5 video hash mismatch: {r['id']}")
        r = dict(r, M5=q["m5"] if q else None)
        r["H2RCore"] = core(r)
        output.append(r)
        groups[(r["model"], r["target_embodiment"])].append(r)
    summary = []
    for (model, target), items in sorted(groups.items()):
        row = dict(model=model, target_embodiment=target, n=len(items),
                   protocol=profile["protocol"],
                   judge_models=json.dumps(profile["judge_models"], sort_keys=True),
                   paper_judges=profile["judge_models"] == JUDGE_MODELS)
        for metric in [*WEIGHTS, *COVERAGE_METRICS]:
            values = [] if metric == "M5" and not quality else [r[metric] for r in items]
            row[metric] = mean(values) if values else None
            row["n_" + metric] = len(values)
        row["H2RCore"] = core(row)
        row["n_H2RCore"] = len(items) if row["H2RCore"] is not None else 0
        summary.append(row)
    return output, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--m5", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows, summary = aggregate(read_jsonl(args.scores), read_jsonl(args.m5) if args.m5 else [])
    if not summary:
        raise ValueError("Empty score file")
    write_jsonl(args.output_dir / "per_video.jsonl", rows)
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    print(f"Aggregated {len(rows)} videos into {len(summary)} model/embodiment groups")


if __name__ == "__main__":
    main()
