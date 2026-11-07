"""Join M1–M4 and M5 by exact case ID and summarize each model/embodiment."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from evaluation.common import read_jsonl, write_jsonl

WEIGHTS = {"M1": .15, "M2": .15, "M3": .30, "M4": .30, "M5": .10}


def core(values):
    if any(values.get(m) is None for m in WEIGHTS):
        return None
    for m in WEIGHTS:
        if not math.isfinite(values[m]) or not 0 <= values[m] <= 1:
            raise ValueError(f"Invalid {m} score: {values[m]}")
    return 100 * sum(WEIGHTS[m] * values[m] for m in WEIGHTS)


def aggregate(rows, m5_rows):
    if len({r["id"] for r in rows}) != len(rows) or len({r["id"] for r in m5_rows}) != len(m5_rows):
        raise ValueError("Duplicate IDs; aggregate one generated-model manifest at a time")
    quality = {r["id"]: r for r in m5_rows}
    if set(quality) - {r["id"] for r in rows}:
        raise ValueError("M5 contains IDs outside the M1–M4 population")
    output, groups = [], defaultdict(list)
    for r in rows:
        q = quality.get(r["id"])
        if q and q.get("video_sha256") != r.get("generated_video_sha256"):
            raise ValueError(f"M5 video hash mismatch: {r['id']}")
        r = dict(r, M5=q["m5"] if q else None)
        r["H2RCore"] = core(r)
        output.append(r)
        groups[(r["model"], r["target_embodiment"])].append(r)
    summary = []
    for (model, target), items in sorted(groups.items()):
        row = dict(model=model, target_embodiment=target, n=len(items))
        for metric in [*WEIGHTS, "H2RCore"]:
            values = [r[metric] for r in items if r.get(metric) is not None]
            row[metric] = mean(values) if values else None
            row["n_" + metric] = len(values)
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
