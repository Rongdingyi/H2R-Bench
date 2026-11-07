"""Compute the four-component M5 aggregate using an external VBench checkout."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation.common import write_json, write_jsonl
from evaluation.environment import environment_info

VBENCH_COMMIT = '45e79ec14e69a2187202c675d2dbce1a71843d53'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quality_mean(values):
    if len(values) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError(f'Expected four finite normalized scores in [0,1], got {values}')
    return sum(values)/4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--vbench-root', type=Path, required=True)
    p.add_argument('--weights', type=Path, default=Path.home()/'.cache/vbench')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines() if s.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate case ids')
    files = {
        'musiq': args.weights/'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth',
        'aesthetic': args.weights/'aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth',
        'clip': args.weights/'clip_model/ViT-L-14.pt',
        'amt': args.weights/'amt_model/amt-s.pth',
        'amt_config': args.vbench_root/'pretrained/amt_model/AMT-S.yaml',
    }
    if args.dry_run:
        print(json.dumps({'cases': len(rows), 'vbench_commit': VBENCH_COMMIT,
                          'protocol': 'full-duration, native frame rate',
                          'required_files': {k:str(v) for k,v in files.items()}}, indent=2))
        return
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    commit = subprocess.check_output(['git','-C',str(args.vbench_root),'rev-parse','HEAD'],text=True).strip()
    if commit != VBENCH_COMMIT:
        raise ValueError(f'Use the documented VBench revision {VBENCH_COMMIT}, found {commit}')
    code = {str(p.relative_to(args.vbench_root)): sha256(p)
            for p in sorted((args.vbench_root/'vbench').rglob('*.py'))}
    signature = dict(vbench_commit=commit, vbench_code_sha256=code,
                     weights_sha256={k:sha256(v) for k,v in files.items()},
                     runner_sha256=sha256(Path(__file__)), protocol='native-full-clip',
                     device=args.device, environment=environment_info())
    token = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    write_json(args.output_dir/'run_config.json', signature)
    sys.path.insert(0, str(args.vbench_root.resolve()))
    import torch
    import clip
    from pyiqa.archs.musiq_arch import MUSIQ
    from vbench.motion_smoothness import MotionSmoothness
    from vbench.temporal_flickering import cal_score
    from vbench.imaging_quality import technical_quality
    from vbench.aesthetic_quality import laion_aesthetic, get_aesthetic_model

    torch.manual_seed(20261026)
    musiq = MUSIQ(pretrained_model_path=str(files['musiq'])).to(args.device).eval()
    aesthetic = get_aesthetic_model(str(files['aesthetic'].parent)).to(args.device).eval()
    clip_model, _ = clip.load(str(files['clip']), device=args.device)
    motion = MotionSmoothness(str(files['amt_config']), str(files['amt']), args.device)
    output = args.output_dir/'m5.jsonl'
    done = {r['id']:r for r in [json.loads(s) for s in output.read_text().splitlines()]} if output.exists() else {}
    for row in rows:
        video = Path(row['generated_video']).expanduser()
        if not video.is_absolute():
            video = args.manifest.parent/video
        video_hash = sha256(video)
        cached = done.get(row['id'], {})
        if cached.get('run_signature') == token and cached.get('video_sha256') == video_hash:
            continue
        with torch.no_grad():
            iq, _ = technical_quality(musiq,[str(video)],args.device)
            aq, _ = laion_aesthetic(aesthetic,clip_model,[str(video)],args.device)
            values = list(map(float,[iq, aq, cal_score(str(video)), motion.motion_score(str(video))]))
        done[row['id']] = dict(id=row['id'], video_sha256=video_hash, run_signature=token,
                              completed_at_utc=datetime.now(timezone.utc).isoformat(),
                              **dict(zip(['imaging_quality','aesthetic_quality','temporal_flickering','motion_smoothness'],values)),
                              m5=quality_mean(values))
        write_jsonl(output, [done[r['id']] for r in rows if r['id'] in done])
        print(row['id'], done[row['id']]['m5'], flush=True)


if __name__ == '__main__':
    main()
