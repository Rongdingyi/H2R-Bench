# H2R-Bench

**Benchmarking Human-to-Robot Manipulation Video Generation in World Models**

Dingyi Rong<sup>1,2</sup>, Yue Shi<sup>2&dagger;</sup>, Chaofan Ma<sup>1&ddagger;</sup>,
Jiezhang Cao<sup>1</sup>, Zongrui Wang<sup>1,2</sup>, Zeyu Zhang<sup>1,2</sup>,
Yao Mu<sup>1,2</sup>, Guangtao Zhai<sup>1,2&dagger;</sup>, Ning Liu<sup>1&dagger;</sup>

<sup>1</sup>&nbsp;Shanghai Jiao Tong University&nbsp;&nbsp; <sup>2</sup>&nbsp;Shanghai Artificial Intelligence Laboratory

<sup>&dagger;</sup>&nbsp;Corresponding authors.&nbsp;&nbsp; <sup>&ddagger;</sup>&nbsp;Project lead.

[![Project Page](https://img.shields.io/badge/Project-Page-5b9a22?logo=googlechrome&logoColor=white&labelColor=555555)](https://rongdingyi.github.io/H2R-Bench/)
[![Paper](https://img.shields.io/badge/Paper-PDF-d64545?logo=adobeacrobatreader&logoColor=white&labelColor=555555)](https://arxiv.org/pdf/2608.13049)
[![arXiv](https://img.shields.io/badge/arXiv-2608.13049-b31b1b?logo=arxiv&logoColor=white&labelColor=555555)](https://arxiv.org/abs/2608.13049)
[![Hugging Face Papers](https://img.shields.io/badge/Hugging%20Face-Papers-FFD21E?logo=huggingface&logoColor=FFD21E&labelColor=555555)](https://huggingface.co/papers/2608.13049)
[![GitHub](https://img.shields.io/badge/GitHub-H2R--Bench-1683bd?logo=github&logoColor=white&labelColor=555555)](https://github.com/Rongdingyi/H2R-Bench)

---

## About

Large-scale manipulation data is essential for robot learning, yet collecting robot demonstrations remains expensive and difficult to scale. Egocentric human manipulation videos are abundant, but transferring them across embodiments is hard because human hands and robotic end-effectors differ. Video world models offer a possible bridge — H2R-Bench
asks whether they actually cross it.

Given an egocentric human demonstration and a target robot embodiment, a model must generate the corresponding robot manipulation video. H2R-Bench then scores that video along five dimensions:

| | Metric | Weight |
|---|---|---|
| M1 | Goal-state completion | 0.15 |
| M2 | Action-event completion | 0.15 |
| M3 | Functional contact transfer | 0.30 |
| M4 | Embodiment correctness | 0.30 |
| M5 | Video quality (task-agnostic) | 0.10 |

`H2RCore = 100 × (0.15·M1 + 0.15·M2 + 0.30·M3 + 0.30·M4 + 0.10·M5)`

**Scale:** 120 egocentric source clips (EgoDex test split) × 2 target embodiments
(parallel-jaw gripper, dexterous hand) = 240 transfer cases, evenly spread over
6 manipulation families. 11 video generation models evaluated.

**Headline finding:** generic video quality and transfer validity are nearly
rank-independent (Spearman ρ = 0.14). Polished clips routinely retain human hands,
realize the wrong end-effector, or show object changes unsupported by any visible robot
interaction.

## Leaderboard

H2RCore (0–100), higher is better.

| Model | Conditioning | Gripper | Hand |
|---|---|---|---|
| Seedance 2.0 | video | **77.3** | **84.6** |
| Wan2.7 | video | 76.5 | 83.1 |
| Kling-V3 | video | 74.5 | 81.7 |
| Mitty-EPIC14B | frame | 61.5 | 56.1 |
| Grok Imagine Video | frame | 50.1 | 49.2 |
| Veo 3.1 | frame | 49.6 | 57.0 |
| Wan2.2 | frame | 32.4 | 33.7 |
| LTX-2.3 | frame | 32.1 | 39.8 |
| SkyReels-V3-R2V | frame | 31.5 | 34.6 |
| LongCat | frame | 31.0 | 32.0 |
| HunyuanVideo 1.5-I2V | frame | 30.0 | 30.7 |

Full per-metric breakdown on the [project page](https://rongdingyi.github.io/H2R-Bench/).

## Benchmark and prompts

| Resource | Contents |
|---|---|
| [Scene IDs](benchmark/scene_ids.txt) | 120 scenes, 20 per manipulation family |
| [Annotations](benchmark/annotations.jsonl) | Corresponding entities, goal checks, action events and functional-contact specifications |
| [Generation prompts](prompts/generation.jsonl) | 240 prompts: one per scene and target embodiment |
| [Judge prompts](prompts/judging/) | M1–M4 scoring instructions and response formats |
| [Evaluation code](evaluation/) | Frame sampling, judge calls, M1–M5 scoring and aggregation |

Use `scene_id` to match annotations and generation prompts. A generation ID is
`<scene_id>__mechanical_arm` or `<scene_id>__dexterous_hand`; `mechanical_arm`
denotes the parallel-jaw-gripper target. The internal `case_id` appends
`/episode_01` to the scene ID. The six families cover rigid-object rearrangement,
mechanism actuation, insertion/connection, deformable configuration,
bulk-content transfer and surface/material transformation.

Generation prompts are the shared English task prompts for each scene and
target embodiment. Models receive visual inputs through their native interfaces.

## Evaluation

Requires Python 3.10+. Provide source clips as `<source-dir>/<scene_id>.mp4`
and generated clips as `<generated-dir>/<generation_id>/output.mp4`.

```bash
pip install -r requirements.txt

python -B -m evaluation.prepare \
  --source-dir /path/to/source-videos \
  --generated-dir /path/to/generated-videos \
  --model my-model --output outputs/my-model/manifest.jsonl

python -B -m evaluation.run \
  --manifest outputs/my-model/manifest.jsonl \
  --output-dir outputs/my-model/judges --dry-run
```

For actual scoring, remove `--dry-run` after configuring the selected judges.
The client uses OpenAI-compatible Chat Completions endpoints.

| Judge | Default model ID | Required environment variables |
|---|---|---|
| Qwen | `qwen3.7-plus` | `H2R_QWEN_BASE_URL`, `H2R_QWEN_API_KEY` |
| Gemini | `gemini-3.5-flash` | `H2R_GEMINI_BASE_URL`, `H2R_GEMINI_API_KEY` |
| GPT | `gpt-5.4` | `H2R_GPT_BASE_URL`, `H2R_GPT_API_KEY` |

Select judges with `--providers qwen,gemini,gpt`; override model IDs with
`H2R_QWEN_MODEL`, `H2R_GEMINI_MODEL` and `H2R_GPT_MODEL`. IDs reflect the
recorded models; availability depends on the service. Runs save prompts,
raw judgments, frame indices and media hashes, and resume matching cached
responses. Failed requests are retried up to three times; missing scores are
not filled with zero.

The annotations preserve the main experiment's 108 original and 12 corrected
scenes. M1/M2 use 32 frames (M1: final eight) or 24 frames (M1: final six),
respectively; M3 uses 24 source and 24 generated frames, and M4 uses 12 generated
frames. The earlier 32-frame rubric was recovered from the original conversation;
it is not a verified copy of every final API request. The portable runner is
versioned `release-mixed-spec-v1`, with stricter response validation and removal
of annotation-time frame indices from M3 inputs.

M1 and M2 use weighted normalized goal/event scores; action-order diagnostics
are excluded from M2. M3 averages five contact dimensions and applies the
source-grounding failure gate. M4 uses weights 0.20/0.20/0.20/0.25/0.15 for
presence, human absence, category, end effector and structure, with the retained
subtype cap and hard-failure rule. Selected judges are averaged per video.

### Video quality and final scores

M5 averages four normalized VBench components: imaging quality, aesthetic
quality, temporal flickering and motion smoothness, using the full clip at its
native frame rate. Install [VBench](https://github.com/Vchitect/VBench) at commit
`45e79ec14e69a2187202c675d2dbce1a71843d53` with its GPU dependencies and weights.
The dry run below lists the required checkpoints; remove `--dry-run` to compute.

```bash
python -B -m evaluation.video_quality \
  --manifest outputs/my-model/manifest.jsonl \
  --vbench-root /path/to/VBench --weights /path/to/vbench-weights \
  --output-dir outputs/my-model/m5 --dry-run

python -B -m evaluation.aggregate \
  --scores outputs/my-model/judges/scores.jsonl \
  --m5 outputs/my-model/m5/m5.jsonl \
  --output-dir outputs/my-model/summary
```

Aggregation checks case IDs and video hashes, then reports means and valid
counts per model/embodiment. Without M5, H2RCore is left empty. The resulting
weighted score measures visual transfer quality, not physical executability
or joint task-success frequency.

Offline tests (no API calls):

```bash
pip install pytest
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

## Citation

```bibtex
@article{rong2026h2rbench,
  title={H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models},
  author={Rong, Dingyi and Shi, Yue and Ma, Chaofan and Cao, Jiezhang and Wang, Zongrui and Zhang, Zeyu and Mu, Yao and Zhai, Guangtao and Liu, Ning},
  journal={arXiv preprint arXiv:2608.13049},
  year={2026}
}
```
