# H2R-Bench

**Benchmarking Human-to-Robot Manipulation Video Generation in World Models**

Dingyi Rong, Yue Shi, Chaofan Ma, Jiezhang Cao, Zongrui Wang,
Zeyu Zhang, Yao Mu, Guangtao Zhai, Ning Liu

Shanghai Jiao Tong University

[Project page](https://rongdingyi.github.io/H2R-Bench/) &middot;
Paper (arXiv, coming soon)

---

## About

Large-scale manipulation data is essential for robot learning, yet collecting robot
demonstrations remains expensive and difficult to scale. Egocentric human manipulation
videos are abundant, but transferring them across embodiments is hard because human hands
and robotic end-effectors differ. Video world models offer a possible bridge — H2R-Bench
asks whether they actually cross it.

Given an egocentric human demonstration and a target robot embodiment, a model must
generate the corresponding robot manipulation video. H2R-Bench then scores that video
along five dimensions:

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

## Repository layout

```
docs/          project page (served by GitHub Pages)
```

Evaluation code and benchmark annotations are not in this repository yet.

## Citation

```bibtex
@article{rong2026h2rbench,
  title={H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation in World Models},
  author={Rong, Dingyi and Shi, Yue and Ma, Chaofan and Cao, Jiezhang and Wang, Zongrui and Zhang, Zeyu and Mu, Yao and Zhai, Guangtao and Liu, Ning},
  journal={arXiv preprint arXiv:ARXIV_ID},
  year={2026}
}
```
