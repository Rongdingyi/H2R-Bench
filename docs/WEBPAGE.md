# H2R-Bench project page

Static project page for **H2R-Bench: Benchmarking Human-to-Robot Manipulation Video
Generation in World Models**.

Built the same way as https://zhenjiemao.github.io/ReRe/ — a single hand-written
`index.html` on the [Nerfies](https://github.com/nerfies/nerfies.github.io) academic
project-page template: Bulma CSS + Font Awesome + Academicons, no build step, no Jekyll,
no framework. All page-specific styling lives in one `<style>` block in the `<head>`.

## Layout

```
web/
├── index.html                 the whole page
├── .nojekyll                  tell GitHub Pages to serve files as-is
└── static/
    ├── css/                   bulma, bulma-carousel, bulma-slider, fontawesome, index.css
    ├── js/                    jquery is loaded from a CDN; the rest are local
    └── images/                figures exported from papers/Figures/
```

`static/images/` currently holds: `overview.png` (teaser), `m5_vs_h2rcore.png`,
`diagnostic_failures.png`, `taskwise_h2rcore.png`, `human_mllm_spearman.png`,
`qualitative.png`, `human_eval_interface.png`, `task_attribute_breakdown.png`.
All numbers in the leaderboard table were copied from
`papers/H2RBench_arxiv.tex` (`tab:main_results_by_embodiment`) and verified against it,
including the bold / underlined best and second-best marks.

## Preview locally

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

## Publish on GitHub Pages

1. Push the contents of this directory to a repo — either as the repo root of
   `<user>.github.io`, or to a `gh-pages` branch, or to `docs/` on `main`.
2. Settings → Pages → pick that branch/folder.

## Before publishing: fill in the placeholders

`index.html` still contains literal `ARXIV_ID` and `ORG` placeholders:

- `https://arxiv.org/pdf/ARXIV_ID` and `https://arxiv.org/abs/ARXIV_ID` — the two hero buttons
- `https://github.com/ORG/H2R-Bench` — the Code button
- `https://huggingface.co/datasets/ORG/H2R-Bench` — the Dataset button
- `arXiv:ARXIV_ID` inside the BibTeX block

Drop the Code / Dataset buttons entirely if there is nothing public to point at yet.

## Refreshing a figure

Figures are PNG exports of the paper's assets. `papers/Figures/` already ships a `.png`
next to most `.pdf`s; for PDF-only ones:

```bash
pdftocairo -png -r 150 -singlefile papers/Figures/qualitative.pdf web/static/images/qualitative
```

## Not yet included

- A **teaser comparison figure** matching `fig:h2r_motivation` in the paper — that figure
  (`Figures/titu`) does not exist in the repo yet, so the page uses `overview.png` as its
  hero image instead.
- **Video samples.** The template's `bulma-carousel` is already loaded, so a
  `results-carousel` of source/generated `<video>` pairs can be added without new
  dependencies. Generated clips live under `predict_videos/`.
