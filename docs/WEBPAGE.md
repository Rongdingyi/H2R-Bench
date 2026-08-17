# H2R-Bench project page

Static project page for **H2R-Bench: Benchmarking Human-to-Robot Manipulation
Video Generation in World Models**.

The site is custom, dependency-free HTML/CSS/JavaScript. It does not use Bulma,
jQuery, a carousel package, Jekyll, or a framework, and it has no build step.

## Structure

```text
docs/
├── index.html                 page content, metadata, tables, and figures
├── .nojekyll                  GitHub Pages passthrough marker
└── static/
    ├── css/index.css          complete visual and responsive system
    ├── js/index.js            navigation, active section, BibTeX copy
    └── images/
        ├── h2r/               optimized paper figures
        ├── favicon.png
        └── og-h2r.jpg         social preview derived from the paper overview
```

The visual design mirrors the benchmark rather than a generic paper template:
orange denotes source/contact evidence, cyan denotes robot/verified evidence,
and the information flow follows Observe → Retarget → Verify. The page includes
the paper's motivation, benchmark comparison, pipeline, 120-scene coverage,
metric design, complete leaderboard, embodiment/task diagnostics, matched
source-conditioning ablation, qualitative examples, human alignment, and scope.

## Responsive behavior

Layouts adapt at 1120, 920, 620, and 390 px. Navigation becomes a keyboard- and
touch-friendly menu; buttons and cards reflow; leaderboard and benchmark tables
support inertial horizontal scrolling; information-dense figures can be panned
at a readable scale instead of being shrunk into illegibility. Reduced-motion
preferences are respected.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000`.

## Paper-sourced assets

Web figures in `static/images/h2r/` are optimized exports from the arXiv source
package. Keep captions, metrics, model names, and numerical results synchronized
with the paper when refreshing an asset. Use a lossless/high-resolution render
as the intermediate, then export a browser-friendly WebP. The social image is
the paper overview, resized as JPEG for broad unfurl compatibility.

## Publishing

GitHub Pages should serve the `docs/` folder on `main`. The canonical, Open
Graph, arXiv, Hugging Face, and repository URLs are already set in `index.html`.
