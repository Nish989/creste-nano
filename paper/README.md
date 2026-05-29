# CREStE-Nano — Conference paper

IEEE conference-format (6 pages) technical paper summarising the project.

## Files

- `main.tex` — the paper source
- `references.bib` — BibTeX bibliography
- (figures pulled from `../docs/architecture.pdf` and `../docs/plots/*.png`)

## How to compile

### Path A — Overleaf (zero install, recommended)

1. Go to [overleaf.com](https://www.overleaf.com) and sign in / create a free account.
2. **New Project → Upload Project** and drag in this whole `paper/` folder.
3. Also upload `docs/architecture.pdf` and the four files in `docs/plots/*.png`
   so the `\includegraphics{...}` paths resolve. (Easiest way: zip the whole
   repo and upload everything.)
4. Click **Recompile**. Overleaf has `IEEEtran.cls` built in.

### Path B — Local TeX Live (BasicTeX is ~100 MB)

```bash
brew install basictex
eval "$(/usr/libexec/path_helper)"   # picks up the new TeX binaries
sudo tlmgr update --self
sudo tlmgr install ieeetran cite booktabs multirow url hyperref \
                   amsmath amssymb amsfonts algorithmic
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
open main.pdf
```

That produces `main.pdf` next to `main.tex`.

## Page budget

Currently sized for 6 pages in IEEEtran conference format
(2-column, 10 pt). If a class assignment requires a different length,
tighten or expand `Related Work` (§II), `Experiments` (§V), and
`Discussion` (§VII) — the body of `Method` (§IV) is dense and harder
to compress without losing the math.

## Editing tips

- Author block at the top of `main.tex` — change institution and email.
- Section headings are standard `\section{...}` — no custom macros needed.
- Figure paths are relative to `paper/`, so `../docs/architecture.pdf`
  resolves to the repo's `docs/architecture.pdf` when you compile from
  inside `paper/`. If you compile from the repo root instead, drop the
  `../` prefix from each `\includegraphics{...}` line.
