#!/bin/bash
# Full build: figures -> number checks -> anonymous RSS PDF (main.pdf) -> arXiv PDF (arxiv.pdf) + upload bundle -> archive copies.
# Usage: bash build/build.sh "one-line note for the archive logs"
set -euo pipefail
cd "$(dirname "$0")/.."
python3 build/make_figs.py
python3 build/verify_numbers.py
tectonic -X compile main.tex
tectonic -X compile --keep-intermediates arxiv.tex
python3 build/make_arxiv.py
rm -f arxiv.aux arxiv.bbl arxiv.blg arxiv.log arxiv.out arxiv.xdv   # kept only long enough to pack the .bbl
bash build/sync_papers_copy.sh "${1:-}"
