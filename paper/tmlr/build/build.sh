#!/bin/bash
# Full build of the TMLR manuscript of paper A (anonymous submission PDF, main.pdf):
#   numbers -> tables -> figures -> number check -> tectonic -> report (pages, where the references start,
#   \todo markers, anonymity of the PDF text and metadata).
# Usage: bash build/build.sh             draft build: stops if numbers.tex is stale or a wording check fails
#        bash build/build.sh --strict    submission build: also stops on unchecked literal numbers, report-only
#                                        values and remaining \todo markers
# The archive copy (~/Documents/research/papers/Dithered_Low-Bit_World_Models/TMLR/) is not touched here; run
# build/sync_papers_copy.sh once the user has approved creating that folder (PLAN_tmlr.md 5.2).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 build/make_numbers.py
python3 build/make_tables.py
python3 build/make_figs.py
python3 build/verify_numbers.py ${1:-} --todos
tectonic -X compile --keep-logs main.tex
rm -f main.aux main.bbl main.blg main.out main.xdv   # in case an earlier run kept intermediates

python3 -I - main.pdf <<'PY'
import sys
import pymupdf
doc = pymupdf.open(sys.argv[1])
n = len(doc)
ref_page = ref_y = bis_page = None
for i, page in enumerate(doc):
    for b in page.get_text("blocks"):
        t = b[4].strip()
        if ref_page is None and t == "References":
            ref_page, ref_y = i + 1, b[1]
        if bis_page is None and t.startswith("Broader Impact Statement"):
            bis_page = i + 1
# text block of a TMLR page: 9 in (648 pt) high; on these pages its first line starts at about y = 84 pt and its
# last line ends at about y = 732 pt (the running head and the page number lie outside).
top, bottom = 84.0, 732.0
frac = lambda y: max(0.0, min(1.0, (y - top) / (bottom - top)))
# end of the main text: the lowest text block above the References heading (Broader Impact Statement included)
end_page = end_y = None
for i in range(ref_page - 1, -1, -1):
    ys = [b[3] for b in doc[i].get_text("blocks") if b[4].strip() and 60 < b[1] < 745
          and (i + 1 < ref_page or b[1] < ref_y - 1)]
    if ys:
        end_page, end_y = i + 1, max(ys)
        break
print(f"\nPDF: {n} pages. Main text (Broader Impact Statement on page {bis_page} included) ends on page {end_page} "
      f"at {frac(end_y):.0%} of the text height = {end_page - 1 + frac(end_y):.2f} pages (target <= 12 for a Regular "
      f"submission); References start on page {ref_page} at {frac(ref_y):.0%}.")
text = "".join(p.get_text() for p in doc)
squash = lambda t: "".join(t.split()).lower()
import os
_priv = os.path.join("build", "private_patterns.txt")   # local, git-ignored extra strings (kind "pdf")
extra = [ln.split("\t", 1)[1].strip() for ln in open(_priv, encoding="utf-8")
         if ln.startswith("pdf\t")] if os.path.exists(_priv) else []
bad = [s for s in ("Yong Hwi", "YongHwi", "Keon Myung", "Chungbuk", "cbnu", "thddydgnl", "kiis", "Tailscale",
                   "JKAIA", "/Users/", "github.com/thddydgnl", *extra) if squash(s) in squash(text)]
meta = {k: v for k, v in doc.metadata.items() if v}
bad += [f"metadata {k}={v!r}" for k, v in meta.items() if k in ("author",) or
        any(squash(s) in squash(v) for s in ("Yong Hwi", "Keon Myung", "Chungbuk", "thddydgnl"))]
print("anonymity:", "OK (no author, school, server or repository strings; empty PDF author)" if not bad else f"FOUND {bad}")
sys.exit(1 if bad else 0)
PY
