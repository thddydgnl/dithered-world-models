# Pack the arXiv upload bundle from the manuscript source.
#  - resolves the \ifdefined\arxivversion switches to the arXiv branch and the \ifXeTeX switch to the pdfLaTeX branch,
#    so that the flattened main.tex compiles with pdfLaTeX alone (arXiv runs pdfLaTeX, not XeTeX);
#  - drops full-line comments (the source becomes public on arXiv);
#  - copies the .bbl (arXiv does not run BibTeX), the class file and the figure PDFs that the text includes;
#  - writes arxiv_source.tar.gz and arxiv_metadata.txt (title, authors, abstract, comments for the arXiv form).
# Usage: python3 build/make_arxiv.py   (after: tectonic -X compile --keep-intermediates arxiv.tex)
import os
import re
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
RSS = os.path.abspath(os.path.join(HERE, ".."))
SRC_DIR = os.path.join(RSS, "arxiv_src")
TAR = os.path.join(RSS, "arxiv_source.tar.gz")
META = os.path.join(RSS, "arxiv_metadata.txt")


def resolve(lines, head, keep_true):
    """Replace every block that starts with the line `head` (an \\if...) by one of its branches.
    Inner conditionals must sit on their own lines; they are tracked by depth."""
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() != head:
            out.append(lines[i])
            i += 1
            continue
        depth, true_part, false_part, in_else = 0, [], [], False
        i += 1
        while True:
            s = lines[i].strip()
            if depth == 0 and s == "\\fi":
                break
            if depth == 0 and s == "\\else":
                in_else = True
                i += 1
                continue
            if s.startswith("\\if"):
                depth += 1
            elif s.startswith("\\fi"):
                depth -= 1
            (false_part if in_else else true_part).append(lines[i])
            i += 1
        out.extend(true_part if keep_true else false_part)
        i += 1
    return out


def detex(t):
    t = t.replace("$\\times$", "\u00d7").replace("\\%", "%").replace("--", "\u2013").replace("~", " ")
    t = re.sub(r"\\(?:emph|textbf|textit)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\\\", " ").replace("\\ ", " ")
    assert "\\" not in t, "unhandled TeX in: " + t
    return re.sub(r"\s+", " ", t).strip()


def main():
    lines = open(os.path.join(RSS, "main.tex")).read().split("\n")
    lines = resolve(lines, "\\ifdefined\\arxivversion", keep_true=True)
    lines = resolve(lines, "\\ifXeTeX", keep_true=False)
    lines = [l for l in lines if not l.lstrip().startswith("%")]
    tex = "\n".join(lines)
    for bad in ("\\ifdefined", "\\ifXeTeX", "fontspec", "Anonymous", "Paper-ID", "RSS", "Stage 2", "this submission"):
        assert bad not in tex, f"{bad!r} is still in the flattened source"
    names = re.findall(r"\\IEEEauthorblockN\{([^}]*)\}", tex)
    assert len(names) >= 2, "author block missing"
    bbl = os.path.join(RSS, "arxiv.bbl")
    if not os.path.exists(bbl):
        sys.exit("arxiv.bbl is missing: run  tectonic -X compile --keep-intermediates arxiv.tex  first")

    if os.path.isdir(SRC_DIR):
        shutil.rmtree(SRC_DIR)
    os.makedirs(os.path.join(SRC_DIR, "figures"))
    open(os.path.join(SRC_DIR, "main.tex"), "w").write(tex)
    shutil.copy(bbl, os.path.join(SRC_DIR, "main.bbl"))
    shutil.copy(os.path.join(RSS, "IEEEtran.cls"), SRC_DIR)
    figs = re.findall(r"\\includegraphics\[[^\]]*\]\{(figures/[^}]*)\}", tex)
    for f in figs:
        shutil.copy(os.path.join(RSS, f), os.path.join(SRC_DIR, f))
    members = ["main.tex", "main.bbl", "IEEEtran.cls"] + figs
    with tarfile.open(TAR, "w:gz") as tar:
        for m in members:
            tar.add(os.path.join(SRC_DIR, m), arcname=m)

    title = detex(re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1))
    affils = [a.split("\\\\")[0].strip() for a in re.findall(r"\\IEEEauthorblockA\{(.*?)\}", tex, re.S)]
    abstract = detex(re.search(r"\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}", tex, re.S).group(1))
    n_tab = tex.count("\\begin{table")
    pages = "?"
    try:
        import pymupdf
        pages = len(pymupdf.open(os.path.join(RSS, "arxiv.pdf")))
    except Exception:
        pass
    with open(META, "w") as f:
        f.write("arXiv 제출 양식에 넣을 내용 (build/make_arxiv.py가 main.tex에서 만든다)\n\n")
        f.write("Title:\n" + title + "\n\n")
        f.write("Authors:\n" + ", ".join(names) + "\n")
        for n, a in zip(names, affils):
            f.write(f"  {n} -- {a}, Chungbuk National University\n")
        f.write("\nAbstract:\n" + abstract + "\n\n")
        f.write(f"Comments (추천):\n{pages} pages, {len(figs)} figures, {n_tab} table\n\n")
        f.write("Categories (추천):\nPrimary cs.RO (Robotics); cross-list cs.LG (Machine Learning)\n\n")
        f.write("메모\n"
                "- 올릴 파일: Dithered_Low-Bit_World_Models_arXiv_source.tar.gz (main.tex, main.bbl, IEEEtran.cls, figures/). arXiv가 pdfLaTeX로 직접 컴파일한다. 올린 뒤 arXiv의 컴파일 결과(미리보기 PDF)를 확인하고 공개(announce)한다.\n"
                "- Comments 칸과 본문 어디에도 투고처(RSS 2027)를 쓰지 않는다. 이중맹검 심사 중 CV에서 RSS 표기를 빼기로 한 결정(D4)과 같은 이유다.\n"
                "- 라이선스: 기본값(arXiv.org perpetual, non-exclusive license)이면 충분하다. CC 라이선스는 되돌릴 수 없으니 학회 정책을 확인한 뒤에만 고른다.\n"
                "- cs 분야에 처음 올리는 계정이면 endorsement가 필요할 수 있다.\n"
                "- 위 쪽수는 맥북(tectonic) 빌드 기준이다. arXiv(pdfLaTeX, Times)에서는 한 줄 차이로 쪽수가 달라질 수 있다.\n"
                "- arXiv ID를 받으면 원고_갱신기록.txt의 메모 칸에 적는다.\n")
    print(f"arXiv bundle: {TAR} ({', '.join(members)}); metadata: {META}; {pages} pages")


if __name__ == "__main__":
    main()
