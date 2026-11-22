"""Helper of build/make_supplement.sh: the anonymized supplementary material of paper A's TMLR submission.

Subcommands (make_supplement.sh calls them with absolute paths):
  trace OUT.json ROOT SCRIPT [ARGS...]       run SCRIPT and record every file under ROOT that it reads or writes, and
                                             every file outside ROOT that is not part of the Python installation
  provenance OUT.json ROOT SCRIPT [ARGS...]  as trace, and attribute the data read to each table or figure written
                                             (values read since the previous output file was written belong to it)
  assemble SNAPSHOT WORK PKG [GIT_SRC]       select, copy and anonymize the package files; patch the hash checks of
                                             make_numbers.py for the redacted design documents; write README.md (with
                                             the design versions; GIT_SRC: a git checkout to find versions changed in place)
  scan PKG OUT.json SNAPSHOT                 search every file of the package (text, .gz, binary, file names) for
                                             identifying strings (author names are also read from SNAPSHOT)
  compare TEST NAME=DIR[:must] ...           compare numbers.tex, tables/ and figures/ of TEST with each DIR
  pdftext A.pdf B.pdf                        compare the text of two PDFs (original against anonymized manuscript)
  report WORK OUT KEY=VALUE ...              write OUT/BUILD_REPORT.md

The identifying strings are written in this file; it is never copied into the package.
"""
import csv
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import runpy
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ================================================================================================ package contents
# Besides every file the four build scripts read (found by tracing them), the package holds the code that produced the
# results, the design documents, the logs and the stage reports (paths relative to the paper-A folder).
INCLUDE = [
    "EXPERIMENT_DESIGN.md",
    "main/DESIGN_main.md", "main/LOG.md", "main/worklogs/*.md", "main/reports/*.md", "main/code/**",
    "rss_ext/DESIGN_*.md", "rss_ext/LOG.md", "rss_ext/*/PLAN.md", "rss_ext/*/LOG.md", "rss_ext/*/REPORT.md",
    "rss_ext/*/code/**",
    "tmlr_ext/DESIGN_tmlr.md", "tmlr_ext/WORKER_RULES.md", "tmlr_ext/LOG.md", "tmlr_ext/RESULTS_tmlr.md",
    "tmlr_ext/*/PLAN.md", "tmlr_ext/*/LOG.md", "tmlr_ext/*/REPORT.md", "tmlr_ext/*/code/**", "tmlr_ext/coordinator/**",
    "paper/tmlr/main.tex", "paper/tmlr/appendix/*.tex", "paper/tmlr/numbers.tex", "paper/tmlr/math_commands.tex",
    "paper/tmlr/tmlr.sty", "paper/tmlr/tmlr.bst", "paper/tmlr/fancyhdr.sty", "paper/tmlr/refs.bib",
    "paper/tmlr/tables/*.tex", "paper/tmlr/figures/*.pdf",
    "paper/tmlr/build/make_numbers.py", "paper/tmlr/build/make_tables.py", "paper/tmlr/build/make_figs.py",
    "paper/tmlr/build/verify_numbers.py", "paper/tmlr/build/design_literals.txt",
]
# the analysis folders of every stage (per-cell summaries and episode-level records written by the analysis code;
# coordinator decision 2026-11-22): every folder named analysis* under these results folders, at any depth. A folder
# whose files are byte-identical to an already listed one (the copies under raw/<pod>/) is left out. The raw job
# records around them are not included.
ANALYSIS_ROOTS = ["rss_ext/*/results", "tmlr_ext/*/results"]
ANALYSIS_DIR = "analysis*"
# never copied, even when listed above (caches, lock files, build intermediates, author-side scripts)
EXCLUDE = [r"(^|/)__pycache__/", r"\.pyc$", r"\.lock$", r"(^|/)\.DS_Store$", r"\.pid$",
           r"^paper/tmlr/(supplement|arxiv)/", r"^paper/tmlr/build/(build|make_arxiv|sync_papers_copy|make_supplement)\.sh$",
           r"^paper/tmlr/build/supplement_tool\.py$",
           r"^paper/tmlr/[^/]+\.(aux|log|out|bbl|blg|synctex\.gz|fls|fdb_latexmk|xdv|pdf)$"]
GENERATED_KEEP = [r"^paper/tmlr/numbers\.tex$", r"^paper/tmlr/tables/"]   # written by the build scripts: byte-exact
TEXT_EXT = {".md", ".py", ".sh", ".txt", ".json", ".jsonl", ".csv", ".tex", ".bib", ".sty", ".bst", ".log", ".err",
            ".yaml", ".yml", ".toml", ".cfg", ".ini", ".n", ".svg", ""}

# ================================================================================================ identifying strings
STATIC_NAMES = ["Yong Hwi Song", "YongHwi Song", "Yong-Hwi Song", "Song Yong Hwi", "Yonghwi Song", "Keon Myung Lee",
                "Keon-Myung Lee", "Hyunsoo Kang"]
KOREAN_NAMES = ["송용휘", "이건명"]


def author_terms(snapshot):
    """Names, e-mail addresses and affiliations of the authors: the static list above plus CITATION.cff and the
    (commented) author block of main.tex, so that a changed spelling is picked up."""
    names, emails, affs = set(STATIC_NAMES), set(), set()
    cff = os.path.join(snapshot, "CITATION.cff")
    if os.path.exists(cff):
        fam = giv = None
        for line in open(cff, encoding="utf-8"):
            m = re.match(r"\s*-?\s*(family-names|given-names|email|affiliation)\s*:\s*\"?([^\"\n]+)\"?", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip()
            if k == "family-names":
                fam = v
            elif k == "given-names":
                giv = v
            elif k == "email":
                emails.add(v)
            else:
                affs.add(v)
            if fam and giv:
                names.update({f"{giv} {fam}", f"{fam} {giv}"})
                fam = giv = None
    tex = os.path.join(snapshot, "paper", "tmlr", "main.tex")
    if os.path.exists(tex):
        t = open(tex, encoding="utf-8").read()
        names.update(x.strip() for x in re.findall(r"\\name\s+(.+?)\s*\\email", t))
        emails.update(re.findall(r"\\email\s+(\S+?)\s*(?:\\\\|\}|$)", t, re.M))
        for a in re.findall(r"\\addr\s+(.+?)\s*(?:\\\\|\}|$)", t, re.M):
            affs.update(p.strip() for p in a.split(",") if p.strip() and p.strip() not in ("Korea", "South Korea"))
    return sorted(n for n in names if len(n) > 4), sorted(emails), sorted(affs)


def name_regex(name):
    """'Yong Hwi Song' -> pattern that also matches 'YongHwi Song', 'Yong-Hwi  Song' (case-insensitive)."""
    parts = [re.escape(p) for p in re.split(r"[\s_-]+", name) if p]
    return r"[\s_-]*".join(parts)


def build_rules(snapshot):
    """Ordered (label, compiled regex, replacement). Longer and more specific forms first."""
    names, emails, affs = author_terms(snapshot)
    R = []

    def add(label, pat, rep, flags=re.I):
        R.append((label, re.compile(pat, flags), rep))

    # home paths and folder names (the folder names are the public repository names)
    add("home path", r"/Users/[^/\s\"'`]+/Documents/physical ai - Neural Networks", "[workspace]")
    add("home path", r"/Users/[^/\s\"'`]+", "~", 0)
    add("home path", r"/home/[A-Za-z0-9._-]+", "~", 0)
    add("home path", r"[A-Za-z]:\\\\?Users\\\\?[^\\\s\"']+", "~", 0)
    add("project folder", r"A_dithered_world_models", "A")
    add("project folder", r"B_near_miss_blindness", "B")
    add("project folder", r"physical ai - Neural Networks", "[workspace]")
    # server, network
    add("server", r"kiis[-_]mvf[-_]gpu", "[server]")
    add("server", r"[A-Za-z0-9-]*racedreamer[A-Za-z0-9-]*", "[server]")
    add("server", r"\bkiis\b", "[server]")
    add("VPN", r"tailscale|tailnet|\b[\w-]+\.ts\.net\b", "[VPN]")
    add("IP address", r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b", "[ip]:[port]", 0)
    add("IP address", r"(?<=ssh )(?:\d{1,3}\.){3}\d{1,3}\b", "[ip]", 0)
    add("IP address", r"(?<=@)(?:\d{1,3}\.){3}\d{1,3}\b", "[ip]", 0)
    # repositories and accounts
    add("repository", r"github\.com/thddydgnl(/[A-Za-z0-9._-]+)?", "[repository]")
    add("account", r"thddydgnl\w*", "[account]")
    add("repository", r"dithered[-_]world[-_]models|near[-_]miss[-_]blindness", "[repository]")
    # people
    for e in emails:
        add("e-mail", re.escape(e), "[e-mail]")
    add("e-mail", r"[A-Za-z0-9._%+-]+@(?:gmail\.com|cbnu\.ac\.kr|chungbuk\.ac\.kr|naver\.com|kakao\.com)", "[e-mail]")
    for n in sorted(names, key=len, reverse=True):
        add("author name", name_regex(n), "[author]")
    add("author name", r"yong[\s_-]*hwi|keon[\s_-]*myung|\bkmlee\b", "[author]")
    for n in KOREAN_NAMES:
        add("author name", re.escape(n), "[author]", 0)
    # institutions
    for a in sorted(affs, key=len, reverse=True):
        add("institution", re.escape(a), "[institution]")
    add("institution", r"Chungbuk(\s+National(\s+University)?)?|\bCBNU\b|충북대학교|충북대|"
        r"School of Information and Communications? Engineering|School of Software|정보통신공학부|소프트웨어학부",
        "[institution]")
    # other venues of this work, the poster, the preprint, other projects of the authors
    add("venue", r"jkaia\s*2026|jkaia|\bkaia\b|한국인공지능학회", "[venue]")
    for rx in private_patterns("venue"):     # further venues, kept in a local git-ignored file
        add("venue", rx, "[venue]")
    add("preprint", r"(arXiv[: ]?)?2611\.11780(v\d+)?|submit/8348128", "[preprint]")
    add("other project", r"drivewam|imle[-_ ]?wm", "[other project]")
    return R


def private_patterns(kind):
    """Extra patterns of one kind from build/private_patterns.txt (local, git-ignored; absent in a fresh clone)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "private_patterns.txt")
    if not os.path.exists(path):
        return []
    return [ln.split("\t", 1)[1].strip() for ln in open(path, encoding="utf-8")
            if ln.startswith(kind + "\t") and ln.split("\t", 1)[1].strip()]


# patterns of the final scan: everything the rules remove, plus generic forms that must not remain
ALLOWED_EMAILS = {"pieter@vanoostrum.org", "daly@mps.mpg.de", "tmlr-editors@jmlr.org", "noreply@anthropic.com"}
UPSTREAM_GITHUB = r"(Farama-Foundation|nicklashansen|gaoyuezhou|deepmind|google-deepmind|facebookresearch|mamba-org|" \
                  r"JmlrOrg|pytorch|NVIDIA|openai|huggingface|astral-sh|conda-forge|triton-lang|nvidia)"


def scan_patterns(snapshot):
    names, emails, affs = author_terms(snapshot)
    P = [("author name", re.compile("|".join([name_regex(n) for n in names] +
                                             [r"yong[\s_-]*hwi", r"keon[\s_-]*myung", r"\bkmlee\b"]), re.I)),
         ("author name", re.compile("|".join(KOREAN_NAMES))),
         # an address ends at its top-level domain: not '@torch.no_grad()' (code text recorded in a result file)
         ("e-mail", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w(])")),
         ("institution", re.compile("|".join([re.escape(a) for a in affs] +
                                             [r"chungbuk", r"\bcbnu\b", r"충북", r"정보통신공학", r"소프트웨어학부",
                                              r"school of software", r"information and communications? engineering"]),
                                    re.I)),
         ("server", re.compile(r"kiis|mvf[-_]gpu|racedreamer", re.I)),
         ("VPN", re.compile(r"tailscale|tailnet|\.ts\.net\b", re.I)),
         ("home path", re.compile(r"/Users/(?!\[)|/home/[A-Za-z]|[A-Za-z]:\\\\?Users\\\\?")),
         ("repository", re.compile(r"thddydgnl|dithered[-_ ]world[-_ ]models|near[-_ ]miss|github\.com/(?!"
                                   + UPSTREAM_GITHUB + r"/)", re.I)),
         ("venue", re.compile("|".join([r"jkaia", r"\bkaia\b", r"인공지능학회"] + private_patterns("scan")), re.I)),
         ("preprint", re.compile(r"2611\.11780|8348128")),
         ("other project", re.compile(r"drivewam|imle[-_ ]?wm|physical ai - neural", re.I)),
         ("personal plan", re.compile(r"대학원 지원|인턴 지원|지원서 서사")),
         # a network address: with a port, after ssh/@/://, after the word IP, or in the Tailscale range 100.64/10
         # (other dotted quads in this material are software versions such as mujoco-py 2.1.2.14)
         ("IP address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b|(?:(?<=ssh )|(?<=@)|(?<=://)|(?<=IP )|"
                                   r"(?<=ip )|(?<=host ))(?:\d{1,3}\.){3}\d{1,3}\b|"
                                   r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
         ("host name", re.compile(r"\b[A-Za-z0-9-]*(?:macbook|imac|mac-mini)[A-Za-z0-9-]*\.local\b", re.I)),
         ("secret", re.compile(r"rpa_[A-Za-z0-9]{10,}|\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}"
                               r"|BEGIN [A-Z ]*PRIVATE KEY|ssh-(?:ed25519|rsa) AAAA")),
         ]
    return P


# in binary files (compressed streams) short generic patterns match random bytes: there only the specific strings
# are searched, and e-mail addresses in a strict lower-case form
BINARY_SKIP = {"IP address", "host name"}
STRICT_EMAIL = re.compile(r"[a-z0-9._%+-]{2,}@[a-z0-9-]{2,}(?:\.[a-z0-9-]{2,})*\.(?:com|org|net|edu|kr|ac\.kr|io|de)\b")


def scan_text(text, P, binary=False):
    hits = []
    for label, rx in P:
        if binary and label in BINARY_SKIP:
            continue
        if binary and label == "e-mail":
            rx = STRICT_EMAIL
        for m in rx.finditer(text):
            s = m.group(0)
            if label == "e-mail" and s.lower() in ALLOWED_EMAILS:
                continue
            if label == "IP address" and s.split(":")[0] in ("1.1.1.1", "8.8.8.8", "8.8.4.4", "127.0.0.1", "0.0.0.0"):
                continue
            a, b = max(0, m.start() - 60), min(len(text), m.end() + 60)
            hits.append((label, s, text[a:b].replace("\n", " ")))
    return hits


# ================================================================================================ tracing
class _Tracer:
    def __init__(self, root, script, provenance):
        self.roots = {os.path.realpath(root), os.path.abspath(root)}
        self.root = os.path.realpath(root)
        self.script = os.path.realpath(script)
        self.provenance = provenance
        self.reads, self.writes, self.outside = set(), set(), set()
        self.events = []
        self.suspend = 0
        skip = {sys.prefix, sys.base_prefix, sys.exec_prefix, "/System", "/usr", "/Library", "/opt", "/dev",
                "/private/var/folders", "/var/folders", "/etc", "/private/etc", "/proc", "/sys", "/Applications"}
        try:
            import site
            skip.update(site.getsitepackages())
            skip.add(site.getusersitepackages())
        except Exception:
            pass
        home = os.path.expanduser("~")
        skip.update(os.path.join(home, p) for p in (".matplotlib", ".cache", ".config", "Library", ".local"))
        skip.update(os.environ[v] for v in ("MPLCONFIGDIR", "XDG_CACHE_HOME") if os.environ.get(v))
        self.skip = tuple(os.path.realpath(s) + os.sep for s in skip if s)
        self.setup = self._setup_ranges(script) if provenance else []

    @staticmethod
    def _setup_ranges(script):
        """Lines of a function before its first '# ----' section comment (loads and filters shared by the sections
        below). A value read there is not attributed to the next output: the sections read what they use."""
        import ast
        src = open(script, encoding="utf-8").read()
        lines = src.split("\n")
        out = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for ln in range(node.lineno, node.end_lineno + 1):
                    if lines[ln - 1].strip().startswith("# ----"):
                        out.append((node.lineno, ln))
                        break
        return out

    def rel(self, path):
        ap = os.path.abspath(path)
        for r in self.roots:
            if ap.startswith(r + os.sep):
                return os.path.relpath(os.path.realpath(ap), self.root)
        return None

    def note(self, src):
        if self.suspend or not src:
            return
        f = sys._getframe(2)
        innermost = True
        for _ in range(16):          # accesses made inside a consistency check (check_*) belong to no output
            if f is None:
                break
            if os.path.realpath(f.f_code.co_filename) == self.script:
                if f.f_code.co_name.startswith("check"):
                    return
                if innermost:
                    innermost = False
                    if any(a <= f.f_lineno < b for a, b in self.setup):
                        return
            f = f.f_back
        if not self.events or self.events[-1] != ("acc", src):
            self.events.append(("acc", src))

    def hook(self, event, args):
        if event != "open":
            return
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        path = os.fsdecode(path)
        r = self.rel(path)
        write = bool(mode and any(c in mode for c in "wax+")) or bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
        if r is None:
            rp = os.path.realpath(os.path.abspath(path))
            if not write and not rp.startswith(self.skip) and os.path.isfile(rp):
                self.outside.add(rp)
            return
        if "__pycache__" in r:
            return
        if write:
            self.writes.add(r)
            self.events.append(("out", r))
        else:
            self.reads.add(r)
            if self.provenance and r.endswith((".md", ".txt", ".tex")) and not r.startswith("paper/tmlr/build/"):
                self.note(r)


def _install_provenance(T, build_dir):
    """Wrap the data that json.load and csv.DictReader return, so that reading a value notes its source file."""
    def leaf(src, v):
        if not isinstance(v, (TDict, TList)):
            T.note(src)

    class TDict(dict):
        def __getitem__(self, k):
            v = dict.__getitem__(self, k)
            leaf(self._src, v)
            return v

        def get(self, k, default=None):
            return self[k] if dict.__contains__(self, k) else default

        def values(self):
            T.note(self._src)
            return dict.values(self)

        def items(self):
            T.note(self._src)
            return dict.items(self)

        def __len__(self):
            T.note(self._src)
            return dict.__len__(self)

    class TList(list):
        def __getitem__(self, i):
            v = list.__getitem__(self, i)
            if isinstance(i, slice):
                T.note(self._src)
            else:
                leaf(self._src, v)
            return v

        def __iter__(self):
            if any(not isinstance(x, (TDict, TList)) for x in list.__iter__(self)):
                T.note(self._src)
            return list.__iter__(self)

        def __len__(self):
            T.note(self._src)
            return list.__len__(self)

    def wrap(o, src):
        if type(o) is dict:
            d = TDict({k: wrap(v, src) for k, v in o.items()})
            d._src = src
            return d
        if type(o) is list:
            lst = TList(wrap(v, src) for v in o)
            lst._src = src
            return lst
        return o

    jload = json.load

    def load(fp, *a, **k):
        data = jload(fp, *a, **k)
        src = T.rel(getattr(fp, "name", "") or "")
        return wrap(data, src) if src else data

    json.load = load

    class TDictReader(csv.DictReader):
        def __init__(self, f, *a, **k):
            super().__init__(f, *a, **k)
            self._src = T.rel(getattr(f, "name", "") or "")

        def __next__(self):
            row = super().__next__()
            if not self._src:
                return row
            d = TDict(row)
            d._src = self._src
            return d

    csv.DictReader = TDictReader

    # values of numbers.tex used by the tables: note the source files that make_numbers records for each key
    class TStr(str):
        def __format__(self, spec):
            for s in self._srcs:
                T.note(s)
            return format(str.__str__(self), spec)

        def __str__(self):
            for s in self._srcs:
                T.note(s)
            return str.__str__(self)

    sys.path.insert(0, build_dir)
    try:
        import make_numbers as mn
    except Exception:
        return
    known = {}

    def srcs_of(source):
        out = []
        for tok in re.findall(r"[\w./-]+\.(?:json|csv|md|txt)", source):
            tok = tok.strip("./")
            if "/" in tok and os.path.exists(os.path.join(T.root, tok)):
                out.append(tok)
            else:
                out.extend(known.get(os.path.basename(tok), []))
        return out

    orig = mn.compute

    def compute(*a, **k):
        T.suspend += 1
        try:
            entries = orig(*a, **k)
        finally:
            T.suspend -= 1
        for r in T.reads:
            known.setdefault(os.path.basename(r), [])
            if r not in known[os.path.basename(r)]:
                known[os.path.basename(r)].append(r)
        res = []
        for key, tex, source, kind in entries:
            t = TStr(tex)
            t._srcs = srcs_of(source)
            res.append((key, t, source, kind))
        return res

    mn.compute = compute


def _record_sha256(digests):
    """Let hashlib.sha256 record every digest it returns (to find which design-document texts the scripts hash)."""
    real = hashlib.sha256

    class Sha256:
        name, digest_size, block_size = "sha256", 32, 64

        def __init__(self, data=b"", **kw):
            self._h = real(data, **kw)

        def update(self, data):
            self._h.update(data)

        def hexdigest(self):
            h = self._h.hexdigest()
            digests.add(h)
            return h

        def digest(self):
            self.hexdigest()
            return self._h.digest()

        def copy(self):
            c = Sha256()
            c._h = self._h.copy()
            return c

    hashlib.sha256 = Sha256


def cmd_trace(argv, provenance=False):
    out, root, script = argv[0], argv[1], os.path.abspath(argv[2])
    T = _Tracer(root, script, provenance)
    sys.argv = [script] + argv[3:]
    sys.path.insert(0, os.path.dirname(script))
    digests = set()
    _record_sha256(digests)
    if provenance:
        _install_provenance(T, os.path.dirname(script))
    sys.addaudithook(T.hook)
    code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        T.suspend += 1
        outputs, together = {}, {}
        pending, last, last_out = set(), set(), None
        for kind, x in T.events:
            if kind == "acc":
                pending.add(x)
            else:               # an output written right after another one, with no read between, was built with it
                if not pending and last_out and os.path.splitext(last_out)[0] != os.path.splitext(x)[0]:
                    together[x] = last_out
                got = pending or last
                outputs.setdefault(x, set()).update(got)
                last, pending, last_out = set(got), set(), x
        reads = sorted(r for r in T.reads - T.writes if os.path.isfile(os.path.join(T.root, r)))
        rec = {"script": os.path.relpath(T.script, T.root), "args": argv[3:], "exit": code, "reads": reads,
               "writes": sorted(T.writes), "outside_reads": sorted(T.outside), "sha256_digests": sorted(digests)}
        if provenance:
            rec["outputs"] = {k: sorted(v - {k}) for k, v in outputs.items()}
            rec["built_together"] = together
        with open(out, "w") as fh:
            json.dump(rec, fh, indent=1, ensure_ascii=False)
    sys.exit(code)


# ================================================================================================ anonymization
def strip_tex_comments(text):
    """Remove every comment of a LaTeX file without changing what it typesets: a line that holds only a comment is
    deleted with its line end (TeX ignores it entirely), a trailing comment is cut back to its '%' (which keeps
    the line end from becoming a space). '%' after a backslash is a literal percent sign (verify_numbers.py rule)."""
    out = []
    for line in text.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        if m is None:
            out.append(line)
        elif line[:m.start()].strip() == "":
            continue
        else:
            out.append(line[:m.start() + 1])
    return "\n".join(out)


def drop_arxiv_branch(text):
    """Replace each \\ifarxiv A \\else B \\fi by B (the anonymous branch); \\newif\\ifarxiv stays."""
    out, pos = [], 0
    for m in re.finditer(r"\\ifarxiv(?![A-Za-z])", text):
        if m.start() < pos or text[max(0, m.start() - 6):m.start()] == "\\newif":
            continue
        e = re.compile(r"\\(else|fi)(?![A-Za-z])")
        m1 = e.search(text, m.end())
        if m1 is None:
            raise SystemExit("unbalanced \\ifarxiv")
        if re.search(r"\\if[A-Za-z]*(?![A-Za-z])", text[m.end():m1.start()]):
            raise SystemExit("nested \\if inside \\ifarxiv: anonymize by hand")
        if m1.group(1) == "else":
            m2 = re.compile(r"\\fi(?![A-Za-z])").search(text, m1.end())
            branch = text[m1.end():m2.start()]
            end = m2.end()
        else:
            branch, end = "", m1.end()
        out.append(text[pos:m.start()])
        out.append(branch[1:] if branch.startswith("\n") else branch)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def strip_bib_comments(text):
    """Drop the comment lines of a .bib file that lie outside entries (BibTeX ignores text between entries)."""
    out, depth = [], 0
    for line in text.split("\n"):
        if depth == 0 and line.lstrip().startswith("%"):
            continue
        out.append(line)
        depth += line.count("{") - line.count("}")
        depth = max(depth, 0)
    return "\n".join(out)


TEX_HEADER = "% Anonymized copy of the submission source for the supplementary material (all comments removed).\n"


def redact(text, rules, counts):
    for label, rx, rep in rules:
        text, n = rx.subn(rep, text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text


def drop_venue_lines(text, counts):
    """In a Markdown document, a line that mentions another venue of this work (the poster and its manuscript) is
    emptied, not only the venue's name: such lines also give deadlines, tracks and page limits. A table row keeps
    its cells; the cells that mention the venue are emptied. The number of lines does not change."""
    out, n = [], 0
    for line in text.split("\n"):
        if "[venue]" in line:
            n += 1
            if line.lstrip().startswith("|"):
                line = "|".join(" [removed] " if "[venue]" in c else c for c in line.split("|"))
            else:
                lead = re.match(r"\s*(?:[-*+]\s+|\d+\.\s+|#+\s+)?", line).group(0)
                line = lead + "(removed: a line about another venue of this work)"
        out.append(line)
    if n:
        counts["lines about other venues emptied"] = n
    return "\n".join(out)


# personal plans of the authors, unrelated to the review (coordinator decision 2026-11-22): a phrase is removed,
# a block (a bold heading line and the lines under it, up to the next blank line) is emptied line by line
PERSONAL_PHRASES = [r"대학원 지원서와 인턴 지원에 링크"]
PERSONAL_BLOCKS = [r"^\*\*지원서 서사 연결\*\*\s*$"]


def drop_personal_notes(text, counts):
    n = 0
    for p in PERSONAL_PHRASES:
        text, k = re.subn(p, "", text)
        n += k
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if any(re.match(p, lines[i]) for p in PERSONAL_BLOCKS):
            while i < len(lines) and lines[i].strip():
                lines[i] = ""
                i += 1
                n += 1
        else:
            i += 1
    if n:
        counts["personal plans removed"] = n
    return "\n".join(lines)


def is_text_bytes(data, ext):
    if b"\x00" in data[:65536]:
        return False
    if ext in TEXT_EXT:
        return True
    try:
        data[:65536].decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def sha(data):
    return hashlib.sha256(data).hexdigest()


def select_files(snapshot, reads):
    files = set()
    for pat in INCLUDE:
        if pat.endswith("/**"):
            for top in _glob_dirs(snapshot, pat[:-3]):
                for dp, dns, fns in os.walk(os.path.join(snapshot, top)):
                    dns[:] = [x for x in dns if x != "__pycache__"]
                    for f in fns:
                        files.add(os.path.relpath(os.path.join(dp, f), snapshot))
        else:
            files.update(_glob(snapshot, pat))
    files |= set(analysis_files(snapshot))
    files |= set(reads)
    keep = set()
    for f in files:
        if any(re.search(x, f) for x in EXCLUDE):
            if f in reads:
                raise SystemExit(f"a build script reads an excluded file: {f}")
            continue
        keep.add(f)
    return sorted(keep)


def analysis_files(snapshot):
    """Files of the analysis folders (ANALYSIS_ROOTS / ANALYSIS_DIR), shortest path first, without exact duplicates."""
    dirs = []
    for root in ANALYSIS_ROOTS:
        for top in _glob_dirs(snapshot, root):
            for dp, dns, fns in os.walk(os.path.join(snapshot, top)):
                dns[:] = [x for x in dns if x != "__pycache__"]
                if fnmatch.fnmatchcase(os.path.basename(dp), ANALYSIS_DIR):
                    dirs.append(os.path.relpath(dp, snapshot))
    seen, out = set(), []
    for d in sorted(dirs, key=lambda x: (x.count("/"), x)):
        members = []
        for dp, dns, fns in os.walk(os.path.join(snapshot, d)):
            for f in fns:
                p = os.path.join(dp, f)
                members.append((os.path.relpath(p, os.path.join(snapshot, d)), sha(open(p, "rb").read())))
        sig = tuple(sorted(m for m in members if not m[0].endswith(".DS_Store")))
        if sig in seen:
            continue
        seen.add(sig)
        out += [os.path.join(d, m[0]) for m in members]
    return out


def _glob(snapshot, pat):
    parts = pat.split("/")
    cur = [""]
    for p in parts:
        nxt = []
        for c in cur:
            d = os.path.join(snapshot, c)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if fnmatch.fnmatchcase(name, p):
                    nxt.append(os.path.join(c, name) if c else name)
        cur = nxt
    return [c for c in cur if os.path.isfile(os.path.join(snapshot, c))]


def _glob_dirs(snapshot, pat):
    parts = pat.split("/")
    cur = [""]
    for p in parts:
        nxt = []
        for c in cur:
            d = os.path.join(snapshot, c)
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if fnmatch.fnmatchcase(name, p) and os.path.isdir(os.path.join(d, name)):
                    nxt.append(os.path.join(c, name) if c else name)
        cur = nxt
    return cur


def hashed_text_map(orig, red, digests):
    """For a redacted document: sha256 of each redacted text -> sha256 of its original, for the texts whose original
    hash a build script computed (digests): the whole file, or the file cut before any line (as is, without the last
    newline, or with trailing newlines collapsed to one, the form make_numbers._design_prefix hashes). The redaction
    never adds or removes a line, so line j of the copy is line j of the original."""
    def variant(lines, i):          # i = 0: the whole file; i >= 1: cut after line (i - 1) // 3, three forms
        if i == 0:
            return None
        p = "\n".join(lines[:(i - 1) // 3 + 1]) + "\n"
        return (p, p[:-1], p.rstrip("\n") + "\n")[(i - 1) % 3].encode()

    ol = orig.decode("utf-8").split("\n")
    n = 1 + 3 * len(ol)
    used = [i for i in range(n) if sha(orig if i == 0 else variant(ol, i)) in digests]
    if not used:
        return {}
    rl = red.decode("utf-8").split("\n")
    if len(ol) != len(rl):
        raise SystemExit("redaction changed the number of lines of a document whose hash a build script checks")
    m = {}
    for i in used:
        ho = sha(orig if i == 0 else variant(ol, i))
        hr = sha(red if i == 0 else variant(rl, i))
        if hr != ho:
            m[hr] = ho
    return m


SHIM_START = "# --- [supplement] "
SHIM = '''# --- [supplement] hash checks of redacted design documents ---------------------------------------------------------
# This block is the only difference between this copy of make_numbers.py and the authors' copy. The design documents
# in this supplement are redacted copies (names, server and account details removed), so their SHA-256 hashes differ
# from the hashes logged when the originals were frozen. build/redacted_hashes.json, written by the packaging script
# from the original files, maps the hash of each redacted text that the checks below compute (the whole file, and the
# file cut before each later section) to the hash of the original text. hashlib.sha256 here reports the original's
# hash for exactly those texts and is unchanged for every other input.
import hashlib as _hashlib
import json as _json
import os as _os

_REDACTED = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "redacted_hashes.json"),
                            encoding="utf-8"))["map"]
_sha256 = _hashlib.sha256


class _OriginalHashOfRedacted:
    def __init__(self, data=b"", **kw):
        self._h = _sha256(data, **kw)

    def update(self, data):
        self._h.update(data)

    def hexdigest(self):
        h = self._h.hexdigest()
        return _REDACTED.get(h, h)

    def digest(self):
        return bytes.fromhex(self.hexdigest())

    def copy(self):
        c = _OriginalHashOfRedacted()
        c._h = self._h.copy()
        return c


if not getattr(_hashlib.sha256, "_supplement", False):
    _OriginalHashOfRedacted._supplement = True
    _hashlib.sha256 = _OriginalHashOfRedacted
# --- [supplement] end -----------------------------------------------------------------------------------------------
'''


def insert_shim(src):
    """Insert SHIM after the module docstring of make_numbers.py."""
    import ast
    tree = ast.parse(src)
    first = tree.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)):
        raise SystemExit("make_numbers.py has no module docstring: cannot place the supplement block")
    lines = src.split("\n")
    end = first.end_lineno
    return "\n".join(lines[:end] + [SHIM.rstrip("\n")] + lines[end:])


# ================================================================================================ manuscript floats
def expand_inputs(tmlr_dir, text):
    def rep(m):
        p = os.path.join(tmlr_dir, m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        return open(p, encoding="utf-8").read() if os.path.exists(p) else m.group(0)
    return re.sub(r"\\input\{(appendix/[^}]+)\}", rep, text)


def manuscript_floats(tmlr_dir):
    """Tables and figures of the manuscript in source order, numbered as LaTeX numbers them (one counter each, by
    \\caption), with their section, label and the generated files they include."""
    t = strip_tex_comments(expand_inputs(tmlr_dir, open(os.path.join(tmlr_dir, "main.tex"), encoding="utf-8").read()))
    t = t.split("\\begin{document}", 1)[-1]
    tok = re.compile(r"\\appendix\b|\\section\*?\{([^}]*)\}|\\begin\{(table|figure)\*?\}|\\end\{(table|figure)\*?\}")
    floats, sec_main, sec_app, in_app, cur, start = [], 0, 0, False, None, None
    counters = {"table": 0, "figure": 0}
    sec_label = ""
    for m in tok.finditer(t):
        s = m.group(0)
        if s.startswith("\\appendix"):
            in_app = True
        elif s.startswith("\\section"):
            if in_app:
                sec_app += 1
                sec_label = "App. " + chr(64 + sec_app)
            else:
                sec_main += 1
                sec_label = f"Sec. {sec_main}"
        elif s.startswith("\\begin"):
            cur, start = m.group(2), m.end()
        elif s.startswith("\\end") and cur:
            body = t[start:m.start()]
            ncap = len(re.findall(r"\\caption\b", body))
            if ncap:
                counters[cur] += 1
                lab = re.search(r"\\label\{([^}]+)\}", body)
                files = re.findall(r"\\inputrows\{([^}]+)\}|\\input\{(tables/[^}]+)\}", body)
                files = [a or b for a, b in files]
                files += ["figures/" + g for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)]
                floats.append({"kind": cur, "num": counters[cur], "section": sec_label,
                               "label": lab.group(1) if lab else "", "files": files})
            cur = None
    return floats


# ================================================================================================ assemble
def cmd_assemble(argv):
    snapshot, work, pkg = (os.path.abspath(a) for a in argv[:3])
    traces = {}
    for f in sorted(os.listdir(os.path.join(work, "trace"))):
        if f.endswith(".json"):
            traces[f[:-5]] = json.load(open(os.path.join(work, "trace", f)))
    reads = sorted(set().union(*(set(t["reads"]) for t in traces.values())))
    digests = set().union(*(set(t.get("sha256_digests", [])) for t in traces.values()))
    script_reads = set().union(*(set(t["reads"]) for k, t in traces.items() if not k.startswith("prov_")))
    rules = build_rules(snapshot)
    files = select_files(snapshot, set(reads))
    if os.path.exists(pkg):
        shutil.rmtree(pkg)
    record, hash_map, hashed_docs = [], {}, []
    for rel in files:
        src = os.path.join(snapshot, rel)
        dst = os.path.join(pkg, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = open(src, "rb").read()
        ext = os.path.splitext(rel[:-3] if rel.endswith(".gz") else rel)[1].lower()
        counts, kind, new = {}, "binary", data
        if rel.endswith(".gz"):
            raw = gzip.decompress(data)
            if is_text_bytes(raw, ext):
                kind = "gz text"
                t = raw.decode("utf-8")
                t2 = redact(t, rules, counts)
                if t2 != t:
                    bio = io.BytesIO()
                    with gzip.GzipFile(fileobj=bio, mode="wb", mtime=0, filename="") as gz:
                        gz.write(t2.encode("utf-8"))
                    new = bio.getvalue()
        elif is_text_bytes(data, ext):
            kind = "text"
            t = data.decode("utf-8")
            t1 = t
            if rel == "paper/tmlr/main.tex" or re.match(r"paper/tmlr/appendix/.*\.tex$", rel):
                t1 = strip_tex_comments(t1)          # first: comments also mention \ifarxiv
                if rel.endswith("main.tex"):
                    t1 = TEX_HEADER + drop_arxiv_branch(t1)
                    counts["comments and the arXiv branch removed"] = 1
                else:
                    counts["comments removed"] = 1
            elif rel == "paper/tmlr/refs.bib":
                t1 = strip_bib_comments(t1)
                counts["BibTeX comment lines removed"] = 1
            t2 = redact(t1, rules, counts)
            if rel.endswith(".md"):
                t2 = drop_personal_notes(drop_venue_lines(t2, counts), counts)
            if any(re.search(x, rel) for x in GENERATED_KEEP) and t2 != t:
                raise SystemExit(f"redaction would change the generated file {rel}: fix the rules or the source")
            new = t2.encode("utf-8")
            if rel in script_reads and new != data and kind == "text":
                m = hashed_text_map(data, new, digests)
                if m:
                    hash_map.update(m)
                    hashed_docs.append(rel)
        with open(dst, "wb") as fh:
            fh.write(new)
        shutil.copystat(src, dst)
        record.append({"path": rel, "kind": kind, "bytes": len(new), "changed": new != data, "rules": counts,
                       "sha256_original": sha(data), "sha256_copy": sha(new)})
    # make_numbers.py: hash checks of the redacted design documents
    mnp = os.path.join(pkg, "paper", "tmlr", "build", "make_numbers.py")
    if hash_map:
        with open(os.path.join(pkg, "paper", "tmlr", "build", "redacted_hashes.json"), "w", encoding="utf-8") as fh:
            json.dump({"note": "sha256 of a redacted text -> sha256 of the original text (see the [supplement] block "
                               "of make_numbers.py)", "documents": hashed_docs, "map": dict(sorted(hash_map.items()))},
                      fh, indent=1)
        s = open(mnp, encoding="utf-8").read()
        if SHIM_START not in s:
            s2 = insert_shim(s)
            open(mnp, "w", encoding="utf-8").write(s2)
            for r in record:
                if r["path"] == "paper/tmlr/build/make_numbers.py":
                    r["changed"] = True
                    r["rules"]["hash-check block inserted"] = 1
                    r["sha256_copy"] = sha(s2.encode())
            record.append({"path": "paper/tmlr/build/redacted_hashes.json", "kind": "text", "bytes": 0,
                           "changed": True, "rules": {"written by the packaging script": 1},
                           "sha256_original": "", "sha256_copy": ""})
    write_readme(snapshot, work, pkg, record, traces, hashed_docs, argv[3] if len(argv) > 3 else None)
    json.dump({"files": record, "reads": reads, "hashed_docs": hashed_docs},
              open(os.path.join(work, "assemble_record.json"), "w"), indent=1, ensure_ascii=False)
    n_changed = sum(r["changed"] for r in record)
    print(f"assembled {len(record)} files ({n_changed} changed by anonymization) into {pkg}")


# ================================================================================================ README
PYVER = ".".join(map(str, sys.version_info[:3]))
DOC_GUIDE = [
    ("EXPERIMENT_DESIGN.md", "first design of the study, written before the main study"),
    ("main/DESIGN_main.md", "design of the main study E0-E7, versions v1-v5.3; each later version is appended as a "
                            "dated section, so an earlier version is the document cut before its first later section"),
    ("rss_ext/DESIGN_ext.md", "design of the extension E8-E10"),
    ("rss_ext/DESIGN_E8c.md", "design of E8c (fixed after E8-E10)"),
    ("tmlr_ext/DESIGN_tmlr.md", "design of E11-E17 (v1.0) and, in its sec. 11, the decisions appended after the freeze"),
    ("tmlr_ext/WORKER_RULES.md", "operating rules of the E11-E17 runs"),
    ("*/*/PLAN.md", "operational plan of each stage or group of runs, fixed before its main run"),
    ("main/LOG.md", "coordinator log of the main study (UTC times, design hashes, incidents, decisions)"),
    ("rss_ext/LOG.md", "coordinator log of E8-E10 (with the cost ledger total)"),
    ("tmlr_ext/LOG.md", "coordinator log of E11-E17 and the post hoc runs (design hashes, costs, decisions)"),
    ("*/*/LOG.md", "log of each stage or group of runs (machines, checks, deviations)"),
    ("main/worklogs/*.md", "worker logs of the main study"),
    ("main/reports/*.md", "stage reports of E0-E7; `FINAL_A.md` is the final report of the main study"),
    ("*/*/REPORT.md", "stage reports of E8-E17"),
    ("tmlr_ext/RESULTS_tmlr.md", "results note of E11-E17, with the re-analyses R1-R5 (sec. 7)"),
    ("tmlr_ext/coordinator/*.txt", "outputs of the coordinator's scripts that recomputed verdicts from the raw job "
                                   "records or compared stages (`reanalysis/`: R1-R5)"),
]


def _size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} kB"


def write_readme(snapshot, work, pkg, record, traces, hashed_docs, git_src=None):
    tmlr = os.path.join(pkg, "paper", "tmlr")
    title = re.search(r"\\title\{(.+?)\}\n", open(os.path.join(tmlr, "main.tex"), encoding="utf-8").read(), re.S)
    title = re.sub(r"\\\\", " ", title.group(1)).replace("  ", " ") if title else "paper A"
    # outputs -> inputs (provenance traces of make_tables.py and make_figs.py)
    prov, together = {}, {}
    stem = lambda p: os.path.splitext(os.path.relpath(p, "paper/tmlr"))[0]   # noqa: E731
    for name in ("prov_make_tables", "prov_make_figs"):
        for out, ins in traces.get(name, {}).get("outputs", {}).items():
            prov.setdefault(stem(out), set()).update(i for i in ins if not i.startswith("paper/tmlr/"))
        for out, prev in traces.get(name, {}).get("built_together", {}).items():
            together[stem(out)] = stem(prev)
    floats = manuscript_floats(tmlr)
    item_of = {}
    for f in floats:
        for x in f["files"]:
            item_of[os.path.splitext(x)[0]] = ("Table " if f["kind"] == "table" else "Figure ") + str(f["num"])
    rows = []
    for f in floats:
        name = ("Table " if f["kind"] == "table" else "Figure ") + str(f["num"])
        gen = [x for x in f["files"] if os.path.splitext(x)[0] in prov]
        if gen:
            own = {os.path.splitext(x)[0] for x in gen}
            mates = [together[s] for s in own if s in together and together[s] not in own]
            if mates and all(s in together for s in own):
                ins_txt = f"built in one step with {item_of.get(mates[0], mates[0])}: the files listed there"
            else:
                ins = sorted(set().union(*(prov[os.path.splitext(x)[0]] for x in gen)))
                ins_txt = "<br>".join(f"`{i}`" for i in ins)
            rows.append(f"| {name} | {f['section']} | `{f['label']}` | " + ", ".join(f"`{g}`" for g in gen) + " | "
                        + ins_txt + " |")
        else:
            rows.append(f"| {name} | {f['section']} | `{f['label']}` | typed in the source | design constants and "
                        f"`\\nv{{}}` values only (sources in `numbers.tex`) |")
    used = {os.path.splitext(x)[0] for f in floats for x in f["files"]}
    for s in sorted(set(prov) - used):
        if s.startswith(("tables/", "figures/")):
            rows.append(f"| (not in the manuscript) | | | `{s}` | " + "<br>".join(f"`{i}`" for i in sorted(prov[s])) + " |")
    # sizes
    by_top = {}
    for r in record:
        top = r["path"].split("/")[0] if "/" in r["path"] else "(top level)"
        a = by_top.setdefault(top, [0, 0])
        a[0] += 1
        a[1] += r["bytes"]
    changed = [r for r in record if r["changed"] and r["path"] != "paper/tmlr/build/redacted_hashes.json"]
    numbers = open(os.path.join(tmlr, "numbers.tex"), encoding="utf-8").read()
    nvals = len(re.findall(r"\\defnv\{", numbers))
    excl = excluded_summary(snapshot, {r["path"] for r in record})
    shipped_current = None
    p = os.path.join(work, "logs", "shipped_vs_regenerated.txt")
    if os.path.exists(p):
        m = re.search(r": (\d+) identical, (\d+) different, (\d+) missing", open(p, encoding="utf-8").read())
        shipped_current = bool(m) and m.group(2) == "0" and m.group(3) == "0"
    verify_out = ""
    p = os.path.join(work, "logs", "baseline_verify_numbers.txt")
    if os.path.exists(p):
        verify_out = "\n".join(l for l in open(p, encoding="utf-8").read().split("\n")
                               if re.match(r"(numbers\.tex:|wording checks|registered literal|--strict)", l))
    L = []
    L += [f"# Supplementary material: {title}", "",
          "Anonymized supplementary material of a TMLR submission. It holds the code of every stage as it ran, the "
          "design documents, the run logs, the stage reports, the result files that the paper's numbers, tables and "
          "figures are computed from, the analysis folders of the extension stages (per-cell summaries and "
          "episode-level records), and the manuscript source with the scripts that recompute every number.", "",
          "**Language.** The design documents, logs and stage reports were written in Korean as the study ran; the "
          "code, the result files and the manuscript are in English. Appendices A-C of the paper summarize the "
          "designs in English (versions, planned tests, stages and cells).", "",
          "## Recompute the numbers (no GPU; Python standard library only)", "",
          "```bash", "cd paper/tmlr",
          "python3 build/make_numbers.py && python3 build/make_tables.py && python3 build/verify_numbers.py --strict",
          "python3 build/make_figs.py      # figures; needs matplotlib and pymupdf", "```", "",
          f"Tested with Python {PYVER}. The files are UTF-8 and some are opened without an explicit encoding, so use "
          "a UTF-8 locale (the default on Linux and macOS; on Windows run `python3 -X utf8 ...` or set "
          "`PYTHONUTF8=1`).", "",
          f"- `make_numbers.py` writes `numbers.tex`: every result number of the paper ({nvals} values) as a macro "
          "`\\nv{key}`, each with the file and field it comes from in the comment next to it. The shipped "
          "`numbers.tex`, `tables/` and `figures/` are the copies the manuscript was typeset from"
          + ("; the commands rewrite them byte for byte (checked when this package was built)." if shipped_current
             else "."),
          "- `make_tables.py` writes `tables/*.tex` (every numeric table) from the result files.",
          "- `verify_numbers.py --strict` re-runs the computation, checks that `numbers.tex` is current, that every "
          "macro in the text is defined, that statements whose wording depends on the data agree with it, and that "
          "every number still typed in the text is a registered design constant. A few values exist only in stage "
          "reports and logs (compute hours, costs, the number of simulated planner runs); it lists them and checks "
          "that each appears in its source document.",
          "- `make_figs.py` writes `figures/*.pdf` (and PNG previews); unchanged data give byte-identical PDFs.",
          "- With the TMLR style files included, `tectonic main.tex` (or pdflatex + bibtex) typesets the manuscript "
          "from `main.tex`, an anonymized copy of the submission source with all comments removed.", ""]
    if verify_out:
        L += ["Output of the last command when this package was built:", "", "```", verify_out, "```", ""]
    L += ["## Tables and figures", "",
          "Numbers in the text: `numbers.tex` (each macro names its source). Generated tables and figures, with the "
          "result files whose data the generating script read while building each one (traced when this package was "
          "built):", "",
          "| Item | Where | Label | Generated file | Result files read |", "|---|---|---|---|---|"] + rows + [""]
    present = {r["path"] for r in record}
    docs = [(p, d) for p, d in DOC_GUIDE if any(fnmatch.fnmatchcase(x, p) for x in present)]
    L += ["## Design documents, logs and reports (Korean)", "",
          "| Path | What it is |", "|---|---|"] + [f"| `{p}` | {d} |" for p, d in docs] + [""]
    hrows = design_hash_check(pkg, [p for p, _ in DOC_GUIDE[:6]], snapshot, git_src)
    if hrows:
        L += ["## Design versions and their SHA-256 hashes", "",
              "The versions of the design documents listed in Appendix A of the paper, with the time (UTC) and the "
              "SHA-256 hash logged when each was frozen, before the data it governs existed. Later changes were "
              "appended as dated sections, so an earlier version is the document cut before its first later "
              "section. The hashes are also recorded in `main/LOG.md`, `rss_ext/LOG.md` and `tmlr_ext/LOG.md` (v5.1 "
              "only by its first 8 digits, in the worker logs). The last column was checked when this package was "
              "built against the copies in it: a **redacted copy** (names, server and account details removed) no "
              "longer reproduces the hash of its original, and the hash its copy gives instead is shown.", "",
              "| Version | Frozen (UTC) | SHA-256 of the original | Document | Copy in this supplement |",
              "|---|---|---|---|---|"] + [f"| {a} | {t} | `{h}` | {d} | {how} |" for a, t, h, d, how in hrows] + [""]
    L += ["## Layout", "",
          "The folder structure is the one the scripts expect (`paper/tmlr/build/*.py` read `../../../main`, "
          "`../../../rss_ext` and `../../../tmlr_ext` by relative paths).", "",
          "| Folder | Contents | Files | Size |", "|---|---|---|---|"]
    desc = {"main": "main study E0-E7: design (`DESIGN_main.md`, all versions appended), coordinator log (`LOG.md`), "
                    "worker logs (`worklogs/`), code (`code/tdmpc2`, `code/dinowm`, `code/theory`), stage reports "
                    "(`reports/`), result files read by the scripts (`results/`)",
            "rss_ext": "extension E8-E10 and E8c: designs, log, one folder per stage with its plan, log, report, code "
                       "and result files (`results/**/analysis*/`: per-cell summaries and episode-level records)",
            "tmlr_ext": "E11-E17 and the post hoc runs: design (`DESIGN_tmlr.md`), worker rules, log, results note "
                        "(`RESULTS_tmlr.md`); one folder per group of runs (`TD_tdmpc2` 5M, `MT_317m` 317M, "
                        "`DW_dinowm` DINO-WM, `K_kernel` integer kernel) with its plan, code, log, report and result "
                        "files (`results/**/analysis*/`: per-cell summaries and episode-level records); "
                        "`coordinator/`: the scripts that recomputed the verdicts from the raw job records and the "
                        "re-analyses R1-R5, with their outputs",
            "paper": "`paper/tmlr`: manuscript source (anonymized), `numbers.tex`, `tables/`, `figures/`, style files, "
                     "and the build scripts in `build/`",
            "(top level)": "this README and the first experiment design (`EXPERIMENT_DESIGN.md`)"}
    for top in sorted(by_top, key=lambda x: (x == "(top level)", x)):
        n, b = by_top[top]
        L.append(f"| `{top}` | {desc.get(top, '')} | {n} | {_size(b)} |")
    L += ["", "## Anonymization", "",
          "Author names, e-mail addresses, institutions, the name of the university server and of the VPN used to "
          "reach it, home-directory paths, host names, repository and account names, other venues of this work and "
          "the preprint identifier were replaced by bracketed tokens (`[author]`, `[server]`, `[VPN]`, `~`, "
          "`[workspace]`, `[repository]`, `[venue]`, `[preprint]`, ...), and lines of the logs and reports that "
          "discuss another venue of this work were emptied. The manuscript copy has no comments and only the "
          "anonymous branch of its arXiv switch. No number was changed.", ""]
    L += ["**Design-document hashes.** Redacting a document changes its SHA-256 hash, so a redacted copy no longer "
          "has the hash logged when the original was frozen; see the section on design versions below for which "
          "logged hashes the copies here reproduce."]
    if hashed_docs:
        L += ["", "**One change to a build script.** `make_numbers.py` recomputes the logged hashes of "
              + ", ".join(f"`{d}`" for d in hashed_docs) + " (of its earlier versions, the document cut before "
              "each later section) and stops if they differ. In this copy a marked block at its top maps the hash of "
              "each redacted text to the hash of the original, from `build/redacted_hashes.json`, which the "
              "packaging script computed from the original files. The block changes no other hash and is the only "
              "change to the build scripts."]
    L += ["", "Files changed by the anonymization (the SHA-256 of the copy differs from the original's; the hash of "
          "each original is given):", "",
          "| File | What was replaced (count) | SHA-256 of the original (first 12) |", "|---|---|---|"]
    groups = {}
    for r in changed:
        what = ", ".join(f"{k} ({v})" if v > 1 else k for k, v in sorted(r["rules"].items()))
        groups.setdefault((os.path.dirname(r["path"]), what, r["path"].endswith(".json.gz")), []).append(r)
    for (d, what, gz), rs in sorted(groups.items(), key=lambda g: g[1][0]["path"]):
        if gz and len(rs) > 3:
            L.append(f"| `{d}/*.json.gz` ({len(rs)} files) | {what} | (each file) |")
        else:
            L += [f"| `{r['path']}` | {what} | `{r['sha256_original'][:12]}` |" for r in rs]
    L += ["", "## Not included", "",
          f"- The raw job records of every machine (the per-job outputs that the analysis code reduced to the result "
          f"files and analysis folders here): {excl['n']} files, {_size(excl['bytes'])} in the authors' copy ("
          + ", ".join(f"{k} {_size(v)}" for k, v in excl["by"].items()) + "). The analysis code and the "
          "recomputation scripts in `tmlr_ext/coordinator/` are included with their outputs, but they need those "
          "records to run again.",
          "- Model checkpoints and datasets: the public releases of TD-MPC2 and DINO-WM; their SHA-256 hashes are in "
          "`*/code/inputs/ckpt_SHA256SUMS` and the stage reports. Upstream commits and software versions are in "
          "Appendix C of the paper.",
          "- The authors' build scripts for the PDF and other versions of the paper.", ""]
    open(os.path.join(pkg, "README.md"), "w", encoding="utf-8").write("\n".join(L))


def design_hash_check(pkg, docs, snapshot=None, git_src=None):
    """Rows (version, logged hash, how this package reproduces it) for the table of versions in appendix/app_A.tex;
    hashes given as \\nv{} keys are read from numbers.tex."""
    tmlr = os.path.join(pkg, "paper", "tmlr")
    nv = dict(re.findall(r"\\defnv\{([^}]+)\}\{([^}]*)\}", open(os.path.join(tmlr, "numbers.tex"), encoding="utf-8").read()))
    p = os.path.join(tmlr, "appendix", "app_A.tex")
    if not os.path.exists(p):
        return []
    want = []
    for line in open(p, encoding="utf-8"):
        cols = [c.strip() for c in line.split(" & ")]
        if len(cols) < 3 or not re.search(r"[0-9a-f]{12}|\\nv\{plan\.[^}]*sha", cols[2]):
            continue
        hashes = [nv.get(m.group(1), "?") if m.group(1) else m.group(0)
                  for m in re.finditer(r"\\nv\{([^}]+)\}|[0-9a-f]{12}", cols[2])]
        labels = [x.strip() for x in cols[0].split(",")]
        labels = labels if len(labels) == len(hashes) else [cols[0]] * len(hashes)
        when = re.sub(r"\\nv\{([^}]+)\}", lambda m: nv.get(m.group(1), "?"), cols[1]).strip()
        times = [x.strip() for x in when.split(",")]
        for i in range(1, len(times)):  # "2026-10-22 09:00, 09:30": the later times carry no date
            if re.fullmatch(r"\d\d:\d\d", times[i]) and re.match(r"\d{4}-\d\d-\d\d", times[0]):
                times[i] = times[0][:10] + " " + times[i]
        if len(times) != len(hashes):
            times = [when] * len(hashes)
        want += [(a, t, h) for a, t, h in zip(labels, times, hashes)]
    seen, uniq = set(), []
    for w in want:                      # the table may appear in both branches of the arXiv switch
        if w[2] not in seen:
            seen.add(w[2])
            uniq.append(w)
    want = uniq

    def cut(b, key):
        """The whole file (key None) or the file cut before line j + 1 in one of three newline forms (key (j, form))."""
        if key is None:
            return b
        j, form = key
        pre = "\n".join(b.decode("utf-8").split("\n")[:j]) + "\n"
        return (pre, pre[:-1], pre.rstrip("\n") + "\n")[form].encode()

    def index(b):
        """First 12 hex digits of the SHA-256 of every cut -> key, for one document."""
        idx = {sha(b)[:12]: None}
        lines = b.decode("utf-8").split("\n")
        for j in range(1, len(lines)):
            pre = "\n".join(lines[:j]) + "\n"
            for form, t in enumerate((pre, pre[:-1], pre.rstrip("\n") + "\n")):
                idx.setdefault(sha(t.encode())[:12], (j, form))
        return idx

    def git_versions(d):
        """Committed versions of a document (bytes), newest first; empty without git."""
        if not git_src:
            return []
        import subprocess
        try:
            cs = subprocess.run(["git", "-C", git_src, "log", "--format=%H", "--", d], capture_output=True,
                                text=True, timeout=60).stdout.split()
            return [subprocess.run(["git", "-C", git_src, "show", f"{c}:{d}"], capture_output=True,
                                   timeout=60).stdout for c in cs]
        except (OSError, subprocess.SubprocessError):
            return []

    def deleted_ranges(version, current):
        """Line ranges (1-based, of current) whose removal turns current into version, if version is current with
        some lines inserted later (no other edits); else None."""
        import difflib
        v, c = version.decode("utf-8").split("\n"), current.decode("utf-8").split("\n")
        ops = difflib.SequenceMatcher(None, v, c, autojunk=False).get_opcodes()
        if any(op not in ("equal", "insert") for op, *_ in ops):
            return None
        return [(j1 + 1, j2) for op, i1, i2, j1, j2 in ops if op == "insert"]

    def without(b, ranges):
        lines = b.decode("utf-8").split("\n")
        drop = {k for a, z in ranges for k in range(a, z + 1)}
        return "\n".join(x for k, x in enumerate(lines, 1) if k not in drop).encode()

    def fmt(ranges):
        return ", ".join(f"{a}" if a == z else f"{a}-{z}" for a, z in ranges)

    logs = ""
    for lp in ("main/LOG.md", "rss_ext/LOG.md", "tmlr_ext/LOG.md"):
        if snapshot and os.path.exists(os.path.join(snapshot, lp)):
            logs += open(os.path.join(snapshot, lp), encoding="utf-8").read()
    texts = {}
    for d in docs:
        po, pc = os.path.join(snapshot or pkg, d), os.path.join(pkg, d)
        if os.path.exists(po) and os.path.exists(pc):
            bo = open(po, "rb").read()
            texts[d] = (bo, open(pc, "rb").read(), index(bo))
    rows = []
    for label, when, h in want:
        label = re.sub(r"\\[a-zA-Z]+\{?|[{}$]", "", label).replace("--", "-")
        full, doc, status = None, None, None
        for d, (bo, bc, idx) in texts.items():      # 1. the original document or a cut of it
            if h[:12] in idx:
                key = idx[h[:12]]
                how = "whole file" if key is None else f"file cut before line {key[0] + 1}"
                full, doc = sha(cut(bo, key)), d
                hc = sha(cut(bc, key))
                status = (f"unchanged copy; reproduced by the {how}" if hc == full else
                          f"**redacted copy**: the {how} of this copy has SHA-256 `{hc}`; the original's is the "
                          f"logged one")
                break
        if not status:                    # 2. a committed version that later insertions changed in place
            for d, (bo, bc, idx) in texts.items():
                for v in git_versions(d):
                    if sha(v).startswith(h):
                        rg = deleted_ranges(v, bo)
                        if rg is not None and without(bo, rg) == v:
                            full, doc = sha(v), d
                            hc = sha(without(bc, rg))
                            status = (f"unchanged copy; reproduced by the file without lines {fmt(rg)}, which were "
                                      f"inserted later" if hc == full else
                                      f"**redacted copy**: this copy without lines {fmt(rg)} has SHA-256 `{hc}`; "
                                      f"the original's is the logged one")
                        break
                if status:
                    break
        if not full:                      # 3. only the logged value
            m = re.findall(h + r"[0-9a-f]{52}", logs)
            full = m[0] if m else h
            status = "not reproducible from the current documents; the hash is the one logged at the freeze"
        rows.append((label, when, full, f"`{doc}`" if doc else "", status))
    return rows


def excluded_summary(snapshot, included):
    by, n, b = {}, 0, 0
    for top in ("main", "rss_ext", "tmlr_ext"):
        for dp, dns, fns in os.walk(os.path.join(snapshot, top)):
            dns[:] = [x for x in dns if x != "__pycache__"]
            for f in fns:
                rel = os.path.relpath(os.path.join(dp, f), snapshot)
                if rel in included or "/results/" not in "/" + rel or any(re.search(x, rel) for x in EXCLUDE):
                    continue
                s = os.path.getsize(os.path.join(dp, f))
                by[f"`{top}`"] = by.get(f"`{top}`", 0) + s
                n += 1
                b += s
    return {"n": n, "bytes": b, "by": by}


# ================================================================================================ scan
def cmd_scan(argv):
    pkg, out = os.path.abspath(argv[0]), argv[1]
    snapshot = argv[2] if len(argv) > 2 else pkg
    P = scan_patterns(snapshot)
    hits, nfiles, kinds = [], 0, {}
    for dp, dns, fns in os.walk(pkg):
        for f in fns:
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, pkg)
            nfiles += 1
            for label, s, ctx in scan_text(rel, P):
                hits.append({"file": rel, "where": "file name", "label": label, "match": s, "context": ctx})
            data = open(path, "rb").read()
            ext = os.path.splitext(f[:-3] if f.endswith(".gz") else f)[1].lower()
            if f.endswith(".gz"):
                try:
                    data = gzip.decompress(data)
                    k = "gz"
                except OSError:
                    k = "binary"
            else:
                k = "text" if is_text_bytes(data, ext) else "binary"
            kinds[k] = kinds.get(k, 0) + 1
            texts = []
            if k == "binary":
                texts += [(data.decode("latin-1"), True), (data.decode("utf-8", "ignore"), True)]
                if f.endswith(".pdf"):              # metadata, page text and every decompressed stream of a PDF
                    try:
                        import pymupdf
                        with pymupdf.open(path) as doc:
                            parts = [json.dumps(doc.metadata)] + [pg.get_text() for pg in doc]
                            for x in range(1, doc.xref_length()):
                                if doc.xref_is_stream(x):
                                    parts.append(doc.xref_stream(x).decode("latin-1"))
                                parts.append(doc.xref_object(x))
                        texts.append(("\n".join(parts), False))
                    except ImportError:
                        pass
            else:
                texts.append((data.decode("utf-8", "replace"), False))
            seen = set()
            for t, binary in texts:
                for label, s, ctx in scan_text(t, P, binary):
                    if (label, s) in seen:
                        continue
                    seen.add((label, s))
                    hits.append({"file": rel, "where": k, "label": label, "match": s, "context": ctx[:200]})
    json.dump({"files": nfiles, "kinds": kinds, "hits": hits}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"scanned {nfiles} files ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}): {len(hits)} hits")
    for h in hits[:40]:
        print(f"  HIT {h['label']:14s} {h['file']} [{h['where']}]: {h['match']!r}  ...{h['context'][:120]}...")
    sys.exit(1 if hits else 0)


# ================================================================================================ compare
def _outputs(root, pdf_only=False):
    out = {"numbers.tex"} if os.path.exists(os.path.join(root, "numbers.tex")) else set()
    for d in ("tables", "figures"):
        if os.path.isdir(os.path.join(root, d)):
            out.update(os.path.join(d, f) for f in os.listdir(os.path.join(root, d))
                       if not f.startswith(".") and not (pdf_only and f.endswith(".png")))
    return out


def cmd_compare(argv):
    """compare TEST NAME=DIR[:must] ...: numbers.tex, tables/ and figures/ of TEST against each DIR. A difference
    against a DIR marked ':must' fails (exit 1); the others are reported only. Figure PNG previews are compared only
    when both sides have them."""
    test, res, bad = argv[0], [], 0
    for spec in argv[1:]:
        label, other = spec.split("=", 1)
        must = other.endswith(":must")
        other = other[:-5] if must else other
        if not os.path.isdir(other):
            continue
        fa, fb = _outputs(other), _outputs(test)
        common = fa & fb
        same = sorted(f for f in common if open(os.path.join(other, f), "rb").read() == open(os.path.join(test, f), "rb").read())
        diff = sorted(f for f in common if f not in same)
        only_a = sorted(f for f in fa - fb if not f.endswith(".png"))
        only_b = sorted(f for f in fb - fa if not f.endswith(".png"))
        res.append(f"{label}: {len(same)} identical, {len(diff)} different, {len(only_a)} missing in the test, "
                   f"{len(only_b)} new in the test")
        for f in diff:
            res.append(f"  DIFFERENT {f}")
            if f.endswith(".tex"):
                import difflib
                a = open(os.path.join(other, f), encoding="utf-8").read().split("\n")
                b = open(os.path.join(test, f), encoding="utf-8").read().split("\n")
                for line in list(difflib.unified_diff(a, b, lineterm="", n=0))[2:14]:
                    res.append("    " + line[:200])
        res += [f"  MISSING IN TEST {f}" for f in only_a] + [f"  NEW IN TEST {f}" for f in only_b]
        if must:
            bad += len(diff) + len(only_a) + len(only_b)
    print("\n".join(res))
    sys.exit(1 if bad else 0)


def cmd_report(argv):
    """report WORK OUT STATUS...: write OUT/BUILD_REPORT.md from the work folder (record, scan, logs)."""
    work, out = argv[0], argv[1]
    st = dict(a.split("=", 1) for a in argv[2:])
    rec = json.load(open(os.path.join(work, "assemble_record.json")))
    scan = json.load(open(os.path.join(work, "scan.json")))
    files = rec["files"]
    total = sum(r["bytes"] for r in files)
    changed = [r for r in files if r["changed"]]
    L = ["# Build report: anonymized supplementary material (paper A, TMLR)", "",
         f"- built: {st.get('date', '')}; source: `{st.get('src', '')}` (git {st.get('git', '?')})",
         f"- result: **{st.get('result', '?')}**",
         f"- package: {len(files)} files, {total / 1e6:.2f} MB unpacked; zip {int(st.get('zip_bytes', 0)) / 1e6:.2f} MB "
         f"(TMLR limit 100 MB)",
         f"- baseline (unmodified snapshot): exit codes {st.get('baseline', '')}",
         f"- scan: {scan['files']} files ({', '.join(f'{v} {k}' for k, v in sorted(scan['kinds'].items()))}), "
         f"{len(scan['hits'])} identifying strings left",
         f"- clean-folder test: three commands exit {st.get('test', '?')}, make_figs exit {st.get('figs', '?')}, "
         f"comparison exit {st.get('compare', '?')}" + (f", manuscript text {st['tex']}" if st.get("tex") else ""), ""]
    if scan["hits"]:
        L += ["## Identifying strings left", "", "| File | Where | Kind | Match | Context |", "|---|---|---|---|---|"]
        for h in scan["hits"]:
            L.append(f"| `{h['file']}` | {h['where']} | {h['label']} | `{h['match']}` | {h['context'][:120]} |")
        L.append("")
    for name in ("compare.txt", "test_commands.txt", "test_make_figs.txt", "tex_compare.txt"):
        p = os.path.join(work, "logs", name)
        if os.path.exists(p):
            body = open(p, encoding="utf-8", errors="replace").read().rstrip()
            L += [f"## {name}", "", "```", body[-6000:], "```", ""]
    outside = set()
    for f in os.listdir(os.path.join(work, "trace")):
        outside.update(json.load(open(os.path.join(work, "trace", f))).get("outside_reads", []))
    L += ["## Files read outside the package by the baseline run (non-system)", ""]
    L += [f"- `{x}`" for x in sorted(outside)] or ["- none"]
    L += ["", f"## Files changed by the anonymization ({len(changed)})", "",
          "| File | Replaced | SHA-256 original | SHA-256 copy |", "|---|---|---|---|"]
    for r in changed:
        what = ", ".join(f"{k} {v}" for k, v in sorted(r["rules"].items()))
        L.append(f"| `{r['path']}` | {what} | `{r['sha256_original'][:16]}` | `{r['sha256_copy'][:16]}` |")
    L += ["", "Design documents whose hash checks in make_numbers.py use build/redacted_hashes.json: "
          + (", ".join(f"`{d}`" for d in rec["hashed_docs"]) or "none"), ""]
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "BUILD_REPORT.md"), "w", encoding="utf-8").write("\n".join(L))


def cmd_pdftext(argv):
    import pymupdf

    def text(p):
        with pymupdf.open(p) as d:
            return [pg.get_text() for pg in d]
    a, b = text(argv[0]), text(argv[1])
    same = a == b
    print(f"pages {len(a)} vs {len(b)}; text {'identical' if same else 'DIFFERENT'}")
    if not same:
        import difflib
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  page {i + 1}:")
                for line in list(difflib.unified_diff(x.split("\n"), y.split("\n"), lineterm="", n=0))[2:10]:
                    print("    " + line[:160])
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "trace":
        cmd_trace(args)
    elif cmd == "provenance":
        cmd_trace(args, provenance=True)
    elif cmd == "assemble":
        cmd_assemble(args)
    elif cmd == "scan":
        cmd_scan(args)
    elif cmd == "compare":
        cmd_compare(args)
    elif cmd == "report":
        cmd_report(args)
    elif cmd == "pdftext":
        cmd_pdftext(args)
    else:
        raise SystemExit(__doc__)
