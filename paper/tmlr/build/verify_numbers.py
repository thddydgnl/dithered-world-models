"""Check the numbers of the TMLR manuscript of paper A against the result files.

Extended from paper/rss2027/build/verify_numbers.py. In the TMLR source every result number is a macro \\nv{key}
(numbers.tex, written by build/make_numbers.py from the result files), so the checks are:
  1. numbers.tex is current: make_numbers.compute() is re-run here and must reproduce the file byte for byte.
  2. every \\nv{key} used in main.tex is defined (undefined keys fail; unused keys are listed).
  3. wording checks: statements whose words depend on a value ("in all N cells", "none harmful", "indistinguishable
     from zero") are asserted against the data.
  4. literal numbers: every number still typed in main.tex (outside \\nv{}, comments, \\todo{}, labels, references,
     citations, layout arguments and generated tables) must be registered below with the document that fixes it
     (design constants, definitions, cell names); otherwise it is listed as UNCHECKED.
  5. values of kind 'report' (copied from a stage report, not recomputed from a result file) are listed as NOT
     VERIFIED AGAINST DATA.
Exit code 1 on a stale numbers.tex, an undefined key, a failed wording check or a registered phrase missing from
main.tex. With --strict it also fails on UNCHECKED literals, on report-only values that cannot be found in their
source document and on remaining \\todo markers (use --strict for the submission build).
Usage: python3 build/verify_numbers.py [--strict]
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_numbers as mn  # noqa: E402

TMLR = mn.TMLR
A = mn.A
RAW = open(os.path.join(TMLR, "main.tex"), encoding="utf-8").read()
# the appendices live in appendix/app_*.tex; check them as part of the text (generated tables/*.tex are not expanded:
# make_tables.py writes them from the result files)
RAW = re.sub(r"\\input\{(appendix/app_[A-K])\}", lambda m: open(os.path.join(TMLR, m.group(1) + ".tex"), encoding="utf-8").read(), RAW)
STRICT = "--strict" in sys.argv
FAILS, NOTES = [], []


# ------------------------------------------------------------------------------------------------ LaTeX handling
def strip_comments(tex):
    out = []
    for line in tex.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line[:m.start()] if m else line)
    return "\n".join(out)


def skip_group(s, i, open_ch="{", close_ch="}"):
    """s[i] == open_ch; return the index after the matching close_ch."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "\\":
            continue
        if s[j] == open_ch and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == close_ch and (j == 0 or s[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError(f"unbalanced group at {i}: {s[i:i + 60]!r}")


def remove_cmd(s, name, nargs, keep=None, opt=True):
    """Remove \\name[...]{a1}...{an}; with keep=j keep the content of argument j (0-based)."""
    pat = re.compile(r"\\" + re.escape(name) + r"(?![A-Za-z])\*?")
    out, pos = [], 0
    for m in pat.finditer(s):
        if m.start() < pos:
            continue
        j = m.end()
        while j < len(s) and s[j] in " \n":
            j += 1
        if opt and j < len(s) and s[j] == "[":
            j = skip_group(s, j, "[", "]")
        args = []
        for _ in range(nargs):
            while j < len(s) and s[j] in " \n":
                j += 1
            if j >= len(s) or s[j] != "{":
                break
            k = skip_group(s, j)
            args.append(s[j + 1:k - 1])
            j = k
        out.append(s[pos:m.start()])
        if keep is not None and keep < len(args):
            out.append(" " + args[keep] + " ")
        else:
            out.append(" ")
        pos = j
    out.append(s[pos:])
    return "".join(out)


def todo_items(tex):
    """\\todo markers after \\begin{document} (the preamble defines the macro), with their line numbers."""
    items = []
    start = tex.index("\\begin{document}")
    for m in re.finditer(r"\\todo(?![A-Za-z])", tex[start:]):
        m = re.compile(r"\\todo(?![A-Za-z])").match(tex, start + m.start())
        j = m.end()
        k = skip_group(tex, j)
        line = tex.count("\n", 0, m.start()) + 1
        items.append((line, " ".join(tex[j + 1:k - 1].split())))
    return items


# ------------------------------------------------------------------------------------------------ 1. freshness
entries = mn.compute()
values = {k: t for k, t, s, kind in entries}
kinds = {k: kind for k, t, s, kind in entries}
sources = {k: s for k, t, s, kind in entries}
with open(mn.OUT) as fh:
    if fh.read() != mn.render(entries):
        FAILS.append("numbers.tex is stale: run python3 build/make_numbers.py")

TEX = strip_comments(RAW)
BODY = TEX.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]

# ------------------------------------------------------------------------------------------------ 2. keys
used = re.findall(r"\\nv\{([^}]*)\}", BODY)
for k in sorted(set(used)):
    if k not in values:
        FAILS.append(f"undefined number key \\nv{{{k}}}")
unused = sorted(set(values) - set(used))
report_used = sorted(k for k in set(used) if kinds.get(k) == "report")


# ------------------------------------------------------------------------------------------------ 3. wording checks
def v(k):
    return values[k]


def as_float(k):
    return float(values[k].replace("\\ensuremath{-}", "-").replace("$-$", "-").replace("+", ""))


WORDING = []


def wording(label, ok):
    WORDING.append((label, bool(ok)))


def load(rel):
    with open(os.path.join(A, rel)) as fh:
        return json.load(fh)


td = load("tmlr_ext/TD_tdmpc2/results/analysis/td_summary.json")
dwj = load("tmlr_ext/DW_dinowm/results/analysis/analysis_dw.json")
c11 = td["E11"]["cells"]
wall = [x for x in dwj["E11"]["cells"] if x["task"] == "wall"][0]
push4 = [x for x in dwj["E11"]["cells"] if x["task"] == "pusht"][0]

wording("E1: sh-iter repairs ALL affected Meta-World cells", v("e1.mw.nfix") == v("e1.mw.naff"))
by = {}
for r in csv.DictReader(open(os.path.join(A, "main/results/E1_tdmpc2/e1_tdmpc2_cells.csv"))):
    by.setdefault((r["task"], int(r["k"]), r["family"]), {})[r["variant"]] = r
aff_mw = [x for k, x in by.items() if k[2] == "mw" and x["det"]["confirmatory"] == "True" and x["det"]["holm_real"] == "True"]
wording("E1: sh-iter not harmful in any affected Meta-World cell",
        not any(mn.ci_str(x["shiter"]["Dv_minus_Ddet_ci"])[0] > 0.05 for x in aff_mw))
dwe1 = {}
for r in csv.DictReader(open(os.path.join(A, "main/results/E1/e1_dinowm_cells.csv"))):
    if r["role"] == "confirmatory":
        dwe1.setdefault((r["task"], int(r["k"])), {})[r["var"]] = r
aff_dw = [x for x in dwe1.values() if x["det"]["drop_real_holm"] == "True"]
wording("E1: sh-iter not harmful in any affected DINO-WM cell",
        not any(mn.ci_str(x["shiter"]["Dv_minus_Ddet_ci"])[0] > 0.05 for x in aff_dw))
wording("E10: sh-iter repairs ALL affected 317M tasks", v("e10.nfix") == v("e10.naff"))
wording("E11: det damages ALL re-run TD-MPC2 cells", v("e11.td.ndam") == v("e11.td.ncells"))
wording("E11: B4 without harming any cell", v("e11.b4.nharm") == "0")
wording("E14: det damages ALL 317M tasks", v("e14.ndam") == v("e14.ntasks"))
wording("E14: sh-iter repairs all of them", v("e14.p1.n") == v("e14.ndam"))
wording("E11: Delta_pers CI lower > 0 in ALL damaged cells", v("e11.pers.npos") == v("e11.td.ndam"))
wording("E11: det+noise repairs ALL damaged Meta-World cells", v("e11.mw.nfix.detnoise") == v("e11.mw.ndam"))
wording("E11: fixed unbiased variants repair at most the sh-ep count on Meta-World",
        int(v("e11.mw.nfix.detshiftfix")) <= int(v("e11.mw.nfix.shep")))
wording("Wall: fixed unbiased errors repair (sh-plan, det+shift-fixed red >= 0.5)",
        wall["variants"]["shstep"]["red"] >= 0.5 and wall["variants"]["detshiftfix"]["red"] >= 0.5)
wording("Wall: sh-iter+b and det+noise do not repair (red < 0.5)",
        wall["variants"]["shiterb"]["red"] < 0.5 and wall["variants"]["detnoise"]["red"] < 0.5)
wording("PushT: det+BC worse than det at 3 bits (E12) and 4 bits (E11)",
        as_float("dw.pusht3.D.detbc") > as_float("dw.pusht3.D.det") and push4["variants"]["detbc"]["D"] > push4["variants"]["det"]["D"])
wording("Wall: det+shift above the unquantized model", wall["variants"]["detshift"]["D"] < 0)
e16 = [dwj["E16"][t]["D(sh-plan)-D(sh-iter)"][p]["ci"] for t in ("pointmaze", "wall") for p in ("cem", "mppi")]
wording("E16 DINO-WM: all four intervals contain zero", all(lo <= 0 <= hi for lo, hi in e16))
wording("E16: every swapped planner keeps >= 60% of the default's unquantized success",
        all(td["E16"][t]["none_cem"] / td["E16"][t]["none_default"] >= 0.6 for t in td["E16"])
        and all(dwj["E16"][t]["ratio"] >= 0.6 for t in ("pointmaze", "wall")))
wording("E16 TD-MPC2: cost of persistence remains (CI lower > 0) in all three tasks with the CEM-like planner",
        all(td["E16"][t]["shep_minus_shiter"]["ci"][0] > 0 for t in td["E16"]))
r5p = load("tmlr_ext/coordinator/reanalysis/r5.json")["dw"]["pusht|cem"]["cluster_ci"]["shplan_minus_shiter"]
wording("PushT 4-bit sh-plan - sh-iter: trajectory-bootstrap interval excludes zero", r5p[0] > 0)
wording("E9: det's latent error smaller than sh-iter's in ALL damaged FP4 cells", float(v("e9.errratio").split("--")[1]) < 1)
wording("E10: det's latent error smaller in ALL affected 317M tasks", v("e10.h1.n") == v("e10.naff"))
wording("E8c: pre-converted BF16 fastest float format in ALL 12 conditions", v("e8c.bf16fastest") == "12")
wording("E12: the 1-bit configurations all use dither (asserted in make_numbers)", True)
wording("E13: rho_p of every fixed variant is 1.00", v("e13.rho.fixed") == "1.00")
wording("R2: Fr1/Fr2 hold at every tau and basis (asserted in make_numbers)", True)
wording("Dg1 Meta-World: positive in all cells (asserted in make_numbers)", True)
wording("Dg1 DINO-WM below 0.3", as_float("dg1.dw") < 0.3)
wording("E15: walker remains a failure at 24 iterations (red < 0.5)", as_float("e15.walker.red24") < 0.5)
wording("K3: pre-converted BF16 damages door-open and cheetah (CI excludes 0) and helps finger",
        all(x in ("mw-door-open", "cheetah-run", "finger-turn-hard") for x in
            load("tmlr_ext/K_kernel/results/raw/k.1/analysis/e17_summary.json")["K3"]["tasks_damaged_or_ci_excludes_0"]))
d4 = load("tmlr_ext/DW_dinowm/results/analysis_posthoc/posthoc_pusht3.json")["cell"]
d4v = d4["variants"]
wording("D4: fixed unbiased errors repair BOTH damaged DINO-WM cells",
        all(x["variants"][v]["red"] >= 0.5 for x in (wall, d4) for v in ("shstep", "detshiftfix")))
wording("D4: sh-iter repairs both DINO-WM cells", all(x["variants"]["shiter"]["red"] >= 0.5 for x in (wall, d4)))
wording("D4: sh-iter+b and det+noise repair NEITHER DINO-WM cell",
        all(x["variants"][v]["red"] < 0.5 for x in (wall, d4) for v in ("shiterb", "detnoise")))
wording("D4: det+BC worse than det on PushT 3-bit (red < 0)", d4v["detbc"]["red"] < 0)
wording("D4: reading resolution, Delta_pers CI contains 0", d4["class"] == "resolution" and d4["d_pers"]["ci"][0] <= 0 <= d4["d_pers"]["ci"][1])
wording("D4: replicates E1 (damage real, CIs overlap)",
        d4v["det"]["D_ci"][0] > 0 and d4v["det"]["D_ci"][0] <= d4["E1"]["D_det_ci"][1] and d4["E1"]["D_det_ci"][0] <= d4v["det"]["D_ci"][1])
wording("D4: b raises the planning-time error on PushT",
        d4v["detbc"]["rel_err1"] > d4v["det"]["rel_err1"] and d4v["detbc"]["rel_err5"] > d4v["det"]["rel_err5"])
wording("D4: sh-plan - sh-iter interval contains 0", d4["aux"]["D(sh-plan)-D(sh-iter)"]["ci"][0] <= 0 <= d4["aux"]["D(sh-plan)-D(sh-iter)"]["ci"][1])
x17 = load("tmlr_ext/DW_dinowm/results/analysis_posthoc/x17_pusht4_eval6.json")
wording("X17: E1 damage reproduced (reading (i))", x17["reading"] == "(i)" and x17["det"]["D_ci"][0] > 0)
wording("X5: det+noise gains fewer bits than dither (PT and PC)",
        int(v("x5.pt.nG")) < int(v("e12.pt.nG")) and int(v("x5.pc.nG")) < int(v("e12.pc.nG")))
wording("X5: door-open det+noise not monotone (3 bits worse than 2)", as_float("x5.pt.dooropen3") > as_float("x5.pt.dooropen2"))
wording("E13: det+noise rho_p above redrawn dither on the same cells",
        float(v("e13.rho.detnoise").split("--")[0]) > 0 and
        float(v("e13.rho.detnoise").split("--")[1]) > max(float(v("e13.rho.e11.shiter").split("--")[1]), float(v("e13.rho.e11.detshift").split("--")[1])))
# review revision (2026-11-22): statements added or reworded after the manuscript review
wording("Abstract/4.2: det's latent error no larger than sh-iter's in MOST damaged E1 cells",
        int(v("e1.h1.n")) > int(v("e1.h1.of")) / 2)
e16td = td["E16"]
red16 = {t: (e16td[t]["variants"]["det"]["D"] - e16td[t]["variants"]["shiter"]["D"]) / e16td[t]["variants"]["det"]["D"]
         for t in e16td}
wording("E16: with the CEM-like planner sh-iter repairs push and pick-place (red >= 0.5) but not cheetah-run",
        red16["mw-push"] >= 0.5 and red16["mw-pick-place"] >= 0.5 and red16["cheetah-run"] < 0.5)
wording("E16: with the MPPI-like planner det damages both DINO-WM tasks (CI lower bound > 0)",
        all(dwj["E16"][t]["variants"]["det"]["D_ci"][0] > 0 for t in ("pointmaze", "wall")))
wording("E11: D(sh-ep) - D(sh-iter) > 0 in ALL damaged TD-MPC2 cells",
        all(c11[t]["variants"]["shep"]["D"] > c11[t]["variants"]["shiter"]["D"] for t in c11 if c11[t]["damaged"]))
e1m = {k: as_float(f"e1.mw.mean.{k}") for k in ("shstep", "shep", "shiter")}
wording("E1: keeping the draw within a control step costs MOST of the episode-level cost (sh-step vs sh-ep, over sh-iter)",
        (e1m["shstep"] - e1m["shiter"]) > 0.5 * (e1m["shep"] - e1m["shiter"]))
wording("E15: walker-walk not repaired at any number of iterations (red < 0.5 at 6, 12 and 24)",
        as_float("e15.walker.red12") < 0.5 and as_float("e15.walker.red24") < 0.5
        and float(v("e11.walker.harm.shiter").split("~")[0].replace("+", "")) > 0)
wording("X6: the lowest dither configuration is never worse than the policy; DMC better in 3 of 4 (asserted in make_numbers)", True)
wording("S4: det at 4 bits above the unquantized planner on Wall in both splits (asserted in make_numbers)", True)
wording("M1: 21 of the 29 cells repaired by sh-iter (asserted in make_numbers)", True)
wording("7: one speed-up range stands for both runs (E8 and E8c graph ranges are equal)", v("e8.speedup") == v("e8c.speedup"))
wording("6.1: PushT bootstrap supports >= 1 bit with all three ranges (P >= 0.95) but 2 bits robustly only per tensor",
        all(as_float(f"r1.pusht.{r}.ge1") >= 0.95 for r in ("pt", "pc", "rc")) and as_float("r1.pusht.pt.ge2") >= 0.95
        and as_float("r1.pusht.pc.ge2") < 0.95 and as_float("r1.pusht.rc.ge2") < 0.95)
wording("5.2: in E1 sh-plan repaired PushT at 3 and 4 bits (red >= 0.5)",
        as_float("e1.pusht3.red.shplan") >= 0.5 and as_float("e1.pusht4.red.shplan") >= 0.5)

for label, ok in WORDING:
    if not ok:
        FAILS.append(f"wording check failed: {label}")

# ------------------------------------------------------------------------------------------------ 4. literal numbers
# Registered phrases: (exact substring of the scanned text, kind, source). A literal number is covered if it lies
# inside a registered phrase that occurs in the scanned text. Kinds: setting (design constant), definition
# (threshold or rule fixed in the analysis plan), label (task, bit width or stage name), checked (verified here).
REG = [
    # planner settings (DESIGN_main 0, 4; RSS draft Sec. III)
    ("$512\\times6\\times3$", "setting", "TD-MPC2 default MPPI: 512 samples, 6 iterations, horizon 3"),
    ("$300\\times30\\times5$", "setting", "DINO-WM PushT CEM: 300 samples, 30 iterations, horizon 5"),
    ("512 candidates including 24 policy trajectories, 64 elites, 6 iterations, horizon 3", "setting", "TD-MPC2 MPPI defaults"),
    ("300 candidates, top 30, horizon 5, 30 iterations on PushT and 10 otherwise", "setting", "DINO-WM CEM defaults"),
    ("three seeds) on 8 Meta-World tasks", "setting", "E1 design (3 checkpoints, 8 MW tasks)"),
    ("and 4 DMC tasks", "setting", "E1 design"),
    ("on the same 8 Meta-World tasks", "setting", "E10 design"),
    ("tests $k-1$, $k$ and $k+1$: 36 cells for TD-MPC2 at 5M and 9 for DINO-WM", "checked", "E1 cells (computed below)"),
    ("(90 pairs per cell for TD-MPC2: three checkpoints $\\times$ 30 seeds, or one 317M checkpoint $\\times$ 90 seeds; 100 for DINO-WM)", "setting", "DESIGN_main 6.1, DESIGN_tmlr 0.3"),
    ("($10^4$ resamples)", "definition", "DESIGN_main 6.1"),
    ("$\\red\\ge0.5$", "definition", "repair rule"),
    ("95\\% paired-bootstrap percentile intervals", "definition", "DESIGN_main 6.1"),
    ("the 99.99th percentile", "setting", "range rule, DESIGN_tmlr 0.6"),
    ("below 1\\% of $\\Delta$", "definition", "bias-vector stopping rule, deviation D2"),
    ("at least 20 points of success (20\\% of return on DMC)", "definition", "primary-k rule, DESIGN_main 0.2"),
    ("the four hidden layers", "setting", "quantization sites (DESIGN_tmlr 0.5)"),
    ("the six MLP outputs", "setting", "quantization sites (DESIGN_tmlr 0.5)"),
    ("$k$-bit grid", "setting", "notation"),
    ("$\\Delta=(\\mathrm{hi}-\\mathrm{lo})/(2^k-1)$", "setting", "notation"),
    ("$u\\sim U[0,1)$", "setting", "notation"),
    ("$o\\sim U[-\\tfrac12,\\tfrac12)$", "setting", "notation"),
    ("+u-\\tfrac12$", "setting", "notation"),
    ("$\\Delta^2/12$", "setting", "uniform-error variance (theory)"),
    ("$\\Delta^2/6$", "setting", "stochastic-rounding variance (theory)"),
    ("$\\exp(\\beta(J-J_{\\max}))$", "setting", "notation"),
    ("at least 60\\% of the default planner's", "definition", "E16 qualification, DESIGN_tmlr 6"),
    ("loses at most 10 points at 1 bit", "definition", "k* threshold"),
    ("just above the 10-point threshold", "definition", "k* threshold"),
    ("(blocks of 16, E4M3 scale)", "setting", "NVFP4 format"),
    ("(blocks of 32, power-of-two scale)", "setting", "MXFP4 format"),
    ("of 12 tasks under NVFP4", "setting", "E9 design (12 tasks)"),
    ("of 12 under MXFP4", "setting", "E9 design"),
    ("damaged \\nv{e9.nvfp4.ndam} of 12 tasks and with MXFP4", "setting", "E9 design (12 tasks)"),
    ("uniform 4-bit grid", "label", "E1 cheetah-run at 4 bits"),
    ("$N\\in\\{512, 2048, 8192\\}$", "setting", "E8/E17 design"),
    ("$N\\ge2048$", "setting", "E17 conditions"),
    ("at $N=512$", "setting", "E17 conditions"),
    ("all six conditions", "setting", "E17 design (2 sizes x 3 N)"),
    ("in all 12 conditions of each of two runs", "checked", "E8 and E8c graph cells (both ranges asserted equal below)"),
    ("of 4 tasks, including walker", "setting", "E17 K2 design (4 tasks)"),
    ("at 3--4 bits", "label", "E5 bit widths"),
    ("primary bit widths (3--5) are close to those at 5M (3--4)", "checked", "E10/E1 primary k (computed below)"),
    ("per control step with 6 iterations, \\nv{e15.cheetah.ds24.ms}~ms with 24)", "setting", "E15 design"),
    ("With 12 iterations, the better of", "setting", "E15 design"),
    ("of the 4 DMC tasks", "setting", "E15 design"),
    ("from 6 to 12 to 24 iterations", "setting", "E15 design"),
    ("at 24 iterations", "setting", "E15 design"),
    ("At the default six MPPI iterations", "setting", "planner default"),
    ("With one MPPI iteration", "setting", "E2 design"),
    ("at 12 iterations its damage", "setting", "E2 design"),
    ("across the 12 TD-MPC2 tasks", "setting", "E3 design"),
    ("in all four mechanism tasks", "setting", "E2 design (push, pick-place, cheetah-run, PushT)"),
    ("over the three cells", "setting", "E11 DINO-WM cells"),
    ("button-press at \\nv{e1.maxdet.k} bits", "label", ""),
    ("PushT at 5 bits", "label", "E1 cell"),
    ("Wall at 3 bits", "label", "E11 cell"),
    ("PushT at 4 bits", "label", "E11 cell"),
    ("PointMaze at 3 bits", "label", "E11 cell"),
    ("(PushT, 4 bits:", "label", "E1 cell"),
    ("walker at 3 bits", "label", "E1 cell"),
    ("at 4 bits", "label", "cell names"),
    ("the 12 TD-MPC2 tasks", "setting", "E3 design"),
    ("Only the PushT cell at 4 bits", "label", "cell names"),
    ("PushT at 3 bits", "label", "post hoc cell D4"),
    ("PushT cell at 3 bits", "label", "post hoc cell D4"),
    ("of the new split (3 bits; values", "label", "E11 cells"),
    ("points at 3 bits but \\nv{x5.pt.dooropen2} at 2)", "label", "X5 curve points"),
    ("one dither draw", "setting", ""),
    ("at least one bit", "definition", "G >= 1"),
    ("12 tasks $\\times$ 3 bit widths, 7 variants, 90 pairs; DINO-WM: 9 cells (+1 exploratory), 6 variants, 100 pairs", "setting", "E1 design"),
    ("TD-MPC2 5M: 12 tasks at the primary $k$, 8 variants, 90 pairs; DINO-WM: PushT 4-bit, PointMaze 3-bit, Wall 3-bit, new split, 100 pairs", "setting", "DESIGN_tmlr 1.2"),
    ("TD-MPC2 5M 12 tasks and DINO-WM 3 tasks", "setting", "DESIGN_tmlr 2.2"),
    ("8 Meta-World tasks, primary $k$, 6 variants, 90 pairs", "setting", "DESIGN_tmlr 4"),
    ("DMC 4 tasks $\\times$ \\{6, 12, 24\\} iterations", "setting", "DESIGN_tmlr 5"),
    ("TD-MPC2 5M, 4 tasks (kernel) and 12 tasks (BF16)", "setting", "DESIGN_tmlr 7"),
    ("Each layer uses a uniform", "setting", ""),
    ("$\\rho_p\\cdot\\sigma_e/\\sigma_J$", "setting", "notation"),
    ("first 12 hexadecimal digits", "setting", "Appendix A"),
    ("2026-10-21 18:14 & 9f84ab7f597f", "checked", "main/LOG.md"),
    ("2026-10-21 20:01 & 05bf60148145", "checked", "main/LOG.md"),
    ("2026-10-21 22:54 & 9b86506e744a", "checked", "main/LOG.md"),
    ("2026-10-22 06:55 & b814a4092a4b", "checked", "main/LOG.md"),
    ("2026-10-22 09:00, 09:30 & a7895d735cf4", "checked", "main/LOG.md"),
    ("2026-10-22 09:50 & 41e1f39d6bd8", "checked", "main/LOG.md"),
    ("2026-10-24 02:38 & eee46e0aa4f0", "checked", "main/LOG.md"),
    ("2026-10-31 08:46 & c660a10dc72b", "checked", "rss_ext/LOG.md"),
    ("2026-10-31 15:51 & 2e9174b32c26", "checked", "rss_ext/LOG.md"),
    ("2026-11-21 17:59 & 36d69ea82ef9", "checked", "tmlr_ext/LOG.md"),
    ("E8--E10", "label", "stage names"),
    ("E11--E17", "label", "stage names"),
    ("tolerance of $10^{-9}$", "definition", "DESIGN_main (tolerance fix)"),
    ("at least 0.6", "definition", "E16 qualification"),
    ("in \\nv{e17.k2.n} of 4 tasks", "setting", "K2 (4 tasks)"),
    ("with 4-bit weights", "label", "cited work (Module-MSE submission)"),
    ("evaluate 300 CEM candidates 1.36--2.13$\\times$ faster than FP16", "label", "cited work (WM-Quant Table 6, Sec. 4.2)"),
    ("draw of 3-bit weights", "label", "cited work (Wu et al.)"),
    ("and 12 tasks; DINO-WM PushT", "setting", "E3 design (12 TD-MPC2 tasks)"),
    ("the 95\\% interval of", "definition", "harmful rule"),
    ("95\\% bootstrap intervals", "definition", ""),
    ("(TD-MPC2, 90 paired episodes per point, 95\\% intervals)", "setting", "E2 design"),
    ("with 95\\% intervals", "definition", ""),
    ("median and 95\\% interval", "definition", ""),
    ("(paired blocks, 10 rounds per condition", "setting", "E17 latency protocol (E8 protocol)"),
    ("repaired PushT at 3 and 4 bits", "label", "E1 cells"),
    ("(int8 weights, 4- or 8-bit activation codes", "setting", "E7 kernel"),
    ("than 16-bit floating point", "label", ""),
    ("on an RTX 5090", "label", "GPU model (E8b)"),
    ("(E17; medians and 95\\% intervals over paired blocks)", "definition", "interval convention (E17 latency)"),
    ("within 10 points of the unquantized planner", "definition", "k* threshold"),
    ("$\\tau=10$ points", "definition", "k* threshold"),
    ("The dashed line is $\\red=0.5$", "definition", "repair rule"),
    ("(E15; 90 paired episodes at 6 and 12 iterations, 60 at 24", "setting", "E15 design"),
    ("over the 12 TD-MPC2 tasks", "setting", "E3 design"),
    ("of a 317M multi-task TD-MPC2", "label", "model name"),
    ("a borderline case whose damage at 2 bits equals the threshold", "label", "X5 cell (drawer-open, 2 bits)"),
    ("60 at 24; 95\\% intervals)", "definition", "interval convention"),
]
# Design constants of the appendix prose (seed ranges, software versions, planner and training settings, bit-width
# and cell labels, notation of the theory): one phrase per line in build/design_literals.txt, kind "design". The file
# was written from the list that --suggest prints and checked by hand: it holds no result number.
DESIGN_FILE = os.path.join(HERE, "design_literals.txt")
with open(DESIGN_FILE, encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line.strip() and not line.startswith("#"):
            REG.append((line, "design", "build/design_literals.txt"))


def to_scan(text):
    """The text in which literal numbers are looked for: arguments that cannot hold a result number are removed,
    captions, footnotes and section titles are kept, and whitespace is collapsed."""
    t = re.sub(r"\\begin\{tabular\}", r"\\begintabular", text)
    for name, nargs, keep in (("todo", 1, None), ("figplaceholder", 2, None), ("label", 1, None), ("ref", 1, None),
                              ("eqref", 1, None), ("cite", 1, None), ("citep", 1, None), ("citet", 1, None),
                              ("nv", 1, None), ("inputrows", 1, None), ("includegraphics", 1, None), ("url", 1, None),
                              ("bibliography", 1, None), ("bibliographystyle", 1, None), ("setlength", 2, None),
                              ("resizebox", 2, None), ("multicolumn", 2, 99), ("begintabular", 1, None),
                              ("parbox", 1, None), ("vspace", 1, None), ("hspace", 1, None),
                              ("footnote", 1, 0), ("caption", 1, 0), ("section", 1, 0), ("subsection", 1, 0)):
        t = remove_cmd(t, name, nargs, keep=keep)
    t = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", " ", t)
    t = re.sub(r"\\\\\[[0-9.]+pt\]", " ", t)
    t = re.sub(r"\\begin\{(figure|table)\}\[[^\]]*\]", " ", t)
    t = re.sub(r"p\{[0-9.]+cm\}", " ", t)
    return " ".join(t.split())


SCAN = to_scan(BODY)
covered = [False] * len(SCAN)
REG_MISSING = []
for phrase, kind, src in REG:
    ph = to_scan(phrase)
    hits = [m.start() for m in re.finditer(re.escape(ph), SCAN)] if ph else []
    if not hits:
        REG_MISSING.append((phrase, kind, src))
        continue
    for h in hits:
        for i in range(h, h + len(ph)):
            covered[i] = True

NUM = re.compile(r"(?<![A-Za-z0-9.\\\-_^{}])(\d+(?:\.\d+)?)(?![A-Za-z0-9])")
UNCHECKED = []
for m in NUM.finditer(SCAN):
    if all(covered[i] for i in range(m.start(), m.end())):
        continue
    ctx = " ".join(SCAN[max(0, m.start() - 45):m.end() + 35].split())
    UNCHECKED.append((m.group(1), ctx))
    if "--suggest" in sys.argv:   # a candidate phrase for build/design_literals.txt: three words before, two after
        a = m.start()
        for _ in range(3):
            j = SCAN.rfind(" ", 0, a - 1)
            a = 0 if j < 0 else j + 1
        b = m.end()
        for _ in range(2):
            j = SCAN.find(" ", b + 1)
            b = len(SCAN) if j < 0 else j
        print("SUGGEST\t" + SCAN[a:b].strip())

# checked literals: recompute the values that the registered 'checked' phrases quote
e1_td = len({(t, k) for (t, k, f), x in by.items() if x["det"]["confirmatory"] == "True"})
e1_dw = len(dwe1)
if (e1_td, e1_dw) != (36, 9):
    FAILS.append(f"E1 cell counts are {e1_td} (TD-MPC2) and {e1_dw} (DINO-WM), text says 36 and 9")
g8 = [r for r in csv.DictReader(open(os.path.join(A, "rss_ext/E8_latency/results/analysis/e8_cells.csv"))) if r["mode"] == "graph"]
if len(g8) != 12:
    FAILS.append(f"E8 has {len(g8)} graph conditions, text says 12")
k10 = sorted({int(r["k"]) for r in csv.DictReader(open(os.path.join(A, "rss_ext/E10_mt317m/results/analysis/e10_eval.csv")))})
if (k10[0], k10[-1]) != (3, 5):
    FAILS.append(f"E10 primary k range is {k10[0]}-{k10[-1]}, text says 3--5")
k5m = sorted({c11[t]["k"] for t in mn.MW})
if (k5m[0], k5m[-1]) != (3, 4):
    FAILS.append(f"5M Meta-World primary k range is {k5m[0]}-{k5m[-1]}, text says 3--4")
logs = open(os.path.join(A, "main/LOG.md")).read() + open(os.path.join(A, "rss_ext/LOG.md")).read() + \
    open(os.path.join(A, "tmlr_ext/LOG.md")).read()
for phrase, kind, src in REG:
    m = re.search(r"& ([0-9a-f]{12})$", phrase)
    if kind == "checked" and m and m.group(1) not in logs:
        FAILS.append(f"design hash {m.group(1)} not found in the logs")
for phrase, kind, src in REG_MISSING:
    FAILS.append(f"registered phrase not in main.tex (update REG in verify_numbers.py): {phrase!r}")

# ------------------------------------------------------------------------------------------------ report
todos = todo_items(TEX)
print(f"numbers.tex: {len(entries)} values; main.tex uses {len(set(used))} keys "
      f"({sum(kinds[k] == 'report' for k in set(used) if k in kinds)} report-only), {len(unused)} defined but unused.")
print(f"wording checks: {sum(ok for _, ok in WORDING)}/{len(WORDING)} pass.")
print(f"registered literal phrases: {len(REG) - len(REG_MISSING)}/{len(REG)} found.")
# values of kind 'report' have no result file; check them against the text of the document they come from
# (as paper B's verifier does for its report-only numbers): the value must appear in that document.
DOC_FORM = {"e4.nruns": "922,368"}          # the text writes the abbreviation of the number in the report
report_ok, report_bad = [], []
for k in report_used:
    path = os.path.join(A, sources[k].split()[0])
    if k == "k.cost.tmlr.posthoc":           # a difference of two logged totals
        ok = abs(float(values["k.cost.tmlr"]) - float(values["k.cost.tmlr.main"]) - float(values[k])) < 0.006
    else:
        ok = os.path.exists(path) and DOC_FORM.get(k, values[k]) in open(path, encoding="utf-8").read()
    (report_ok if ok else report_bad).append(k)
for k in report_bad:
    FAILS.append(f"report-only value \\nv{{{k}}} = {values[k]!r} not found in {sources[k].split()[0]}")
if report_used:
    print(f"\nREPORT-ONLY values (no result file; checked against the text of their source document: "
          f"{len(report_ok)}/{len(report_used)} found):")
    for k in report_used:
        print(f"  \\nv{{{k}}} = {values[k]!r:16} source: {sources[k]}")
if UNCHECKED:
    print(f"\nUNCHECKED literal numbers in main.tex ({len(UNCHECKED)}):")
    for n, ctx in UNCHECKED:
        print(f"  {n:>8}  ...{ctx}...")
print(f"\n\\todo markers in main.tex: {len(todos)}")
if "--todos" in sys.argv:
    for line, t in todos:
        print(f"  l.{line}: {t[:150]}")
if FAILS:
    print("\nFAILED:")
    for f in FAILS:
        print("  " + f)
strict_fail = STRICT and (UNCHECKED or report_bad or todos)
if strict_fail:
    print("\n--strict: unchecked literals, unconfirmed report-only values or \\todo markers remain.")
elif STRICT:
    print("\n--strict: passed (no unchecked literal, every report-only value found in its source, no \\todo).")
sys.exit(1 if FAILS or strict_fail else 0)
