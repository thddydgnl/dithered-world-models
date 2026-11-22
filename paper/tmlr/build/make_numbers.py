"""Write numbers.tex: every result number that main.tex quotes, recomputed from the A result files.

main.tex never types a result number; it writes \\nv{key}, and numbers.tex defines each key.
Each entry records where the value comes from and how it was obtained:
  * "recomputed": computed here from a result file (CSV/JSON written by the experiment analyses);
  * "report":     a value that exists only in a stage report (no machine-readable result file here);
                  verify_numbers.py lists these as not yet verified against data.
Rounding is half away from zero on the decimal value (Decimal), the way the reports print numbers.
Negative numbers are written as \\ensuremath{-}x and intervals as "x~[lo, hi]", so a value typesets correctly
in text and inside math mode.

Usage:  python3 build/make_numbers.py          (writes ../numbers.tex)
        import make_numbers; make_numbers.compute()  -> list of (key, tex, source, kind)
"""
import csv
import json
import os
import re
import statistics
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
TMLR = os.path.normpath(os.path.join(HERE, ".."))
A = os.path.normpath(os.path.join(TMLR, "..", ".."))
RES = os.path.join(A, "main", "results")
EXT = os.path.join(A, "rss_ext")
TX = os.path.join(A, "tmlr_ext")
OUT = os.path.join(TMLR, "numbers.tex")

MW = ["mw-push", "mw-pick-place", "mw-door-open", "mw-drawer-open", "mw-button-press-topdown",
      "mw-peg-insert-side", "mw-window-open", "mw-sweep-into"]
DMC = ["cheetah-run", "walker-walk", "quadruped-walk", "finger-turn-hard"]
SHORT = {"mw-push": "push", "mw-pick-place": "pickplace", "mw-door-open": "dooropen",
         "mw-drawer-open": "draweropen", "mw-button-press-topdown": "buttonpress",
         "mw-peg-insert-side": "peginsert", "mw-window-open": "windowopen", "mw-sweep-into": "sweepinto",
         "cheetah-run": "cheetah", "walker-walk": "walker", "quadruped-walk": "quadruped",
         "finger-turn-hard": "finger"}
TOL = 1e-9


# ---------------------------------------------------------------------------------------------- formatting
def _q(x, nd):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def num(x, nd=0, sign=False, scale=1.0, keep_negzero=False):
    """Number as LaTeX text: '$-$0.24', '+13.3', '42'."""
    v = _q(float(x) * scale, nd)
    neg = v < 0 or (keep_negzero and float(x) < 0)
    s = f"{abs(v):.{nd}f}"
    if neg:
        return "\\ensuremath{-}" + s   # a true minus sign in text and in math mode
    return ("+" if sign else "") + s


def pci(x, lo, hi, nd=0, scale=1.0, sign=True):
    """Point estimate with its 95% interval: '+42 [31, 53]'."""
    # '~' keeps the estimate with its interval and is also a space inside math mode
    return f"{num(x, nd, sign, scale)}~[{num(lo, nd, False, scale)}, {num(hi, nd, False, scale)}]"


def rng(lo, hi, nd=0, scale=1.0, keep_negzero=False):
    """Range as LaTeX text: '0.33--0.52'; with a negative end 'from -1.1 to 2.2' reads better as '-1.1 to 2.2'
    (an en dash next to a minus sign is easy to misread), and a value that rounds to zero is printed as 0."""
    a, b = num(lo, nd, False, scale, keep_negzero), num(hi, nd, False, scale, keep_negzero)
    if a.startswith("\\ensuremath{-}") or b.startswith("\\ensuremath{-}"):
        return f"{a} to {b}"
    return f"{a}--{b}"


def ci_str(s):
    lo, hi = s.strip("[]").replace(";", ",").split(",")
    return float(lo), float(hi)


def words_list(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


ENTRIES = []


def put(key, tex, source, kind="recomputed"):
    assert re.fullmatch(r"[A-Za-z0-9.\-]+", key), key
    assert all(e[0] != key for e in ENTRIES), f"duplicate key {key}"
    ENTRIES.append((key, str(tex), source, kind))


def load(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------- E1 .. E10
def earlier_results():
    src = "main/results/E1_tdmpc2/e1_tdmpc2_cells.csv"
    by = {}
    for r in csv.DictReader(open(os.path.join(A, src))):
        by.setdefault((r["task"], int(r["k"]), r["family"]), {})[r["variant"]] = r
    aff = {k: v for k, v in by.items() if v["det"]["confirmatory"] == "True" and v["det"]["holm_real"] == "True"}
    mw = {k: v for k, v in aff.items() if k[2] == "mw"}
    dmc = {k: v for k, v in aff.items() if k[2] == "dmc"}
    red_mw = [float(v["shiter"]["red"]) for v in mw.values()]
    red_dmc = [float(v["shiter"]["red"]) for v in dmc.values()]
    put("e1.mw.naff", len(mw), src)
    put("e1.mw.nfix", sum(r >= 0.5 - TOL for r in red_mw), src)
    put("e1.mw.redmed", num(100 * statistics.median(red_mw)), src)
    wk = min(mw.items(), key=lambda kv: float(kv[1]["shiter"]["red"]))
    put("e1.mw.weakest.keep", num(100 * (1 - float(wk[1]["shiter"]["red"]))), src + f" ({wk[0][0]} k={wk[0][1]})")
    put("e1.mw.weakest.k", wk[0][1], src)
    put("e1.dmc.naff", len(dmc), src)
    put("e1.dmc.nfix", sum(r >= 0.5 - TOL for r in red_dmc), src)
    conf = {k: v for k, v in by.items() if v["det"]["confirmatory"] == "True"}
    dmax = max(conf.items(), key=lambda kv: float(kv[1]["det"]["D"]))
    put("e1.maxdet", num(100 * float(dmax[1]["det"]["D"])), src + f" ({dmax[0][0]} k={dmax[0][1]})")
    put("e1.maxdet.k", dmax[0][1], src)
    pp = by[("mw-pick-place", 3, "mw")]
    put("e1.pickplace.err.det", num(pp["det"]["pt_err1h"], 2), src)
    put("e1.pickplace.err.shiter", num(pp["shiter"]["pt_err1h"], 2), src)
    put("e1.pickplace.D.det", num(pp["det"]["D"], 1, scale=100), src)
    put("e1.pickplace.D.shiter", num(pp["shiter"]["D"], 1, scale=100), src)
    wa = by[("walker-walk", 3, "dmc")]
    put("e1.walker.D.shiter", num(wa["shiter"]["D"], 1, scale=100), src)
    put("e1.walker.D.det", num(wa["det"]["D"], 1, scale=100), src)
    ch4 = by[("cheetah-run", 4, "dmc")]
    put("e1.cheetah4.D.det", num(ch4["det"]["D"], 1, scale=100), src)
    means = {v: 100 * statistics.mean(float(x[v]["D"]) for x in mw.values())
             for v in ("det", "ind", "shiter", "shstep", "shep", "detshift")}
    for v, m in means.items():
        put(f"e1.mw.mean.{v}", num(m, 1), src)
    h1_td = sum(float(v["det"]["pt_err1h"]) <= float(v["shiter"]["pt_err1h"]) for v in aff.values())
    # PLAN 4.6: on held states (same states for all variants) -- 1-step and 3-step latent errors
    h1_held = sum(float(v["det"]["held_pt_err1h"]) <= float(v["shiter"]["held_pt_err1h"]) for v in aff.values())
    h3_held = sum(float(v["det"]["held_pt_err3"]) <= float(v["shiter"]["held_pt_err3"]) for v in aff.values())
    put("e1.td.naff", len(aff), src)
    put("e1.td.held.err1", h1_held, src + " (held_pt_err1h)")
    put("e1.td.held.err3", h3_held, src + " (held_pt_err3)")

    src = "main/results/E1/e1_dinowm_cells.csv"
    dw = {}
    base = {}
    expl = {}
    for r in csv.DictReader(open(os.path.join(A, src))):
        if r["role"] == "confirmatory":
            dw.setdefault((r["task"], int(r["k"])), {})[r["var"]] = r
        elif r["role"] == "baseline":
            base[r["task"]] = r
        elif r["role"] == "exploratory":
            expl.setdefault((r["task"], int(r["k"])), {})[r["var"]] = r
    dw_aff = {k: v for k, v in dw.items() if v["det"]["drop_real_holm"] == "True"}
    put("e1.dw.naff", len(dw_aff), src)
    put("e1.dw.nfix", sum(float(v["shiter"]["red"]) >= 0.5 - TOL for v in dw_aff.values()), src)
    h1_dw = sum(float(v["det"]["rel_err1"]) <= float(v["shiter"]["rel_err1"]) for v in dw_aff.values())
    put("e1.h1.n", h1_td + h1_dw, "E1 cells (pt_err1h, rel_err1)")
    put("e1.h1.of", len(aff) + len(dw_aff), "E1 cells")
    # review M1: of these cells, those in which sh-iter removes at least half of the damage
    fix_td = sum(float(v["det"]["pt_err1h"]) <= float(v["shiter"]["pt_err1h"]) and float(v["shiter"]["red"]) >= 0.5 - TOL
                 for v in aff.values())
    fix_dw = sum(float(v["det"]["rel_err1"]) <= float(v["shiter"]["rel_err1"]) and float(v["shiter"]["red"]) >= 0.5 - TOL
                 for v in dw_aff.values())
    assert (fix_td, fix_dw) == (17, 4), (fix_td, fix_dw)   # review recount: TD-MPC2 17, DINO-WM 4
    put("e1.h1.fix", fix_td + fix_dw, "E1 cells (det error <= sh-iter's and red_shiter >= 0.5)")
    put("e1.pusht5.D.det", num(dw[("pusht", 5)]["det"]["D"], 0, scale=100), src)
    p4 = dw[("pusht", 4)]
    put("e1.pusht4.err5.det", num(p4["det"]["rel_err5"], 2), src)
    put("e1.pusht4.err5.shiter", num(p4["shiter"]["rel_err5"], 2), src)
    put("e1.pusht4.D.det", pci(float(p4["det"]["D"]), *ci_str(p4["det"]["D_ci"]), 0, 100), src)
    put("e1.pusht3.red.shplan", num(dw[("pusht", 3)]["shstep"]["red"], 2), src + " (sh-plan = shstep)")
    put("e1.pusht4.red.shplan", num(p4["shstep"]["red"], 2), src + " (sh-plan = shstep)")
    pm3 = dw[("pointmaze", 3)]
    put("e1.pm3.D.det", pci(float(pm3["det"]["D"]), *ci_str(pm3["det"]["D_ci"]), 0, 100), src)
    w3 = expl[("wall", 3)]
    put("e1.wall3.D.det", pci(float(w3["det"]["D"]), *ci_str(w3["det"]["D_ci"]), 0, 100), src + " (exploratory cell)")
    put("e1.wall.none", num(base["wall"]["success"], 0, scale=100), src)
    put("e1.wall3.det.success", num(w3["det"]["success"], 0, scale=100), src)
    w4 = dw[("wall", 4)]["det"]
    lo4, hi4 = ci_str(w4["D_ci"])
    assert float(w4["D"]) < 0 and hi4 < 0, w4["D_ci"]
    put("e1.wall4.det.above", pci(-float(w4["D"]), -hi4, -lo4, 0, 100, sign=False), src + " (det at 4 bits above none)")
    put("e1.pusht.none", num(base["pusht"]["success"], 0, scale=100), src)
    reds = [float(w3[v]["red"]) for v in ("ind", "shiter", "shstep", "detshift")]
    put("e1.wall3.red.range", rng(min(reds), max(reds), 2), src + " (four dither variants)")

    src = "rss_ext/E10_mt317m/results/analysis/e10_eval.csv"
    e10 = list(csv.DictReader(open(os.path.join(A, src))))
    e10a = [r for r in e10 if r["drop_real"] == "True"]
    ratios = [float(r["pt_err1h_det"]) / float(r["pt_err1h_sh"]) for r in e10a]
    put("e10.naff", len(e10a), src)
    put("e10.nfix", sum(float(r["red_point"]) >= 0.5 - TOL for r in e10a), src)
    put("e10.errratio", rng(min(ratios), max(ratios), 2), src)
    put("e10.ddet", rng(min(float(r["D_det"]) for r in e10a), max(float(r["D_det"]) for r in e10a), 0, 100), src)
    put("e10.red", rng(min(float(r["red_point"]) for r in e10a), max(float(r["red_point"]) for r in e10a), 0, 100), src)
    put("e10.h1.n", sum(x <= 1 for x in ratios), src)
    peg = [r for r in e10 if r["task"] == "mw-peg-insert-side"][0]
    put("e10.peg.D.det", num(peg["D_det"], 1, scale=100), src)
    put("e10.peg.D.shiter", num(peg["D_sh"], 1, scale=100), src)

    src = "main/results/E2_tdmpc2/e2_tdmpc2_summary.json + main/results/E1/e1_dinowm_v52.json"
    lb = load(os.path.join(RES, "E2_tdmpc2", "e2_tdmpc2_summary.json"))["layer_b"]
    tasks = ("mw-push", "mw-pick-place", "cheetah-run")
    pt = load(os.path.join(RES, "E1", "e1_dinowm_v52.json"))["b2_H2a_prime_pusht_k4"]
    r_shiter = [lb[t]["b2_H2a_prime"]["r_shiter"]["r"] for t in tasks] + [pt["r_shiter"]]
    put("e2.drift.shiter.max", num(100 * max(r_shiter), 0), src)
    put("e2.drift.shiter.list", words_list(num(x, 2) for x in r_shiter), src)
    r_ds = [lb[t]["b2_H2a_prime"]["r_detshift"]["r"] for t in tasks] + [pt["r_detshift"]]
    put("e2.drift.detshift.max", num(100 * max(r_ds), 0), src)
    r_pc = [lb[t]["b2_H2a_prime"][v]["r"] for t in tasks for v in ("r_shstep", "r_shep")]
    put("e2.drift.posctrl", rng(100 * min(r_pc), 100 * max(r_pc), 0), src)
    put("e2.drift.pusht.shplan", num(pt["positive_control_r_shplan"], 2), src)
    src = "main/results/E2_tdmpc2/e2_tdmpc2_cells.csv"
    e2 = {(r["task"], r["setting"], r["variant"]): r for r in csv.DictReader(open(os.path.join(A, src)))}
    d12 = {v: [100 * float(e2[(t, "it12", v)]["D"]) for t in tasks] for v in ("det", "shstep", "shiter")}
    for v, xs in d12.items():
        put(f"e2.it12.{v}", rng(min(xs), max(xs), 0), src)
    d1 = {v: [float(e2[(t, "it1", v)]["D"]) for t in tasks] for v in ("det", "shiter")}
    # "with one iteration sh-iter is no better than det": sh-iter damage >= det damage - 5 points in every task
    assert all(s >= d - 0.05 for s, d in zip(d1["shiter"], d1["det"])), d1
    put("e2.it1.check", "", src + " (assert: sh-iter no better than det at 1 iteration)")

    src = "main/results/E3_tdmpc2/e3_tdmpc2_final.json"
    dx = load(os.path.join(A, src))["DMC_explanation"]
    s_mw = [r["S_det"] for r in dx["rows"] if r["task"].startswith("mw-")]
    s_dmc = [r["S_det"] for r in dx["rows"] if not r["task"].startswith("mw-")]
    put("e3.share.dmc", rng(min(s_dmc), max(s_dmc), 2), src)
    put("e3.share.mw", rng(min(s_mw), max(s_mw), 2), src)
    put("e3.spearman", num(dx["spearman_S_red"], 2), src)

    src = "rss_ext/E8_latency/results/analysis/e8_cells.csv"
    g = [r for r in csv.DictReader(open(os.path.join(A, src))) if r["mode"] == "graph"]
    assert len(g) == 12  # "in all 12 conditions of each of two runs" (main text, Section 7)
    sp = [float(r["T_fp32_ms"]) / float(r["T_int_det_ms"]) for r in g]
    put("e8.speedup", rng(min(sp), max(sp), 1), src)
    ov = [100 * float(r["overhead_median"]) for r in g]
    put("e8.overhead", f"{num(min(ov), 1, sign=True)} to {num(max(ov), 1, sign=True)}", src)
    src = "rss_ext/E8c_latency/results/analysis/e8c_cells.csv"
    c8 = list(csv.DictReader(open(os.path.join(A, src))))
    gw = [r for r in c8 if r["mode"] == "graph"]
    gs = [r for r in c8 if r["mode"] == "gstep" and r["rest"] == "fp32"]
    assert len(gw) == 12 and len(gs) == 12
    sp_c = [float(r["T_fp32_ms"]) / float(r["T_int-det_ms"]) for r in gw]
    ov_c = [100 * float(r["overhead_median"]) for r in gw]
    put("e8c.speedup", rng(min(sp_c), max(sp_c), 1), src)
    put("e8.overhead.max", num(max(ov + ov_c), 1), "e8_cells.csv + e8c_cells.csv")
    n_bc = sum(r["minfloat_is"] == "bf16cast" for r in gw)
    put("e8c.bf16fastest", n_bc, src)
    r_bc = [float(r["int_det_over_bf16cast"]) for r in gw]
    put("e8c.int.over.bf16", rng(min(r_bc), max(r_bc), 2), src)
    l0 = [float(r["T_int-det-L0tf32_ms"]) / float(r["T_bf16cast_ms"]) for r in gw]
    put("e8c.tf32first.over.bf16", rng(min(l0), max(l0), 2), src + " [post hoc]")
    t_graph = {(r["size"], r["N"]): float(r["T_fp32_ms"]) for r in gw}
    share = [t_graph[(r["size"], r["N"])] / float(r["T_fp32_ms"]) for r in gs]
    put("e8c.wmshare", rng(100 * min(share), 100 * max(share), 0), src)
    st = [float(r["T_fp32_ms"]) / float(r["T_int-det_ms"]) for r in gs]
    put("e8c.step.speedup", rng(min(st), max(st), 2), src)
    assert all(r["minfloat_is"] == "bf16cast" for r in gs)
    st_bc = [float(r["int_det_over_minfloat"]) for r in gs]
    put("e8c.step.over.bf16", rng(min(st_bc), max(st_bc), 2), src)
    om = [100 * float(r["overhead_median"]) for r in gs]
    put("e8c.step.overhead.max", num(max(om), 1), src)

    src = "rss_ext/E9_fp4/results/e9_cells.csv"
    e9 = {(r["task"], r["fmt"]): r for r in csv.DictReader(open(os.path.join(A, src)))}
    for fmt in ("nvfp4", "mxfp4"):
        dam = [r for (t, f), r in e9.items() if f == fmt and r["damaged"] == "True"]
        put(f"e9.{fmt}.ndam", len(dam), src)
    put("e9.nvfp4.cheetah", num(e9[("cheetah-run", "nvfp4")]["D_det"], 1, scale=100), src)
    put("e9.nvfp4.walker", num(e9[("walker-walk", "nvfp4")]["D_det"], 1, scale=100), src)
    put("e9.mxfp4.cheetah", num(e9[("cheetah-run", "mxfp4")]["D_det"], 1, scale=100), src)
    put("e9.mxfp4.windowopen", num(e9[("mw-window-open", "mxfp4")]["D_det"], 1, scale=100), src)
    dam = [r for r in e9.values() if r["damaged"] == "True"]
    rat = [float(r["pt_err1h_det"]) / float(r["pt_err1h_shiter"]) for r in dam]
    put("e9.errratio", rng(min(rat), max(rat), 2), src + " (damaged cells)")
    put("e9.ndam", len(dam), src)

    # values that the RSS draft took from stage reports, recomputed here from the stage result files (the asserted
    # strings are the report values, so the text does not change)
    put("e4.nruns", "922{,}368", "main/reports/E4_theory.md 10 (922,368 planner runs; not stored in a result file)", "report")
    c4 = load(os.path.join(RES, "E4", "e4_comparisons.json"))
    sl = [c4["C2"][k]["slope_through_origin"] for k in ("cem_elite|all", "cem_argmin_ntc|all", "mppi_same_draw|all")] + \
         [c4["C4"][p]["drift_vs_pred"]["slope_through_origin"] for p in ("cem|all", "mppi|all")]
    put("e4.slopes", rng(min(sl), max(sl), 2), "main/results/E4/e4_comparisons.json C2 (selection) + C4 (SAA drift) slopes")
    assert rng(min(sl), max(sl), 2) == "0.92--1.05"
    f7 = load(os.path.join(RES, "E7_tdmpc2", "family_points_e7_tdmpc2.json"))
    s7 = load(os.path.join(RES, "E7_tdmpc2", "e7_tdmpc2_summary.json"))["latency_rel_fp32"]
    slow = [statistics.median(v for k, v in s7.items() if f"|{kern}_det|" in k) for kern in ("int8", "int4", "int4p")]
    put("e7.eager.slower", num(100 * max(slow), 0), "main/results/E7_tdmpc2/e7_tdmpc2_summary.json latency_rel_fp32 "
        "(median over conditions, largest of the three kernels)")
    ov = [v for kern in ("int8", "int4", "int4p") for v in f7["overhead"][kern]["per_condition"].values()]
    put("e7.overhead.max", num(100 * max(ov), 1), "main/results/E7_tdmpc2/family_points_e7_tdmpc2.json overhead")
    f5 = load(os.path.join(RES, "E5_tdmpc2", "family_points_e5_tdmpc2.json"))
    f5d = load(os.path.join(RES, "E5_dinowm", "family_points_e5_dinowm.json"))
    put("e5.h3b.td", f5["H3b"]["passing"], "main/results/E5_tdmpc2/family_points_e5_tdmpc2.json H3b")
    put("e5.h3b.td.of", f5["H3b"]["cells"], "family_points_e5_tdmpc2.json H3b")
    put("e5.h3b.dw", f5d["H3b"]["passing"], "main/results/E5_dinowm/family_points_e5_dinowm.json H3b")
    put("e5.h3b.dw.of", f5d["H3b"]["cells"], "family_points_e5_dinowm.json H3b")
    bmw = [x["D_Bdet"] for x in f5["H3b_cells"].values() if x["family"] == "mw"]
    put("e5.pc.mw.range", rng(100 * min(bmw), 100 * max(bmw), 1), "family_points_e5_tdmpc2.json H3b_cells (Meta-World D(B+det))")
    e3d = load(os.path.join(RES, "E3_dinowm", "family_points_e3_dinowm.json"))
    w3 = [x for x in e3d["tasks"]["wall_k3"]["levels"] if x["level"] == "L3"][0]
    put("e3.wall.singleoffset", num(100 * w3["D"], 0, sign=True), "family_points_e3_dinowm.json tasks.wall_k3 (level L3)")
    b8 = load(os.path.join(EXT, "E8_latency", "results", "e8b", "e8b.json"))["gemm"]
    sp = [b8[f"gemm|L1|{n}|bf16"]["T_ms"] / b8[f"gemm|L1|{n}|nvfp4"]["T_ms"] for n in (512, 2048, 8192)]
    put("e8b.nvfp4.gemm", num(max(sp), 1), "rss_ext/E8_latency/results/e8b/e8b.json gemm (BF16 / NVFP4, 4096-wide layer)")
    e3s = load(os.path.join(RES, "E3_tdmpc2", "e3_tdmpc2_summary.json"))
    rho = [e3s["sweep"][t]["spearman"] for t in e3s["H4a"]["eligible_tasks"]] + \
          [x["rho_Neff_D"] for x in e3d["tasks"].values() if x.get("eligible") and x.get("role") == "primary"]
    ok = [r for r in rho if r <= -0.8 + TOL]
    put("e3.h4a.n", len(ok), "e3_tdmpc2_summary.json sweep + family_points_e3_dinowm.json (qualifying tasks passing)")
    put("e3.h4a.of", len(rho), "e3_tdmpc2_summary.json H4a + family_points_e3_dinowm.json")
    put("e3.h4a.exact", sum(abs(r + 0.8) < 1e-6 for r in ok), "E3 sweeps (Spearman exactly -0.8)")
    fp2 = load(os.path.join(RES, "E2_tdmpc2", "family_points_tdmpc2.json"))
    fpd = load(os.path.join(RES, "E2_dinowm", "family_points_dinowm.json"))
    put("e2.h2c.n", fp2["tdmpc2_tasks_holding"]["H2c"] + int(fpd["H2c_pusht"]), "family_points_tdmpc2.json + family_points_dinowm.json")
    put("e3.h4b.rho", num(load(os.path.join(RES, "E3_tdmpc2", "e3_tdmpc2_final.json"))["H4b"]["spearman"], 2),
        "main/results/E3_tdmpc2/e3_tdmpc2_final.json H4b")
    w4 = [v["D"] for k, v in load(os.path.join(RES, "E5_tdmpc2", "e5_tdmpc2_summary.json"))["weights"].items() if k.endswith("|w4rtn")]
    put("e5.weights.w4", rng(min(w4), max(w4), 2), "main/results/E5_tdmpc2/e5_tdmpc2_summary.json weights (W4 RTN D, fraction)")
    put("e5.weights.w4.pts", rng(min(w4), max(w4), 0, 100), "main/results/E5_tdmpc2/e5_tdmpc2_summary.json weights (W4 RTN D, points or % of return)")
    was = {"e7.eager.slower": "18", "e7.overhead.max": "6.9", "e5.h3b.td": "10", "e5.h3b.td.of": "19", "e5.h3b.dw": "1",
           "e5.h3b.dw.of": "3", "e5.pc.mw.range": "0.0--16.7", "e3.wall.singleoffset": "+37", "e8b.nvfp4.gemm": "5.3",
           "e3.h4a.n": "5", "e3.h4a.of": "6", "e3.h4a.exact": "3", "e2.h2c.n": "3", "e3.h4b.rho": "0.03",
           "e5.weights.w4": "0.42--0.93"}
    now = {k: t for k, t, s, kind in ENTRIES}
    assert all(now[k] == v for k, v in was.items()), {k: (now[k], v) for k, v in was.items() if now[k] != v}


# ---------------------------------------------------------------------------------------------- E11 .. E17
def tmlr_results():
    src_td = "tmlr_ext/TD_tdmpc2/results/analysis/td_summary.json"
    td = load(os.path.join(A, src_td))
    c = td["E11"]["cells"]
    units = td["E11"]["units"]
    jd = td["judgements"]

    # ---- C1: E11 TD-MPC2 5M damage, new seeds
    n_dam = sum(bool(c[t]["damaged"]) for t in c)
    put("e11.td.ndam", n_dam, src_td)
    put("e11.td.ncells", len(c), src_td)
    for unit, tasks, nd in (("mw", MW, 1), ("dmc", DMC, 1)):
        dd = [c[t]["variants"]["det"]["D"] for t in tasks]
        put(f"e11.{unit}.ddet", rng(min(dd), max(dd), nd, 100), src_td)
        put(f"e11.{unit}.ndam", units[unit]["n_damaged"], src_td)
        put(f"e11.{unit}.ncells", units[unit]["n_cells"], src_td)
    for t in MW + DMC:
        v = c[t]["variants"]
        put(f"e11.{SHORT[t]}.k", c[t]["k"], src_td)
        put(f"e11.{SHORT[t]}.ddet", pci(v["det"]["D"], *v["det"]["D_ci"], 1, 100), src_td)

    # ---- C2: components
    pers = {t: c[t]["contrasts"]["D_pers=detshiftfix-detshift"] for t in c}
    put("e11.pers.npos", sum(pers[t]["ci"][0] > 0 for t in c), src_td + " (CI lower > 0)")
    for unit, tasks in (("mw", MW), ("dmc", DMC)):
        xs = [pers[t]["D"] for t in tasks]
        put(f"e11.{unit}.pers", rng(min(xs), max(xs), 1, 100), src_td)
        for var in ("detbc", "shep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"):
            put(f"e11.{unit}.nfix.{var}", units[unit]["n_fix"][var], src_td)
        for cls in ("persistence", "bias", "resolution"):
            put(f"e11.{unit}.class.{cls}", units[unit]["counts"].get(cls, 0), src_td)
    put("e11.push.pers", pci(pers["mw-push"]["D"], *pers["mw-push"]["ci"], 1, 100), src_td)
    sps = [c[t]["contrasts"]["shep-shiter"] for t in c]
    assert all(x["ci"][0] > 0 for x in sps)
    put("e11.shepminusshiter", rng(min(x["D"] for x in sps), max(x["D"] for x in sps), 1, 100), src_td)
    pp = c["mw-pick-place"]["variants"]["detbc"]
    put("e11.pickplace.red.detbc", pci(pp["red"], *pp["red_ci"], 3, 1, sign=False), src_td)
    bias_cells = [t for t in MW if c[t]["class"] == "bias"]
    assert bias_cells == ["mw-push", "mw-drawer-open", "mw-button-press-topdown"], bias_cells
    put("e11.mwbias.red.detshiftfix", words_list(num(c[t]["variants"]["detshiftfix"]["red"], 2) for t in bias_cells), src_td)
    put("e11.mwbias.red.detnoise", words_list(num(c[t]["variants"]["detnoise"]["red"], 2) for t in bias_cells), src_td)
    harm_shep = [t for t in c if c[t]["variants"]["shep"]["harmful"]]
    put("e11.shep.nharm.mw", sum(t in MW for t in harm_shep), src_td)

    # ---- C4 parts of E11: B3/B4, det+noise
    b4 = jd["B4"]
    put("e11.b4.n", b4["n_red_ge_half"], src_td)
    put("e11.b4.of", b4["n_damaged"], src_td)
    put("e11.b4.nharm", len(b4["harmful_all_cells"]), src_td)
    put("e11.windowopen.red.shiter", num(c["mw-window-open"]["variants"]["shiter"]["red"], 2), src_td)
    dn_mw = [c[t]["variants"]["detnoise"]["red"] for t in MW]
    put("e11.mw.red.detnoise", rng(min(dn_mw), max(dn_mw), 2), src_td)
    dn_fix_dmc = [t for t in DMC if c[t]["fixes"]["detnoise"]]
    assert dn_fix_dmc == ["finger-turn-hard"], dn_fix_dmc
    put("e11.walker.harm.shiter", pci(c["walker-walk"]["variants"]["shiter"]["Dv_minus_Ddet"],
                                      *c["walker-walk"]["variants"]["shiter"]["Dv_minus_Ddet_ci"], 1, 100), src_td)

    # ---- E13: persistence share and the Dg1 summary statistic
    src13 = "tmlr_ext/TD_tdmpc2/results/analysis/td_e13_cells.csv"
    rho = {}
    for r in csv.DictReader(open(os.path.join(A, src13))):
        if r["set"] == "last":
            rho.setdefault(r["cell"].split("_", 1)[1], []).append(float(r["rho_p"]))
    put("e13.rho.detnoise", rng(min(rho["detnoise"]), max(rho["detnoise"]), 2), src13 + " (last set, all rows)")
    fresh = rho["shiter"] + rho["detshift"]
    put("e13.rho.fresh", rng(min(fresh), max(fresh), 2), src13 + " (last set, sh-iter and det+shift, all rows)")
    fixed = rho["det"] + rho["detbc"] + rho["shep"] + rho["detshiftfix"] + rho["shstep"]
    assert all(abs(x - 1) < 1e-6 for x in fixed)
    put("e13.rho.fixed", "1.00", src13 + " (det, det+BC, sh-ep, sh-step, det+shift-fixed)")
    # same twelve E11 cells only (used in a \todo: the approved range pools all E13 rows)
    e13 = td["E13"]["cells"]
    same = [e13[t][f"PTk{c[t]['k']}_{v}"]["last"]["rho_p"] for t in c for v in ("shiter", "detshift")]
    put("e13.rho.fresh.e11cells", rng(min(same), max(same), 2), src_td + " E13 (E11 cells only)")
    for var in ("shiter", "detshift"):
        xs = [e13[t][f"PTk{c[t]['k']}_{var}"]["last"]["rho_p"] for t in c]
        put(f"e13.rho.e11.{var}", rng(min(xs), max(xs), 2), src_td + f" E13 (E11 cells, {var}, last set)")
    dn = [e13[t][f"PTk{c[t]['k']}_detnoise"]["last"]["rho_p"] for t in c]
    assert rng(min(dn), max(dn), 2) == rng(min(rho["detnoise"]), max(rho["detnoise"]), 2)  # det+noise exists only on E11 cells
    h1c = td["E13"]["e1_h1_common_states"]
    put("e13.h1common.n", sum(bool(v["det_le_shiter"]) for v in h1c.values()), src_td + " E13.e1_h1_common_states")
    put("e13.h1common.of", len(h1c), src_td)
    dg = jd["Dg1"]
    put("dg1.mw", num(dg["mw"]["mean_diff"], 2), src_td)
    diffs = [x["diff"] for x in dg["mw"]["per_cell"].values()]
    assert all(d > 0 for d in diffs)
    put("dg1.mw.range", rng(min(diffs), max(diffs), 2), src_td)
    put("dg1.dmc", num(dg["dmc"]["mean_diff"], 2), src_td)
    put("dg1.walker", num(dg["dmc"]["per_cell"]["walker-walk"]["diff"], 2), src_td)
    put("dg1.quadruped", num(dg["dmc"]["per_cell"]["quadruped-walk"]["diff"], 2), src_td)

    # ---- E12: bit limits
    e12 = td["E12"]["tasks"]
    ks = lambda t, cfg: e12[t]["kstar"][cfg]["kstar"]

    def g_alone(t, rg, var):
        return min(ks(t, f"{rg}|det"), ks(t, f"{rg}|detbc")) - ks(t, f"{rg}|{var}")

    for rg in ("PT", "PC"):
        g = [e12[t]["G"][rg] for t in MW]
        put(f"e12.{rg.lower()}.nG", sum(x >= 1 for x in g), src_td + f" (G = best det-family k* - best dither k*, {rg})")
        put(f"e12.{rg.lower()}.nG.shiter", sum(g_alone(t, rg, "shiter") >= 1 for t in MW), src_td + " (sh-iter alone)")
        put(f"e12.{rg.lower()}.nG.detshift", sum(g_alone(t, rg, "detshift") >= 1 for t in MW), src_td + " (det+shift alone)")
        put(f"e12.dmc.G.{rg.lower()}", words_list(str(e12[t]["G"][rg]) for t in DMC), src_td + " (cheetah, walker, quadruped, finger)")
    assert jd["Fr1"]["n_tasks_G_ge_1"] == sum(e12[t]["G"]["PT"] >= 1 for t in MW)
    best = {t: e12[t]["best"] for t in MW}
    one_bit = [t for t in MW if best[t]["kstar"] == 1]
    assert all(all(cfg.split("|")[1] in ("shiter", "detshift") for cfg in best[t]["configs"]) for t in one_bit)
    put("e12.mw.n1bit", len(one_bit), src_td + " (tasks whose lowest k* is 1; all dither configurations)")
    put("e12.mw.n", len(MW), src_td)
    # Figure 1(b): mw-push, per-tensor ranges
    put("e12.push.pt.kstar.det", ks("mw-push", "PT|det"), src_td + " E12.tasks.mw-push.kstar.PT|det")
    put("e12.push.pt.kstar.shiter", ks("mw-push", "PT|shiter"), src_td + " E12.tasks.mw-push.kstar.PT|shiter")

    # ---- E15: DMC x iterations
    it1, it2 = jd["It1"], jd["It2"]
    put("e15.it1.n", it1["n_tasks_red_best_ge_half"], src_td)
    put("e15.it2.n", it2["n_monotone"], src_td)
    rb = it1["red_best_12"]
    for t in DMC:
        put(f"e15.{SHORT[t]}.red12", num(rb[t], 2), src_td)
    w24 = it2["per_task"]["walker-walk"]["red_best_6_12_24"][2]
    put("e15.walker.red24", num(w24, 2), src_td)
    rt = td["E15"]["return_time"]["cheetah-run"]
    put("e15.cheetah.detk1.D", num(rt["det_k1+1_it6"]["D"], 1, scale=100), src_td + " E15.return_time")
    put("e15.cheetah.detk1.ms", num(rt["det_k1+1_it6"]["ms"], 1), src_td)
    put("e15.cheetah.ds12.D", num(rt["detshift_k1_it12"]["D"], 1, scale=100), src_td)
    put("e15.cheetah.ds12.ms", num(rt["detshift_k1_it12"]["ms"], 1), src_td)
    put("e15.cheetah.ds24.D", num(rt["detshift_k1_it24"]["D"], 1, scale=100), src_td)
    put("e15.cheetah.ds24.ms", num(rt["detshift_k1_it24"]["ms"], 1), src_td)
    put("e15.cheetah.k", c["cheetah-run"]["k"], src_td)

    # ---- E16 (TD-MPC2 half): CEM-like planner
    e16 = td["E16"]
    for t in ("mw-push", "mw-pick-place", "cheetah-run"):
        x = e16[t]
        assert x["qualified"]
        put(f"e16.td.cem.{SHORT[t]}", pci(x["shep_minus_shiter"]["D"], *x["shep_minus_shiter"]["ci"], 1, 100), src_td + " E16")
        put(f"e16.td.mppi.{SHORT[t]}", pci(x["default_shep_minus_shiter"]["D"], *x["default_shep_minus_shiter"]["ci"], 1, 100), src_td + " E16")
        put(f"e16.td.none.{SHORT[t]}", num(x["none_cem"] / x["none_default"], 2), src_td + " E16 (none CEM-like / default)")
    ch = e16["cheetah-run"]["variants"]
    put("e16.td.cheetah.D.shiter", num(ch["shiter"]["D"], 1, scale=100), src_td)
    put("e16.td.cheetah.D.det", num(ch["det"]["D"], 1, scale=100), src_td)
    put("e16.td.cheetah.red.shiter", num((ch["det"]["D"] - ch["shiter"]["D"]) / ch["det"]["D"], 2), src_td)

    # ---- E14: 317M
    src14 = "tmlr_ext/MT_317m/results/analysis/e14_eval.csv"
    e14 = list(csv.DictReader(open(os.path.join(A, src14))))
    j14 = load(os.path.join(TX, "MT_317m", "results", "analysis", "e14_judgement.json"))
    dam = [r for r in e14 if r["drop_real"] == "True"]
    put("e14.ndam", len(dam), src14)
    put("e14.ntasks", len(e14), src14)
    put("e14.ddet", rng(min(float(r["D_det"]) for r in dam), max(float(r["D_det"]) for r in dam), 1, 100), src14)
    put("e14.p1.n", len(j14["P1"]["tasks_red_shiter_ge_half"]), "e14_judgement.json")
    put("e14.red.shiter", rng(min(float(r["red_shiter"]) for r in dam), max(float(r["red_shiter"]) for r in dam), 2), src14)
    put("e14.p2.n", j14["P2"]["n"], "e14_judgement.json")
    p2 = [r for r in dam if r["task"] in j14["P2"]["tasks_ci_lower_gt0"]]
    put("e14.p2.range", rng(min(float(r["D_shep-D_shiter"]) for r in p2), max(float(r["D_shep-D_shiter"]) for r in p2), 1, 100), src14)
    push = [r for r in e14 if r["task"] == "mw-push"][0]
    put("e14.push.p2", pci(float(push["D_shep-D_shiter"]), *ci_str(push["D_shep-D_shiter_ci"]), 1, 100), src14)
    put("e14.nfix.detbc", len(j14["descriptive"]["fixes_red_ge_half_among_damaged"]["detbc"]), "e14_judgement.json")
    put("e14.nfix.detshift", len(j14["descriptive"]["fixes_red_ge_half_among_damaged"]["detshift"]), "e14_judgement.json")
    sw = [r for r in e14 if r["task"] == "mw-sweep-into"][0]
    put("e14.sweep.r", num(sw["r_shep_over_det"], 2), src14)
    assert j14["descriptive"]["harmful_eval_tasks"]["shep"] == ["mw-sweep-into"]

    # ---- E11/E12/E13/E16 DINO-WM
    src_dw = "tmlr_ext/DW_dinowm/results/analysis/analysis_dw.json"
    dwj = load(os.path.join(A, src_dw))
    cells = {(x["task"], x["k"]): x for x in dwj["E11"]["cells"]}
    pv = dict(zip([(x["task"], x["k"]) for x in dwj["E11"]["cells"]], dwj["E11"]["holm"]["pvals"]))
    put("dw.ndam", len(dwj["E11"]["damaged"]), src_dw)
    put("dw.ncells", len(cells), src_dw)
    for (task, k), name in ((("pusht", 4), "pusht4"), (("pointmaze", 3), "pm3"), (("wall", 3), "wall3")):
        x = cells[(task, k)]
        d = x["variants"]["det"]
        put(f"dw.{name}.D.det", pci(d["D"], *d["D_ci"], 0, 100), src_dw)
        put(f"dw.{name}.none", num(x["none"]["success"], 0, scale=100), src_dw)
        put(f"dw.{name}.det.success", num(d["success"], 0, scale=100), src_dw)
        put(f"dw.{name}.p", num(pv[(task, k)], 3 if pv[(task, k)] > 0.01 else 4), src_dw)
    cl = cells[("pusht", 4)]["cluster"]["D"]["det"]
    put("dw.pusht4.D.det.cluster", f"[{num(cl['ci'][0], 0, scale=100)}, {num(cl['ci'][1], 0, scale=100)}]", src_dw + " (cluster bootstrap, D1)")
    put("dw.pusht4.nclusters", cl["n_clusters"], src_dw)
    w = cells[("wall", 3)]
    wv = w["variants"]
    for v in ("detbc", "shstep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"):
        put(f"dw.wall3.red.{v}", num(wv[v]["red"], 2), src_dw)
    put("dw.wall3.red.detbc.ci", pci(wv["detbc"]["red"], *wv["detbc"]["red_ci"], 2, sign=False), src_dw)
    put("dw.wall3.dbias", pci(w["d_bias"]["diff"], *w["d_bias"]["ci"], 0, 100), src_dw)
    put("dw.wall3.dpers", pci(w["d_pers"]["diff"], *w["d_pers"]["ci"], 0, 100), src_dw)
    put("dw.wall3.detshift.better", num(-wv["detshift"]["D"], 0, scale=100), src_dw + " (det+shift above none)")
    put("dw.wall3.detshift.success", num(wv["detshift"]["success"], 0, scale=100), src_dw)
    put("dw.wall3.shiterb.minus.shiter", pci(w["aux"]["D(sh-iter+b)-D(sh-iter)"]["diff"], *w["aux"]["D(sh-iter+b)-D(sh-iter)"]["ci"], 0, 100), src_dw)
    put("dw.wall3.D.detnoise", pci(wv["detnoise"]["D"], *wv["detnoise"]["D_ci"], 0, 100), src_dw)
    pu = cells[("pusht", 4)]["variants"]
    put("dw.pusht4.D.detbc", num(pu["detbc"]["D"], 0, True, 100), src_dw)
    put("dw.pusht4.D.det.pt", num(pu["det"]["D"], 0, True, 100), src_dw)
    e12d = dwj["E12"]
    pk = e12d["pusht"]["kstar"]
    put("dw.pusht3.D.det", num(pk["main|det"]["D"]["3"], 0, True, 100), src_dw + " E12 (n = 50)")
    put("dw.pusht3.D.detbc", num(pk["main|detbc"]["D"]["3"], 0, True, 100), src_dw + " E12 (n = 50)")
    put("dw.pusht3.n", pk["main|det"]["n"]["3"], src_dw)
    put("dw.pusht.G", e12d["pusht"]["G"]["main"]["G"], src_dw)
    assert e12d["pusht"]["G"]["main"]["G"] == e12d["pusht"]["G"]["pc"]["G"] == e12d["pusht"]["G"]["rc"]["G"]
    put("dw.pm.G", e12d["pointmaze"]["G"]["main"]["G"], src_dw)
    put("dw.wall.G.max", e12d["wall"]["G"]["main"]["G"], src_dw)
    w4n = e12d["wall"]["cells"]["main"]["det"]["4"]
    assert w4n["D"] < 0 and w4n["D_ci"][1] < 0, w4n
    put("dw.wall4.det.above", pci(-w4n["D"], -w4n["D_ci"][1], -w4n["D_ci"][0], 0, 100, sign=False),
        src_dw + " E12 (det at 4 bits above none, new split)")
    wk = e12d["wall"]["cells"]["main"]["detbc"]["3"]
    put("dw.wall3.D.detbc", pci(wk["D"], *wk["D_ci"], 0, 100), src_dw + " E12/E11 (threshold 10 points)")
    put("dw.wall.kstar.detbc", e12d["wall"]["kstar"]["main|detbc"]["kstar"], src_dw)
    rc3 = e12d["pusht"]["cells"]["rc"]["det"]["3"]["D"]
    put("dw.pusht3.D.det.rc", num(rc3, 0, True, 100), src_dw)
    put("dw.pusht3.D.det.pc", num(pk["pc|det"]["D"]["3"], 0, True, 100), src_dw)
    put("dw.pusht.kstar.det", pk["main|det"]["kstar"], src_dw)
    put("dw.pusht.kstar.shiter", pk["main|shiter"]["kstar"], src_dw)
    dgd = dwj["Dg1"]["cells"][0]
    put("dg1.dw", num(dwj["Dg1"]["unit_mean_diff"], 2), src_dw)
    put("dg1.dw.pr", num(dgd["rho_pr_D"], 2), src_dw)
    put("dg1.dw.lat", num(dgd["rho_lat1_D"], 2), src_dw)
    e16d = dwj["E16"]
    ups = []
    for task, name in (("pointmaze", "pm"), ("wall", "wall")):
        dd = e16d[task]["D(sh-plan)-D(sh-iter)"]
        for pl in ("cem", "mppi"):
            put(f"e16.dw.{pl}.{name}", pci(dd[pl]["diff"], *dd[pl]["ci"], 0, 100), src_dw + " E16")
            ups.append(dd[pl]["ci"][1])
        put(f"e16.dw.none.{name}", num(e16d[task]["ratio"], 2), src_dw + " E16 (none MPPI-like / CEM)")
    put("e16.dw.ciupper", rng(min(ups), max(ups), 0, 100), src_dw)
    wall_rerun = load(os.path.join(TX, "DW_dinowm", "results", "raw", "dw.4", "diag", "wall", "summary_eval.json"))
    put("dw.wall.rerun.none", wall_rerun["successes"], "DW_dinowm/results/raw/dw.4/diag/wall/summary_eval.json (D3)")

    # ---- E17: integer kernel, latency, pre-cast BF16
    src17 = "tmlr_ext/K_kernel/results/raw/k.1/analysis/e17_summary.json"
    k17 = load(os.path.join(A, src17))
    srcl = "tmlr_ext/K_kernel/results/raw/k.1/analysis_lat/e17_lat_summary.json"
    kl = load(os.path.join(A, srcl))
    cl = kl["cells"]
    put("e17.shiter.overhead.max", num(100 * kl["shiter_descriptive"]["max_median"], 2), srcl)
    put("e17.shiter.overhead.max1", num(100 * kl["shiter_descriptive"]["max_median"], 1), srcl)
    put("e17.shiter.overhead.max0", num(100 * kl["shiter_descriptive"]["max_median"], 0), srcl)
    put("e17.detshift.5m512", num(100 * cl["5|512"]["int-detshift"]["median"], 2), srcl)
    put("e17.detshift.317m512", num(100 * cl["317|512"]["int-detshift"]["median"], 2), srcl)
    big = [100 * cl[k]["int-detshift"]["median"] for k in ("5|2048", "5|8192", "317|2048", "317|8192")]
    put("e17.detshift.big", rng(min(big), max(big), 2), srcl)
    put("e17.k1.n", len(kl["K1"]["cells_median_le_5pct"]), srcl)
    put("e17.k1.of", kl["K1"]["n_cells"], srcl)
    # K2: fake-quant (td E11) vs integer kernel; same sign and overlapping intervals
    k2f = td["K2_E11_det_minus_detshift"]
    k2i = k17["K2"]["integer_side"]
    n_ok = 0
    for t in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk"):
        f, i = k2f[t], k2i[t]
        same_sign = (f["D"] > 0) == (i["delta_int"] > 0)
        overlap = f["ci"][0] <= i["delta_int_ci"][1] and i["delta_int_ci"][0] <= f["ci"][1]
        n_ok += same_sign and overlap
        put(f"e17.k2.{SHORT[t]}.fq", pci(f["D"], *f["ci"], 1, 100), src_td + " K2_E11_det_minus_detshift")
        put(f"e17.k2.{SHORT[t]}.int", pci(i["delta_int"], *i["delta_int_ci"], 1, 100), src17)
    put("e17.k2.n", n_ok, "td_summary.json + e17_summary.json (same sign, overlapping CIs)")
    k3 = k17["K3"]
    put("e17.k3.n", k3["n_tasks_ci_contains_0"], src17)
    put("e17.k3.of", k3["n_tasks_with_data"], src17)
    pt3 = k3["per_task"]
    put("e17.bf16.dooropen", pci(pt3["mw-door-open"]["D"], *pt3["mw-door-open"]["D_ci"], 1, 100), src17)
    put("e17.bf16.cheetah", pci(pt3["cheetah-run"]["D"], *pt3["cheetah-run"]["D_ci"], 2, 100), src17)
    put("e17.bf16.finger", pci(pt3["finger-turn-hard"]["D"], *pt3["finger-turn-hard"]["D_ci"], 2, 100), src17)
    # folded per-channel integer configuration vs E12 fake-quant per-channel (coordinator comparison)
    srcpc = "tmlr_ext/coordinator/e17_pc_vs_e12.txt"
    n_in, n_all = 0, 0
    for line in open(os.path.join(A, srcpc)):
        m = re.search(r"\|\s*([+-][0-9.]+) \[([+-][0-9.]+),([+-][0-9.]+)\]\s*$", line)
        if m and "diff" not in line:
            n_all += 1
            lo, hi = float(m.group(2)), float(m.group(3))
            n_in += lo < 0 < hi
    put("e17.pc.n", n_in, srcpc + " (difference CI strictly contains 0)")
    put("e17.pc.of", n_all, srcpc)


# ---------------------------------------------------------------------------------------------- re-analyses R1-R5
def floor_pct(p, nd=1):
    """A probability as a percentage rounded DOWN, for 'at least x%' statements."""
    q = Decimal(repr(float(p) * 100)).quantize(Decimal(1).scaleb(-nd), rounding="ROUND_FLOOR")
    return f"{q:.{nd}f}"


def reanalysis():
    """Coordinator re-analyses R1-R5 (tmlr_ext/coordinator/reanalysis/, 2026-11-22; CLAIMS v2.1 sec. 5.2)."""
    rd = os.path.join(TX, "coordinator", "reanalysis")
    r1 = load(os.path.join(rd, "r1.json"))
    src1 = "tmlr_ext/coordinator/reanalysis/r1.json"
    ps = []
    for basis in ("best", "shiter", "detshift"):
        for fr in ("Fr1", "Fr2"):
            p = r1["fr"][basis][fr]["P_holds"]
            ps.append(p)
            put(f"r1.{basis}.{fr.lower()}", num(p, 3), src1 + f" fr.{basis}.{fr}.P_holds")
    put("r1.fr.pmin", floor_pct(min(ps), 1), src1 + " (smallest P_holds over the three bases, as % rounded down)")
    for t, name in (("mw-pick-place", "pickplace"), ("mw-window-open", "windowopen")):
        for rg in ("PT", "PC"):
            put(f"r1.{name}.{rg.lower()}", num(r1["td"][t][f"{rg}|best"]["P_ge1"], 2), src1 + f" td.{t}.{rg}|best.P_ge1")
    dw = r1["dw"]
    for rg in ("PT", "PC", "RC"):
        put(f"r1.pusht.{rg.lower()}.ge1", num(dw["pusht"][rg]["P_ge1"], 3), src1 + f" dw.pusht.{rg}.P_ge1")
        put(f"r1.pusht.{rg.lower()}.ge2", num(dw["pusht"][rg]["P_ge2"], 2), src1 + f" dw.pusht.{rg}.P_ge2")
        assert dw["pusht"][rg]["G_point"] == 2
    # wording used in the text: PushT gains 2 bits where P(G>=2) is high, '1--2' where only P(G>=1) is
    assert dw["pusht"]["PT"]["P_ge2"] >= 0.95
    assert all(dw["pusht"][rg]["P_ge1"] >= 0.95 and dw["pusht"][rg]["P_ge2"] < 0.95 for rg in ("PC", "RC"))
    put("r1.pusht.pcrc.G", "1--2", src1 + " (PC, RC: P(G>=1) >= 0.95, P(G>=2) < 0.95; CLAIMS v2.1 5.2 item 1)")
    assert all(dw["wall"][rg]["G_point"] == 2 and dw["wall"][rg]["P_ge1"] < 0.95 for rg in ("PT", "PC"))
    put("dw.wall.G.range", "0--2", src1 + " (Wall: point G = 2, P(G>=1) < 0.95; CLAIMS v2 5.1 item 5)")
    for rg in ("PT", "PC"):
        put(f"r1.wall.{rg.lower()}.ge1", num(dw["wall"][rg]["P_ge1"], 2), src1 + f" dw.wall.{rg}.P_ge1")
        put(f"r1.pm.{rg.lower()}.ge1", num(dw["pointmaze"][rg]["P_ge1"], 2 if dw["pointmaze"][rg]["P_ge1"] > 0.01 else 3),
            src1 + f" dw.pointmaze.{rg}.P_ge1")

    # R2: threshold sensitivity (tau 0.05-0.20; '0.10_strict' = D < 0.10)
    src2 = "tmlr_ext/coordinator/reanalysis/r2_fr.csv"
    rows = list(csv.DictReader(open(os.path.join(rd, "r2_fr.csv"))))
    taus = sorted({float(r["tau"]) for r in rows if r["tau"] != "0.10_strict"})
    put("r2.tau", rng(min(taus) * 100, max(taus) * 100, 0), src2 + " (points)")
    assert all(r["holds"] == "True" for r in rows), "an Fr test fails at some tau"
    for fr in ("Fr1", "Fr2"):
        cs = [int(r["count"]) for r in rows if r["label"] == fr and r["dither_basis"] == "best" and r["tau"] != "0.10_strict"]
        put(f"r2.{fr.lower()}.best", rng(min(cs), max(cs), 0), src2 + " (better dither, over tau)")
    strict = [r for r in rows if r["tau"] == "0.10_strict" and r["label"] == "Fr2" and r["dither_basis"] == "best"][0]
    put("r2.fr2.strict", strict["count"], src2 + " (D < 0.10)")
    near = list(csv.DictReader(open(os.path.join(rd, "r2_near_tau.csv"))))
    exact = [r for r in near if r["tau"] == "0.1" and r["family"] == "TD-MPC2" and r["exactly_tau"] == "True"
             and r["task"] in MW]
    put("r2.exact.mw", len(exact), "tmlr_ext/coordinator/reanalysis/r2_near_tau.csv (Meta-World cells with D = 0.100)")

    # R4 [post hoc]: E1 + eval8 pooled (inverse variance, unpaired)
    src4 = "tmlr_ext/coordinator/reanalysis/r4.json"
    r4 = load(os.path.join(rd, "r4.json"))["cells"]
    for task, name in (("pointmaze", "pm3"), ("wall", "wall3")):
        pl = r4[task]["D_det"]["pooled"]
        put(f"r4.{name}.pooled", pci(pl["est"], *pl["ci"], 0, 100), src4 + f" cells.{task}.D_det.pooled")
        put(f"r4.{name}.i2", num(100 * pl["I2"], 0), src4)
    pp = r4["pusht"]["D_det"]["pooled"]
    put("r4.pusht4.pQ", num(pp["p_Q"], 3), src4 + " cells.pusht.D_det.pooled.p_Q")
    put("r4.pusht4.i2", num(100 * pp["I2"], 0), src4)
    pc = r4["pusht"]["sensitivity_eval8_cluster"]["D_det"]["pooled"]
    put("r4.pusht4.pQ.cluster", num(pc["p_Q"], 3), src4 + " (eval8 trajectory-cluster variance)")

    # R5: persistence costs side by side
    src5 = "tmlr_ext/coordinator/reanalysis/r5.json"
    r5 = load(os.path.join(rd, "r5.json"))
    pu = r5["dw"]["pusht|cem"]
    put("r5.pusht4.shplan", pci(pu["shplan_minus_shiter"]["est"], *pu["shplan_minus_shiter"]["ci"], 0, 100), src5 + " dw.pusht|cem")
    cc = pu["cluster_ci"]["shplan_minus_shiter"]
    put("r5.pusht4.shplan.cluster", f"[{num(cc[0], 1, scale=100)}, {num(cc[1], 1, scale=100)}]", src5 + " (trajectory clusters)")
    ups = [x["shplan_minus_shiter"]["ci"][1] for x in r5["dw"].values()]
    put("r5.dw.ciupper", rng(min(ups), max(ups), 0, 100), src5 + " (all five DINO-WM D(sh-plan)-D(sh-iter) cells, incl. PushT)")
    lb = r5["td_lower_bounds"]
    m5 = min(lb["mw5m_shep"])
    put("r5.td5m.minlower", num(m5[0], 1, scale=100), src5 + f" td_lower_bounds.mw5m_shep ({m5[1]})")
    put("r5.td5m.minlower.task", m5[1], src5)
    m3 = min(lb["mt317m_shep"])
    put("r5.td317m.minlower", num(m3[0], 1, scale=100), src5 + f" td_lower_bounds.mt317m_shep ({m3[1]})")
    sb = [x for x in r5["side_by_side"] if x["contrast"] == "D(sh-plan)-D(sh-iter)"]
    a = [x["n_mw5m_lower_above"] for x in sb]
    b = [x["n_mt317m_lower_above"] for x in sb if x["n_mt317m_lower_above"] is not None]
    put("r5.n5m.above", rng(min(a), max(a), 0), src5 + " side_by_side (5M Meta-World cells, of 8)")
    put("r5.n317m.above", rng(min(b), max(b), 0), src5 + " side_by_side (317M tasks, of 8)")


# ---------------------------------------------------------------------------------------------- post hoc D7 (X6)
def x6_values():
    """X6 (D7, post hoc): TD-MPC2 policy without planning (FP32 pi) against the unquantized planner, next to E12's
    lowest-bit dither configuration (CLAIMS v2.1 sec. 5.3, C4)."""
    src = "tmlr_ext/TD_tdmpc2/results/analysis_x6/x6_summary.json"
    x6 = load(os.path.join(A, src))["tasks"]
    mw = {t: x for t, x in x6.items() if x["unit"] == "mw"}
    dmc = {t: x for t, x in x6.items() if x["unit"] != "mw"}
    assert len(mw) == 8 and len(dmc) == 4
    assert all(x["none_vs_e11"]["identical"] == x["none_vs_e11"]["n"] == 90 for x in x6.values())
    pp = mw["mw-pick-place"]
    put("x6.mw.dpol.pickplace", pci(pp["D_policy"], *pp["D_policy_ci"], 1, 100), src)
    rest = {t: x["D_policy"] for t, x in mw.items() if t != "mw-pick-place"}
    assert max(rest, key=rest.get) == "mw-push" and pp["D_policy"] > max(rest.values())
    put("x6.mw.dpol.max7", num(max(rest.values()), 1, scale=100), src + " (largest of the other seven Meta-World tasks: push)")
    # the lowest-bit dither planner is never worse than the policy on Meta-World, and better on 3 of 4 DMC tasks
    assert not any(x["dither_vs_policy"] == "worse" for x in x6.values())
    put("x6.dmc.better", sum(x["dither_vs_policy"] == "better" for x in dmc.values()), src)
    assert dmc["finger-turn-hard"]["dither_vs_policy"] == "same"
    put("x6.dmc.dpol", rng(min(x["D_policy"] for x in dmc.values()), max(x["D_policy"] for x in dmc.values()), 1, 100), src)
    ks = [x["e12_lowest_dither"]["k"] for x in dmc.values()]
    assert all(x["e12_lowest_dither"]["range"] == "PC" for x in dmc.values())
    put("x6.dmc.k", rng(min(ks), max(ks), 0), src + " (per-channel ranges)")
    put("x6.mw.better", sum(x["dither_vs_policy"] == "better" for x in mw.values()), src)
    put("x6.mw.same", sum(x["dither_vs_policy"] == "same" for x in mw.values()), src)


# ---------------------------------------------------------------------------------------------- post hoc D4, D5 (X17), D6 (X5)
def posthoc():
    """Post hoc cells added after the pre-specified analyses (DESIGN_tmlr 11, D4-D6; CLAIMS v2.1 5.3)."""
    # D4: DINO-WM PushT 3-bit E11 cell on the new split (dw.6)
    src = "tmlr_ext/DW_dinowm/results/analysis_posthoc/posthoc_pusht3.json"
    d4 = load(os.path.join(A, src))
    c = d4["cell"]
    v = c["variants"]
    put("d4.D.det", pci(v["det"]["D"], *v["det"]["D_ci"], 0, 100), src)
    cl = c["cluster"]["D"]["det"]["ci"]
    put("d4.D.det.cluster", f"[{num(cl[0], 0, scale=100)}, {num(cl[1], 0, scale=100)}]", src + " (trajectory clusters)")
    put("d4.none", num(c["none"]["success"], 0, scale=100), src)
    for var in ("det", "detbc"):
        put(f"d4.{var}.success", num(v[var]["success"], 0, scale=100), src)
    for var in ("detbc", "shstep", "detshiftfix", "shiterb", "shiter", "detshift", "detnoise"):
        put(f"d4.red.{var}", num(v[var]["red"], 2), src)
    put("d4.dpers", pci(c["d_pers"]["diff"], *c["d_pers"]["ci"], 0, 100), src)
    put("d4.dbias", pci(c["d_bias"]["diff"], *c["d_bias"]["ci"], 0, 100), src)
    aux = c["aux"]
    put("d4.shplan.minus.shiter", pci(aux["D(sh-plan)-D(sh-iter)"]["diff"], *aux["D(sh-plan)-D(sh-iter)"]["ci"], 0, 100), src)
    put("d4.shiterb.minus.shiter", pci(aux["D(sh-iter+b)-D(sh-iter)"]["diff"], *aux["D(sh-iter+b)-D(sh-iter)"]["ci"], 0, 100), src)
    put("d4.detnoise.minus.det", pci(aux["D(det+noise)-D(det)"]["diff"], *aux["D(det+noise)-D(det)"]["ci"], 0, 100), src)
    for var in ("det", "detbc"):
        put(f"d4.err1.{var}", num(v[var]["rel_err1"], 2), src + " (planning-time latent error, 1 step)")
        put(f"d4.err5.{var}", num(v[var]["rel_err5"], 2), src + " (planning-time latent error, 5 steps)")
    assert c["damaged"] and c["class"] == "resolution"
    e1 = c["E1"]
    put("d4.e1.D.det", pci(e1["D_det"], *e1["D_det_ci"], 0, 100), src + " E1 (A/main/results/E1/e1_dinowm_cells.csv)")
    comb = d4["combined_dinowm"]
    put("d4.comb.ndam", comb["n_damaged"], src + " combined_dinowm")
    put("d4.comb.detnoise", comb["detnoise_fixes"], src + " combined_dinowm")
    srcc = "tmlr_ext/DW_dinowm/results/analysis/calib_summary_dw.json"
    cal = [x for x in load(os.path.join(A, srcc))["bias_cells"] if x["task"] == "pusht" and x["ranges"] == "main" and x["k"] == 3][0]
    put("d4.b.rel5", f"{num(cal['latent_rel5_b0'], 2)} to {num(cal['latent_rel5_final'], 2)}", srcc + " (PushT PT k3: b = 0 -> final b)")

    # D5 (X17): PushT 4-bit on the E1 pairs (eval6), RTX 4090
    src = "tmlr_ext/DW_dinowm/results/analysis_posthoc/x17_pusht4_eval6.json"
    x = load(os.path.join(A, src))
    put("x17.D.det", pci(x["det"]["D"], *x["det"]["D_ci"], 0, 100), src)
    put("x17.none", num(x["cell"]["none"]["success"], 0, scale=100), src)
    put("x17.det.success", num(x["det"]["success"], 0, scale=100), src)
    put("x17.e1.det.success", num(x["E1"]["det"], 0, scale=100), src + " E1")
    put("x17.red.shiter", num(x["shiter"]["red"], 2), src)
    put("x17.agree.det", num(100 * x["agreement_with_E1"]["det"]["agreement"], 0), src + " (episode agreement with E1, %)")
    assert x["reading"] == "(i)"

    # D6 (X5): Meta-World det+noise bit curve
    src = "tmlr_ext/TD_tdmpc2/results/analysis_x5/x5_summary.json"
    x5 = load(os.path.join(A, src))
    cnt = x5["counts"]
    for rg in ("PT", "PC"):
        put(f"x5.{rg.lower()}.nG", cnt[rg]["G_noise_ge_1"], src + f" counts.{rg}.G_noise_ge_1")
        put(f"x5.{rg.lower()}.le.shiter", cnt[rg]["noise_le_shiter"], src + f" counts.{rg}.noise_le_shiter")
        assert cnt[rg]["n_tasks"] == len(MW)
    cv = x5["curves"]
    for t, k, name in (("mw-door-open", "3", "dooropen3"), ("mw-door-open", "2", "dooropen2"),
                       ("mw-button-press-topdown", "2", "buttonpress2"), ("mw-sweep-into", "2", "sweepinto2"),
                       ("mw-drawer-open", "1", "draweropen1"), ("mw-drawer-open", "2", "draweropen2")):
        put(f"x5.pt.{name}", num(cv[f"{t}|PT"][k]["D"], 1, scale=100), src + f" curves.{t}|PT.{k}")
    # the one X5 configuration with k* = 1 (drawer-open, per tensor) sits on the threshold at 2 bits
    one = [key for key, x in x5["kstar"].items() if x["kstar_detnoise"] == 1]
    assert one == ["mw-drawer-open|PT"] and abs(cv["mw-drawer-open|PT"]["2"]["D"] - 0.10) < 1e-9, one
    rep = x5["repro_e11"]
    assert all(r["bit_identical_episodes"][v]["identical"] == r["bit_identical_episodes"][v]["n"]
               for r in rep.values() for v in ("none", "detnoise"))
    put("x5.repro.ntasks", len(rep), src + " repro_e11 (all episodes bit-identical)")


# ---------------------------------------------------------------------------------------------- appendices A-K
def _sha12(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _design_prefix(path, marker):
    """Text of an append-only design document up to (not including) a later section, as it was hashed when that
    earlier version was frozen: trailing newlines stripped, one newline added."""
    t = open(os.path.join(A, path), encoding="utf-8").read()
    return t[:t.index(marker)].rstrip("\n") + "\n"


def _spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5


def appendix_values():
    """Values quoted in the appendix files (paper/tmlr/appendix/app_*.tex)."""
    # ---- A: design versions. The design documents are append-only, so an earlier version is the current file cut
    # before the first later section. The cut reproduces the logged hashes of v4, v5 and v5.2; v5.1's hash is not in
    # the coordinator log, only its first 8 hex digits in two worker logs.
    log_main = open(os.path.join(A, "main", "LOG.md"), encoding="utf-8").read()
    dm = "main/DESIGN_main.md"
    for marker, logged in (("## 7. v5 ", "b814a4092a4b"), ("## 8. v5.1", "a7895d735cf4"), ("## 10. v5.3", "41e1f39d6bd8")):
        assert _sha12(_design_prefix(dm, marker)) == logged and logged in log_main, marker
    v51 = _sha12(_design_prefix(dm, "## 9. v5.2"))
    wl = open(os.path.join(A, "main", "worklogs", "E1_tdmpc2.md"), encoding="utf-8").read()
    assert re.search(r"v5\.1 고정\(8절, sha256 " + v51[:8], wl), "v5.1 prefix not in the worker log"
    put("plan.v51.sha", v51, dm + " cut before sec. 9 (append-only; prefix in main/worklogs/E1_tdmpc2.md)")
    log_tx = open(os.path.join(TX, "LOG.md"), encoding="utf-8").read()
    dt = "tmlr_ext/DESIGN_tmlr.md"
    assert _sha12(_design_prefix(dt, "\n---\n\n## 11. 고정 뒤 변경 기록")) == "36d69ea82ef9"   # the separator came with sec. 11
    # The D5/D6 version: hash and time as logged at its freeze. D7 (X6, post hoc) was added to sec. 11 later, so the
    # current file no longer equals this version; the D1-D6 rows must still be there.
    m = re.search(r"- (2026-11-22)T(\d\d:\d\d)Z 설계 D5\(X17\)·D6\(X5\).*?sha256은 `([0-9a-f]{64})`", log_tx)
    assert m, "D5/D6 freeze entry not found in tmlr_ext/LOG.md"
    cur = open(os.path.join(A, dt), encoding="utf-8").read()
    assert all(f"| D{i} |" in cur for i in range(1, 7)), "D1-D6 rows missing from DESIGN_tmlr.md sec. 11"
    put("plan.tmlr.sha", m.group(3)[:12], "tmlr_ext/LOG.md (hash logged at the D5/D6 freeze)")
    put("plan.tmlr.time", f"{m.group(1)} {m.group(2)}", "tmlr_ext/LOG.md")
    # D7 (X6): the current file is the D7 version; its logged hash prefix (16 hex digits) must match
    import hashlib
    h_now = hashlib.sha256(open(os.path.join(A, dt), "rb").read()).hexdigest()
    i7 = log_tx.index("설계 D7(X6)을")
    m7 = re.search(r"sha256 앞 16자리 ([0-9a-f]{16})", log_tx[i7:i7 + 200])
    t7 = re.findall(r"\n- (2026-11-22)T(\d\d:\d\d)Z", log_tx[:i7])[-1]   # the log entry that records the freeze
    assert m7 and h_now.startswith(m7.group(1)), "DESIGN_tmlr.md changed after the D7 freeze"
    put("plan.tmlr.d7.sha", h_now[:12], dt + " (current file; its first 16 hex digits are logged in tmlr_ext/LOG.md)")
    put("plan.tmlr.d7.time", f"{t7[0]} {t7[1]}", "tmlr_ext/LOG.md")

    # ---- B: planned tests of E1-E10 (outcomes; the table itself is built in make_tables.py)
    R = RES
    e1t = load(os.path.join(R, "E1_tdmpc2", "e1_tdmpc2_summary.json"))
    src = "main/results/E1_tdmpc2/e1_tdmpc2_cells.csv"
    by = {}
    for r in csv.DictReader(open(os.path.join(A, src))):
        by.setdefault((r["task"], int(r["k"]), r["family"]), {})[r["variant"]] = r
    aff = {k: v for k, v in by.items() if v["det"]["confirmatory"] == "True" and v["det"]["holm_real"] == "True"}
    h1 = {f: sum(float(v["det"]["pt_err1h"]) <= float(v["shiter"]["pt_err1h"]) for k, v in aff.items() if k[2] == f)
          for f in ("mw", "dmc")}
    put("b.h1.td", h1["mw"] + h1["dmc"], src)
    put("b.h1.td.of", len(aff), src)
    wo = by[("mw-window-open", 4, "mw")]["shiter"]
    put("b.h3a.weakest", pci(float(wo["red"]), *ci_str(wo["red_ci"]), 2, sign=False), src + " (window-open k=4)")
    wk = by[("walker-walk", 3, "dmc")]["shiter"]
    put("b.h3a.walker", pci(float(wk["Dv_minus_Ddet"]), *ci_str(wk["Dv_minus_Ddet_ci"]), 1, 100), src + " (walker k=3)")
    dws = load(os.path.join(R, "E1", "e1_dinowm_summary.json"))["families"]["dinowm"]
    put("b.h1.dw", len(dws["H1"]["cells_meeting"]), "main/results/E1/e1_dinowm_summary.json")
    put("b.h1.dw.of", len(dws["qualifying_cells"]), "main/results/E1/e1_dinowm_summary.json")
    # H2a (frozen OEI) and its E2 replication, H2a_indep and H2a' (layer b, added after E1)
    fp = load(os.path.join(R, "E2_tdmpc2", "family_points_tdmpc2.json"))
    v52 = load(os.path.join(R, "E1", "e1_dinowm_v52.json"))
    srcfp = "main/results/E2_tdmpc2/family_points_tdmpc2.json"
    n_h2a = sum(e1t["H2a"][t]["holds"] for t in e1t["H2a"]) + int(dws["H2a"]["holds"])
    put("b.h2a.n", n_h2a, "e1_tdmpc2_summary.json H2a + e1_dinowm_summary.json H2a")
    ps = e1t["H2a"]["mw-push"]
    put("b.h2a.push.shiter", pci(ps["shiter_oei"], *ps["shiter_oei_ci"], 1), "e1_tdmpc2_summary.json H2a.mw-push")
    n_rep = fp["tdmpc2_tasks_holding"]["H2a_replication"] + int(dws["H2a"]["holds"])
    put("b.h2a.rep.n", n_rep, srcfp + " (+ PushT E1)")
    n_b1 = fp["tdmpc2_tasks_holding"]["b1_H2a_indep"] + int(v52["b1_H2a_indep_pusht"]["holds"])
    put("b.h2aindep.n", n_b1, srcfp + " + e1_dinowm_v52.json")
    rs = [fp["tasks"][t]["b2_H2a_prime"]["r_shiter"]["r"] for t in ("mw-push", "mw-pick-place", "cheetah-run")]
    rs.append(v52["b2_H2a_prime_pusht_k4"]["r_shiter"])
    n_b2 = fp["tdmpc2_tasks_holding"]["b2_H2a_prime"] + int(v52["b2_H2a_prime_pusht_k4"]["holds"])
    put("b.h2aprime.n", n_b2, srcfp + " + e1_dinowm_v52.json")
    put("b.h2aprime.rmax", num(max(rs), 2), srcfp + " (largest drift ratio of sh-iter)")
    put("b.h2b.td", num(e1t["H2b"]["primary_within_task"]["task_average"]
                        if "task_average" in e1t["H2b"]["primary_within_task"] else
                        statistics.mean(x["diff"] for x in e1t["H2b"]["primary_within_task"]["per_task"].values()), 3),
        "e1_tdmpc2_summary.json H2b.primary_within_task")
    put("b.h2b.dw", num(dws["H2b"]["primary_task_average"], 3), "e1_dinowm_summary.json H2b")
    # H2c
    fpd = load(os.path.join(R, "E2_dinowm", "family_points_dinowm.json"))
    put("b.h2c.n", fp["tdmpc2_tasks_holding"]["H2c"] + int(fpd["H2c_pusht"]), srcfp + " + family_points_dinowm.json")
    put("b.h2c.pusht.oei", num(fpd["rho"]["N_absOEI_det"], 2), "family_points_dinowm.json rho")
    put("b.h2c.pusht.d", num(fpd["rho"]["N_D_det"], 2), "family_points_dinowm.json rho")
    # H4a (table size of fixed dither), H4b (near-tie density), DMC rule
    e3s = load(os.path.join(R, "E3_tdmpc2", "e3_tdmpc2_summary.json"))
    e3d = load(os.path.join(R, "E3_dinowm", "family_points_e3_dinowm.json"))
    h4a_td = e3s["H4a"]
    rho_td = [e3s["sweep"][t]["spearman"] for t in h4a_td["eligible_tasks"]]
    rho_dw = [x["rho_Neff_D"] for x in e3d["tasks"].values() if x.get("eligible") and x.get("role") == "primary"]
    passing = [r for r in rho_td + rho_dw if r <= -0.8 + TOL]
    put("b.h4a.td", f"{h4a_td['n_holding']}/{h4a_td['n_eligible']}", "e3_tdmpc2_summary.json H4a")
    put("b.h4a.dw", f"{e3d['H4a_dinowm']['n_passed']}/{e3d['H4a_dinowm']['n_eligible']}", "family_points_e3_dinowm.json")
    put("b.h4a.exact", sum(abs(r + 0.8) < 1e-6 for r in passing), "E3 sweeps (Spearman exactly -0.8)")
    f3 = load(os.path.join(R, "E3_tdmpc2", "e3_tdmpc2_final.json"))
    put("b.h4b.rho", num(f3["H4b"]["spearman"], 2, sign=True), "main/results/E3_tdmpc2/e3_tdmpc2_final.json H4b")
    put("b.h4b.rho.first", num(f3["H4b"]["aux_spearman_first_set"], 2), "e3_tdmpc2_final.json H4b (first set)")
    put("b.dmc.rho", num(f3["DMC_explanation"]["spearman_S_red"], 2), "e3_tdmpc2_final.json DMC_explanation")
    assert f3["DMC_explanation"]["verdict"] == "explained"
    # H3b, QAT rule
    f5 = load(os.path.join(R, "E5_tdmpc2", "family_points_e5_tdmpc2.json"))
    f5d = load(os.path.join(R, "E5_dinowm", "family_points_e5_dinowm.json"))
    put("b.h3b.td", f"{f5['H3b']['passing']}/{f5['H3b']['cells']}", "family_points_e5_tdmpc2.json H3b")
    put("b.h3b.dw", f"{f5d['H3b']['passing']}/{f5d['H3b']['cells']}", "family_points_e5_dinowm.json H3b")
    qd = [x["D_qat_det_ci"][1] for x in f5["qat_cells"].values()]
    put("b.qat.td.n", sum(x < 0.05 for x in qd), "family_points_e5_tdmpc2.json qat_cells (CI upper < 5 points)")
    assert sum(x < 0.05 for x in qd) == 0 and min(qd) > 0.3   # app_G: "never removed the damage", "far above 5"
    put("b.qat.td.of", len(qd), "family_points_e5_tdmpc2.json qat_cells")
    put("b.qat.dw.upper", num(100 * f5d["qat"]["conditions"]["det"]["D_ci"][1], 0), "family_points_e5_dinowm.json qat")
    # H5
    f7 = load(os.path.join(R, "E7_tdmpc2", "family_points_e7_tdmpc2.json"))
    ovs = [v for kern in ("int8", "int4", "int4p") for v in f7["overhead"][kern]["per_condition"].values()]
    put("b.h5.max", num(100 * max(ovs), 1), "family_points_e7_tdmpc2.json overhead (int8, int4, packed int4)")
    put("b.h5.min", num(100 * min(ovs), 1), "family_points_e7_tdmpc2.json overhead")
    pareto_tasks = sorted({t for t, cases in f7["pareto_cases"].items() if cases})
    assert pareto_tasks == ["mw-push"], pareto_tasks
    s7 = load(os.path.join(R, "E7_tdmpc2", "e7_tdmpc2_summary.json"))["latency_rel_fp32"]
    slow = {kern: statistics.median(v for k, v in s7.items() if f"|{kern}_det|" in k) for kern in ("int8", "int4", "int4p")}
    put("b.h5.slow", rng(100 * min(slow.values()), 100 * max(slow.values()), 1),
        "main/results/E7_tdmpc2/e7_tdmpc2_summary.json latency_rel_fp32 (median over conditions per kernel)")
    # E8, E8c, E9, E10 tests
    e8s = load(os.path.join(EXT, "E8_latency", "results", "analysis", "e8_summary.json"))["verdict"]
    put("b.l1.n", len(e8s["L1"]["cells"]), "rss_ext/E8_latency/results/analysis/e8_summary.json")
    put("b.l2.n", len(e8s["L2"]["cells"]), "e8_summary.json")
    put("b.l3.max", num(100 * e8s["L3"]["max_median"], 1), "e8_summary.json")
    c8 = load(os.path.join(EXT, "E8c_latency", "results", "analysis", "e8c_summary.json"))
    put("b.l3.max2", num(100 * c8["e8_replication_same_pod"]["L3"]["median_range"][1], 1), "e8c_summary.json (E8 replication)")
    vc = c8["verdict"]
    put("b.l2c.n", vc["L2c"]["n_pass"], "rss_ext/E8c_latency/results/analysis/e8c_summary.json")
    put("b.l2c.range", rng(*vc["L2c"]["range"], 2), "e8c_summary.json")
    put("b.s1.n", vc["S1"]["n_pass"], "e8c_summary.json")
    put("b.s1.range", rng(*vc["S1"]["range"], 2), "e8c_summary.json")
    put("b.s2.n", vc["S2"]["n_pass"], "e8c_summary.json")
    put("b.s2.range", rng(*vc["S2"]["range"], 2), "e8c_summary.json")
    put("b.s3.max", num(100 * vc["S3"]["median_range"][1], 1), "e8c_summary.json")
    e9 = load(os.path.join(EXT, "E9_fp4", "results", "e9_summary.json"))["judgements"]
    put("b.f1.nv", e9["F1"]["nvfp4"]["n_damaged"], "rss_ext/E9_fp4/results/e9_summary.json")
    put("b.f1.mx", e9["F1"]["mxfp4"]["n_damaged"], "e9_summary.json")
    put("b.f2.mw", f"{e9['F2']['mw']['n_red_ge_half']}/{e9['F2']['mw']['n_damaged']}", "e9_summary.json")
    put("b.f2.dmc", f"{e9['F2']['dmc']['n_red_ge_half']}/{e9['F2']['dmc']['n_damaged']}", "e9_summary.json")
    put("b.f3", f"{e9['F3']['n_meeting']}/{e9['F3']['n_damaged']}", "e9_summary.json")
    j10 = load(os.path.join(EXT, "E10_mt317m", "results", "analysis", "e10_judgement.json"))
    put("b.g1", j10["G1"]["n_damaged"], "rss_ext/E10_mt317m/results/analysis/e10_judgement.json")
    put("b.g2", len(j10["G2"]["tasks_red_ge_half"]), "e10_judgement.json")
    put("b.g3", j10["G3"]["n"], "e10_judgement.json")
    put("b.g.of", len(j10["eval_tasks"]), "e10_judgement.json")

    # ---- C: bias vector (deviation D2), E16 temperature, ranges
    td = load(os.path.join(TX, "TD_tdmpc2", "results", "analysis", "td_summary.json"))
    bs = td["judgements"]["D2_two_readings"]["bias_summary"]
    put("c.bias.td.n", sum(bs["cells_by_stage"].values()), "td_summary.json judgements.D2_two_readings.bias_summary")
    for st, name in (("alpha=1", "a1"), ("alpha=0.5", "a2"), ("alpha=0.25", "a4")):
        put(f"c.bias.td.{name}", bs["cells_by_stage"][st], "td_summary.json bias_summary")
    put("c.bias.td.maxupd", bs["max_updates"], "td_summary.json bias_summary")
    assert bs["n_unconverged"] == 0
    mtb = list(csv.DictReader(open(os.path.join(TX, "MT_317m", "results", "analysis", "e14_bias.csv"))))
    eta = {r["task"]: r["final_eta"] for r in mtb}
    put("c.bias.mt.a2", sum(v == "0.5" for v in eta.values()), "tmlr_ext/MT_317m/results/analysis/e14_bias.csv")
    put("c.bias.mt.a4", sum(v == "0.25" for v in eta.values()), "e14_bias.csv")
    put("c.bias.mt.maxupd", max(int(r["final_n_updates"]) for r in mtb), "e14_bias.csv")
    assert all(r["converged"] == "True" for r in mtb)
    cal = load(os.path.join(TX, "DW_dinowm", "results", "analysis", "calib_summary_dw.json"))
    put("c.bias.dw.n", cal["n_bias_cells"], "tmlr_ext/DW_dinowm/results/analysis/calib_summary_dw.json")
    assert cal["n_unconverged"] == 0 and all(x["stage_eta"] == 1.0 for x in cal["bias_cells"])
    put("c.bias.dw.maxupd", cal["max_updates"], "calib_summary_dw.json")
    dwj = load(os.path.join(TX, "DW_dinowm", "results", "analysis", "analysis_dw.json"))
    for task, name in (("pointmaze", "pm"), ("wall", "wall")):
        put(f"c.e16.c.{name}", num(dwj["E16"][task]["lambda_selection"]["chosen_c"], 0 if
                                   float(dwj["E16"][task]["lambda_selection"]["chosen_c"]).is_integer() else 2),
            "analysis_dw.json E16.lambda_selection")

    # ---- E: mechanism details
    dec = load(os.path.join(R, "E1_tdmpc2", "e1_tdmpc2_posthoc_oei_decomposition.json"))
    for t, k, name in (("mw-push", 3, "push"), ("mw-pick-place", 3, "pickplace"), ("cheetah-run", 5, "cheetah")):
        put(f"e.dec.{name}.det.drift", num(dec[f"{t}:{k}:det"]["drift"], 1 if name != "cheetah" else 2),
            "e1_tdmpc2_posthoc_oei_decomposition.json")
        put(f"e.dec.{name}.shiter.sel", num(dec[f"{t}:{k}:shiter"]["sel_last"], 1 if name != "cheetah" else 2),
            "e1_tdmpc2_posthoc_oei_decomposition.json")
    e2 = load(os.path.join(R, "E2_tdmpc2", "e2_tdmpc2_summary.json"))["layer_b"]
    reg_ref, reg_own = [], []
    for t in ("mw-push", "mw-pick-place", "cheetah-run"):
        b3 = e2[t]["b3_region"]["det"]
        nd = 1 if t != "cheetah-run" else 2
        reg_ref.append(pci(b3["drift_ref"], *b3["drift_ref_ci"], nd))
        reg_own.append(num(b3["drift_minus_ref"], nd))
        assert b3["drift_minus_ref_ci"][1] < 0 and b3["drift_ref"] > 0
    put("e.region.td.ref", words_list(reg_ref), "main/results/E2_tdmpc2/e2_tdmpc2_summary.json layer_b.*.b3_region.det")
    put("e.region.td.own", words_list(reg_own), "e2_tdmpc2_summary.json layer_b.*.b3_region.det")
    sh = e2["mw-pick-place"]["b3_region"]["shiter"]
    put("e.region.pp.shiter.own", pci(sh["drift_minus_ref"], *sh["drift_minus_ref_ci"], 2), "e2_tdmpc2_summary.json b3_region")
    # DINO-WM: the unquantized planner's E1 populations scored by the quantized model (drift_ref, per episode)
    v52c = v52["cells"]
    ref = {}
    for f, task in (("E1_dinowm_pusht_pusht.json", "pusht"), ("E1_dinowm_wp_wall.json", "wall"),
                    ("E1_dinowm_wp_pointmaze.json", "pointmaze")):
        for ep in load(os.path.join(R, "E1", "drift_ref", f))["episodes"].values():
            for k, y in ep["k"].items():
                ref.setdefault((task, int(k)), []).append(y["drift_ref"])
    own = {(c["task"], c["k"]): c["vars"]["det"]["drift"] - statistics.mean(ref[(c["task"], c["k"])]) for c in v52c.values()}
    srcr = "main/results/E1/drift_ref/*.json + e1_dinowm_v52.json cells.*.vars.det.drift"
    assert all(x <= 1e-12 for x in own.values()), own   # text: "at most zero in all ... cells"
    put("e.region.dw.ncells", len(own), srcr + " (det's own drift <= 0 in every cell, asserted)")
    for k in (3, 4):
        put(f"e.region.pusht{k}.ref", num(statistics.mean(ref[("pusht", k)]), 3, sign=True), srcr)
        put(f"e.region.pusht{k}.det", num(v52c[f"pusht_k{k}"]["vars"]["det"]["drift"], 3, sign=True), srcr)
        put(f"e.region.pusht{k}.own", num(own[("pusht", k)], 3), srcr)
    # E3 two-axis contrasts (TD-MPC2, E3 seeds) and DINO-WM 2x2
    for con, name in (("ind-shiter", "crn"), ("shstep-shiter", "shstep"), ("shep-shiter", "shep"), ("shep-shstep", "shepstep")):
        put(f"e.e3.{name}.n", sum(c["contrasts"][con]["D_ci"][0] > 0 for c in e3s["cells"].values()),
            "main/results/E3_tdmpc2/e3_tdmpc2_summary.json cells.*.contrasts")
    put("e.e3.ntasks", len(e3s["cells"]), "e3_tdmpc2_summary.json")
    for con, name in (("indstep-ind", "indstep"), ("indep-ind", "indep")):
        put(f"e.e3.{name}.n", sum(c["contrasts"][con]["D_ci"][0] > 0 for c in e3s["cells"].values()),
            "e3_tdmpc2_summary.json cells.*.contrasts (per-candidate tables more harmful than ind)")
    put("e.dec.pusht4.oei", num(dws["H2a"]["det_oei"], 3), "main/results/E1/e1_dinowm_summary.json H2a (PushT k=4)")
    # drift relative to det in E11 (mechanism tasks) and in the damaged DINO-WM cells
    c11m = td["E11"]["cells"]
    mech = ("mw-push", "mw-pick-place", "cheetah-run")
    dr = lambda vs: [c11m[t]["mech"][v_]["drift_ratio_to_det"] for t in mech for v_ in vs]
    for vs, name in ((("shep", "detshiftfix"), "fixed"), (("shiter", "detshift"), "fresh"), (("detnoise",), "noise"),
                     (("detbc",), "bc")):
        xs = dr(vs)
        val = num(100 * max(xs), 0) if name == "fresh" else rng(100 * min(xs), 100 * max(xs), 0)
        put(f"e.dr.{name}", val, "td_summary.json E11.cells.*.mech.*.drift_ratio_to_det (%; fresh: largest)")
    dwcells = [x for x in dwj["E11"]["cells"] if x["damaged"]] + \
        [load(os.path.join(TX, "DW_dinowm", "results", "analysis_posthoc", "posthoc_pusht3.json"))["cell"]]
    fx = [x["variants"][v_]["drift_ratio_vs_det"] for x in dwcells for v_ in ("shstep", "detshiftfix")]
    put("e.dr.dw.fixed", num(100 * max(fx), 0), "analysis_dw.json + posthoc_pusht3.json (largest fixed unbiased drift ratio, %)")
    nz = [x["variants"]["detnoise"]["drift_ratio_vs_det"] for x in dwcells]
    put("e.dr.dw.noise", rng(100 * min(nz), 100 * max(nz), 0), "analysis_dw.json + posthoc_pusht3.json (det+noise drift ratio, %)")
    e3dw = load(os.path.join(R, "E3_dinowm", "e3_dinowm_summary.json"))["two_axis"]
    dcons = [v["ci"] for cell in e3dw.values() for kk, v in cell["contrasts"].items() if kk.startswith("D_")]
    assert all(lo <= 0 <= hi for lo, hi in dcons), "a DINO-WM two-axis contrast excludes zero"
    put("e.e3.dw.ncon", len(dcons), "main/results/E3_dinowm/e3_dinowm_summary.json two_axis (all CIs contain 0)")
    put("e.e3.dw.ncells", len(e3dw), "e3_dinowm_summary.json two_axis")
    # near-tie density and H4b
    nt = e1t["near_tie"]
    put("e.nt.td.last", rng(min(x["last"]["median"] for x in nt.values()), max(x["last"]["median"] for x in nt.values()), 3),
        "e1_tdmpc2_summary.json near_tie (gap/std median, last set)")
    put("e.nt.td.first", rng(min(x["first"]["median"] for x in nt.values()), max(x["first"]["median"] for x in nt.values()), 3),
        "e1_tdmpc2_summary.json near_tie (first set)")
    h4d = load(os.path.join(R, "E2_dinowm", "h4b_points_dinowm.json"))["tasks"]
    dl = [-x["near_tie_density_last"] for x in h4d.values()]
    put("e.nt.dw.last", rng(min(dl), max(dl), 3), "main/results/E2_dinowm/h4b_points_dinowm.json")
    gains = f3["H4b"]["crn_gain"]
    put("e.crn.n", sum(ci[0] > 0 for ci in f3["H4b"]["crn_gain_ci"]), "e3_tdmpc2_final.json H4b.crn_gain_ci")
    put("e.crn.of", len(gains), "e3_tdmpc2_final.json H4b")
    xs = f3["H4b"]["density_last"] + [x["near_tie_density_last"] for x in h4d.values()]
    xf = f3["H4b"]["density_first"] + [x["near_tie_density_first"] for x in h4d.values()]
    ys = gains + [x["crn_gain"] for x in h4d.values()]
    put("e.h4b.pool.last", num(_spearman(xs, ys), 2), "e3_tdmpc2_final.json + h4b_points_dinowm.json (15 tasks)")
    put("e.h4b.pool.first", num(_spearman(xf, ys), 2, sign=True), "e3_tdmpc2_final.json + h4b_points_dinowm.json (15 tasks)")
    # E13 on the E11 cells: multi-step latent errors and the redefined reward error
    src13 = "tmlr_ext/TD_tdmpc2/results/analysis/td_e13_cells.csv"
    e13 = {(r["task"], r["cell"], r["set"]): r for r in csv.DictReader(open(os.path.join(A, src13)))}
    c11 = td["E11"]["cells"]
    cnt = {"err1h": 0, "err3": 0, "err5": 0}
    rew = 0
    for t in c11:
        d, s = e13[(t, f"PTk{c11[t]['k']}_det", "first")], e13[(t, f"PTk{c11[t]['k']}_shiter", "first")]
        for m in cnt:
            cnt[m] += float(d[m]) <= float(s[m])
        rew += float(d["rew_med"]) > float(s["rew_med"])
    for m, name in (("err1h", "1"), ("err3", "3"), ("err5", "5")):
        put(f"e.e13.err{name}.n", cnt[m], src13 + " (E11 cells, first set, det <= sh-iter)")
    put("e.e13.rew.n", rew, src13 + " (E11 cells, first set, det > sh-iter, median |r_q - r_0| / reward sd)")
    put("e.e13.ncells", len(c11), src13)

    # ---- F: theory (E4)
    c4 = load(os.path.join(R, "E4", "e4_comparisons.json"))
    put("f.jobs", len(load(os.path.join(R, "E4", "manifest.json"))), "main/results/E4/manifest.json")
    put("f.c1.err", num(100 * c4["C1"]["normal"]["median_rel_err"], 2), "main/results/E4/e4_comparisons.json C1.normal")
    put("f.c1.n", c4["C1"]["normal"]["n"], "e4_comparisons.json C1.normal")
    for key, name in (("cem_elite|all", "elite"), ("cem_argmin_ntc|all", "argmin"), ("mppi_same_draw|all", "mppi")):
        x = c4["C2"][key]
        put(f"f.c2.{name}.slope", num(x["slope_through_origin"], 2), f"e4_comparisons.json C2.{key}")
        put(f"f.c2.{name}.r2", num(x["R2"], 3), f"e4_comparisons.json C2.{key}")
        if name != "argmin":
            put(f"f.c2.{name}.n", f"{x['n']:,}".replace(",", "{,}"), f"e4_comparisons.json C2.{key}")
    c3 = c4["C3"]
    put("f.c3.fresh.cover", rng(100 * c3["cem|fresh"]["frac_drift_ci_covers_0"], 100 * c3["mppi|fresh"]["frac_drift_ci_covers_0"], 1),
        "e4_comparisons.json C3 (CEM, MPPI)")
    put("f.c3.redraw.cover", rng(100 * min(c3["cem|fresh"]["frac_redraw_sel_ci_covers_0"], c3["mppi|fresh"]["frac_redraw_sel_ci_covers_0"]),
                                 100 * max(c3["cem|fresh"]["frac_redraw_sel_ci_covers_0"], c3["mppi|fresh"]["frac_redraw_sel_ci_covers_0"]), 1),
        "e4_comparisons.json C3")
    put("f.c3.same.below", rng(100 * c3["mppi|fresh"]["frac_samedraw_sel_ci_below_0"], 100 * c3["cem|fresh"]["frac_samedraw_sel_ci_below_0"], 1),
        "e4_comparisons.json C3 (MPPI, CEM)")
    put("f.c3.pers.cem", num(100 * c3["cem|persistent"]["frac_drift_ci_below_0"], 0), "e4_comparisons.json C3.cem|persistent")
    put("f.c3.pers.mppi", num(100 * c3["mppi|persistent"]["frac_drift_ci_below_0"], 0), "e4_comparisons.json C3.mppi|persistent")
    sl = [c4["C4"][p]["drift_vs_pred"]["slope_through_origin"] for p in ("cem|all", "mppi|all")]
    r2 = [c4["C4"][p]["drift_vs_pred"]["R2"] for p in ("cem|all", "mppi|all")]
    put("f.c4.slope", rng(min(sl), max(sl), 2), "e4_comparisons.json C4.*.drift_vs_pred (CEM, MPPI)")
    put("f.c4.r2", rng(min(r2), max(r2), 3), "e4_comparisons.json C4.*.drift_vs_pred")
    c9 = c4["C9"]["exact_vs_measured"]
    put("f.c9.ratio", num(c9["median_ratio_meas_over_exact"], 2), "e4_comparisons.json C9.exact_vs_measured")
    put("f.c9.r2", num(c9["R2"], 3), "e4_comparisons.json C9.exact_vs_measured")
    s10 = c4["C10"]["summary"]
    same_bit = [s10[b] for b in ("E|cem", "E|mppi", "R|cem", "R|mppi")]
    put("f.c10.worse", sum(round(x["frac_ds_worse_signif"] * x["cells"]) for x in same_bit), "e4_comparisons.json C10 (blocks E, R)")
    put("f.c10.of", sum(x["cells"] for x in same_bit), "e4_comparisons.json C10 (blocks E, R)")
    put("f.c10.med", rng(min(x["median_rel_dmg"] for x in same_bit), max(x["median_rel_dmg"] for x in same_bit), 2),
        "e4_comparisons.json C10 (median damage ratio det+shift / sh-iter)")
    c12 = c4["C12"]
    for fam in ("persistent", "fresh"):
        xs = [c12[f"{p}|{fam}"]["median_persistence"] for p in ("cem", "mppi")]
        put(f"f.c12.{fam}", rng(min(xs), max(xs), 2), f"e4_comparisons.json C12.*|{fam}.median_persistence (CEM, MPPI)")
    bsl = [v["log2_sd_bias_per_bit"] for kk, v in c4["C11"]["bit_slopes"].items() if not kk.startswith("0.03|")]
    put("f.c11.slope", f"{num(max(bsl), 2)} to {num(min(bsl), 2)}", "e4_comparisons.json C11.bit_slopes (field scales 0.1-1)")

    # ---- G: strong baselines
    q = f5["qat_cells"]
    put("g.qat.td.det", rng(100 * min(x["D_qat_det"] for x in q.values()), 100 * max(x["D_qat_det"] for x in q.values()), 0),
        "family_points_e5_tdmpc2.json qat_cells (points or %)")
    put("g.qat.td.shiter.better", sum(x["diff_ci"][0] > 0 for x in q.values()),
        "family_points_e5_tdmpc2.json qat_cells (D(QAT+det) - D(QAT+sh-iter), CI lower > 0)")
    put("g.qat.td.of", len(q), "family_points_e5_tdmpc2.json qat_cells")
    put("g.qat.td.worseB", sum(x["D_qat_det"] > f5["H3b_cells"][c]["D_Bdet"] for c, x in q.items()),
        "family_points_e5_tdmpc2.json (cells where QAT+det is worse than the best post-training B+det)")
    sel = load(os.path.join(R, "E5_tdmpc2", "e5_tdmpc2_summary.json"))["selection"]
    put("g.sel.td.pc", sum(x["best"] == "perchan" for x in sel.values()), "e5_tdmpc2_summary.json selection (per-channel chosen)")
    put("g.sel.td.n", len(sel), "e5_tdmpc2_summary.json selection")
    qn = load(os.path.join(R, "posthoc_tdmpc2", "posthoc_qatnone_summary.json"))["cells"]
    put("g.qat.td.none", rng(100 * min(x["qat_none"]["D"] for x in qn.values()), 100 * max(x["qat_none"]["D"] for x in qn.values()), 0),
        "main/results/posthoc_tdmpc2/posthoc_qatnone_summary.json")
    put("g.qat.td.none.n", sum(x["qat_none"]["D_ci"][0] > 0 for x in qn.values()), "posthoc_qatnone_summary.json (CI lower > 0)")
    qw = f5d["qat"]
    put("g.qat.dw.det", pci(qw["conditions"]["det"]["D"], *qw["conditions"]["det"]["D_ci"], 0, 100), "family_points_e5_dinowm.json qat")
    cc = qw["contrasts"]["D_qatdet_minus_D_E1det"]
    put("g.qat.dw.vs.e1", pci(-cc["diff"], -cc["ci"][1], -cc["ci"][0], 0, 100, sign=False), "family_points_e5_dinowm.json qat (E1 det - QAT det)")
    cc = qw["contrasts"]["D_qatdet_minus_D_qatshiter"]
    put("g.qat.dw.shiter", pci(-cc["diff"], -cc["ci"][1], -cc["ci"][0], 0, 100, sign=False), "family_points_e5_dinowm.json qat (QAT sh-iter - QAT det)")
    e5t = load(os.path.join(R, "E5_tdmpc2", "e5_tdmpc2_summary.json"))["weights"]
    w8 = [v["D"] for k, v in e5t.items() if k.endswith("|w8rtn")]
    w4 = [v["D"] for k, v in e5t.items() if k.endswith("|w4rtn")]
    put("g.w.td.w8", rng(100 * min(w8), 100 * max(w8), 1, keep_negzero=True), "main/results/E5_tdmpc2/e5_tdmpc2_summary.json weights")
    assert rng(min(w4), max(w4), 2) == "0.42--0.93"   # = e5.weights.w4 (fractions)
    wc = load(os.path.join(R, "E5_tdmpc2", "e5_tdmpc2_summary.json"))["weight_contrasts"]
    sr = {t: wc[f"{t}|w4rtn-w4sr4"]["Da_minus_Db"] for t in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk")}
    assert all(sr[t] > 0 for t in ("mw-push", "mw-pick-place", "cheetah-run")) and sr["walker-walk"] < 0
    for t in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk"):   # app_G: damage falls as K grows 1 -> 4
        assert e5t[f"{t}|w4sr1"]["D"] > e5t[f"{t}|w4sr2"]["D"] > e5t[f"{t}|w4sr4"]["D"], t
    put("g.w.td.sr4", words_list(num(100 * sr[t], 0, sign=True) for t in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk")),
        "e5_tdmpc2_summary.json weight_contrasts (D(w4rtn) - D(w4sr4); push, pick-place, cheetah, walker)")
    wd = f5d["weights"]["conditions"]
    put("g.w.dw.w8", pci(wd["w8rtn"]["D"], *wd["w8rtn"]["D_ci"], 0, 100), "family_points_e5_dinowm.json weights")
    put("g.w.dw.w4", num(100 * wd["w4rtn"]["D"], 0, sign=True), "family_points_e5_dinowm.json weights")
    put("g.w.dw.awq", num(100 * wd["w4awq"]["D"], 0, sign=True), "family_points_e5_dinowm.json weights")
    assert all(wd[f"w4sr{k}"]["D"] > wd["w4rtn"]["D"] for k in (1, 2, 4))   # app_G: SR worse than RTN on PushT
    put("g.w.dw.sr", words_list(num(100 * wd[f"w4sr{k}"]["D"], 0, sign=True) for k in (1, 2, 4)), "family_points_e5_dinowm.json weights")
    # bias vector: out-of-sample residual at 317M, MSE observation, PushT
    import glob
    oos, oos0 = [], []
    for f in sorted(glob.glob(os.path.join(TX, "MT_317m", "results", "raw", "mt.*", "diag", "bias_split_*.json"))):
        x = load(f)["test"]
        oos.append(x["b_fit5"]["r_mean"]["dyn1"])
        oos0.append(x["b0"]["r_mean"]["dyn1"])
    put("g.bias.mt.oos", rng(min(oos), max(oos), 3), "tmlr_ext/MT_317m/results/raw/mt.*/diag/bias_split_*.json (dyn1, held-out half)")
    put("g.bias.mt.oos0", rng(min(oos0), max(oos0), 2), "bias_split_*.json (dyn1, b = 0)")
    dr = [r for r in mtb if r["task"] == "mw-drawer-open" and r["layer"] == "dyn1"][0]
    put("g.bias.mt.drawer", f"{num(dr['mse_over_delta2_b0'], 2)} to {num(dr['mse_over_delta2_final'], 2)}",
        "e14_bias.csv (drawer-open k4 dyn1 MSE/Delta^2, b = 0 -> final b)")

    # ---- H: latency details
    ph = load(os.path.join(R, "posthoc_tdmpc2", "e7_posthoc_latency_summary.json"))
    meds = [v["median_pair_overhead"] for v in ph["conditions"].values()]
    put("h.ph.med", rng(100 * min(meds), 100 * max(meds), 1), "main/results/posthoc_tdmpc2/e7_posthoc_latency_summary.json")
    put("h.ph.ci0", sum(v["median_pair_overhead_ci"][0] <= 0 <= v["median_pair_overhead_ci"][1] for v in ph["conditions"].values()),
        "e7_posthoc_latency_summary.json (CI contains 0)")
    put("h.ph.n", len(ph["conditions"]), "e7_posthoc_latency_summary.json")
    put("h.ph.slow", num(100 * ph["state"]["share_blocks_slower_15pct"], 0), "e7_posthoc_latency_summary.json state")
    e8c = load(os.path.join(EXT, "E8_latency", "results", "analysis", "e8_summary.json"))["cells"]
    st = {k.split("|")[1]: v["ratios"] for k, v in e8c.items() if k.startswith("step|")}
    put("h.e8.step5", num(100 * (st["5"]["int-det/fp32"] - 1), 1, sign=True), "e8_summary.json cells.step|5|512")
    put("h.e8.step5.e7", num(100 * (st["5"]["e7-det/fp32"] - 1), 1, sign=True), "e8_summary.json cells.step|5|512")
    put("h.e8.step317", num(st["317"]["int-det/fp32"], 2), "e8_summary.json cells.step|317|512")
    put("h.e8.step317.min", num(st["317"]["int-det/minfast"], 2), "e8_summary.json cells.step|317|512")
    eg = {(k.split("|")[1], k.split("|")[2]): v["ratios"]["int-det/fp32"] for k, v in e8c.items() if k.startswith("eager|")}
    assert all(v["ratios"]["int-det/fp32"] < 1 for k, v in e8c.items() if k.startswith("graph|"))   # app_H: graph faster
    put("h.e8.eager5", num(eg[("5", "512")], 2), "e8_summary.json cells.eager|5|512 (int-det / FP32)")
    put("h.e8.eager.range", rng(min(eg.values()), max(eg.values()), 2), "e8_summary.json cells.eager|* (int-det / FP32)")

    # ---- K: compute and cost (report values: the accounting is in the logs, not in a result file)
    fa = open(os.path.join(A, "main", "reports", "FINAL_A.md"), encoding="utf-8").read()
    m = re.search(r"\*\*합\(이 보고 계산\)\*\* \| \*\*약 (\d+) h\*\* \| \*\*약 (\d+) GPU-h\*\* \| 두 계열 합 \*\*약 (\d+) GPU-시간\*\*", fa)
    put("k.gpuh.td", m.group(1), "main/reports/FINAL_A.md 8.2", "report")
    put("k.gpuh.dw", m.group(2), "main/reports/FINAL_A.md 8.2", "report")
    put("k.gpuh.main", m.group(3), "main/reports/FINAL_A.md 8.2", "report")
    m = re.search(r"CPU (\d+\.\d)시간, 사전 선언 격자 504 job", log_main)
    put("k.cpuh.e4", m.group(1), "main/LOG.md 2026-10-22T09:15Z", "report")
    log_rss = open(os.path.join(EXT, "LOG.md"), encoding="utf-8").read()
    m = re.search(r"원장 합계는 \$(\d+\.\d\d)", log_rss)
    put("k.cost.rss", m.group(1), "rss_ext/LOG.md 2026-10-31T21:50Z (ledger total, E8-E10 and E8c)", "report")
    m = re.search(r"dw 합 \$(\d+\.\d\d), 전체 \$(\d+\.\d\d)", log_tx)
    put("k.cost.tmlr", m.group(2), "tmlr_ext/LOG.md 2026-11-22T07:15:10Z (all pods deleted)", "report")
    m = re.search(r"지출 합 \$(\d+\.\d\d)\(td \$(\d+\.\d\d), mt \$(\d+\.\d\d), k \$(\d+\.\d\d), dw \$(\d+\.\d\d)\)", log_tx)
    put("k.cost.tmlr.main", m.group(1), "tmlr_ext/LOG.md 2026-11-22T01:23:58Z", "report")
    put("k.cost.tmlr.posthoc", num(float(load_cost_total(log_tx)) - float(m.group(1)), 2), "tmlr_ext/LOG.md (total - main runs)", "report")
    tdr = open(os.path.join(TX, "TD_tdmpc2", "REPORT.md"), encoding="utf-8").read()
    put("k.cost.x6", re.search(r"비용: td\.8 \$(\d+\.\d\d)", tdr).group(1), "tmlr_ext/TD_tdmpc2/REPORT.md 12.3 (X6, pod td.8)", "report")


def load_cost_total(log_tx):
    return re.search(r"dw 합 \$(\d+\.\d\d), 전체 \$(\d+\.\d\d)", log_tx).group(2)


def compute():
    ENTRIES.clear()
    earlier_results()
    tmlr_results()
    reanalysis()
    posthoc()
    x6_values()
    appendix_values()
    return list(ENTRIES)


def render(entries):
    lines = [
        "% numbers.tex -- generated by build/make_numbers.py from the A result files. Do not edit by hand.",
        "% \\nv{key} prints the value; an undefined key prints a red ?key? and a LaTeX warning.",
        "% kind 'report' = value copied from a stage report, not recomputed from a result file here.",
    ]
    for key, tex, source, kind in entries:
        lines.append(f"\\defnv{{{key}}}{{{tex}}}% {kind}: {source}")
    return "\n".join(lines) + "\n"


def main():
    entries = compute()
    with open(OUT, "w") as fh:
        fh.write(render(entries))
    n_rep = sum(e[3] == "report" for e in entries)
    print(f"wrote {OUT}: {len(entries)} values ({len(entries) - n_rep} recomputed, {n_rep} report-only)")


if __name__ == "__main__":
    main()
