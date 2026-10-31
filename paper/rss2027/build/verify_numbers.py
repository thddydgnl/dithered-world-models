# Recompute every number quoted in main.tex from the A result files and check that the text quotes it exactly.
# Numbers that only exist in a report (theory simulation, E7 kernel) are listed as "report" checks with their source.
# Usage: python3 verify_numbers.py   (exit code 1 if any check fails)
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(A, "main", "results")
EXT = os.path.join(A, "rss_ext")
TEX = open(os.path.join(HERE, "..", "main.tex")).read()

checks = []  # (label, text that must appear in main.tex, computed value as shown, source)


def check(label, text, computed, source):
    checks.append((label, text, computed, source))


def f(x, nd=0):
    return f"{x:.{nd}f}"


# ---- E1 TD-MPC2 5M -------------------------------------------------------------------------------------------
by = {}
for r in csv.DictReader(open(os.path.join(RES, "E1_tdmpc2", "e1_tdmpc2_cells.csv"))):
    by.setdefault((r["task"], int(r["k"]), r["family"]), {})[r["variant"]] = r
aff = {key: v for key, v in by.items() if v["det"]["confirmatory"] == "True" and v["det"]["holm_real"] == "True"}
mw = {key: v for key, v in aff.items() if key[2] == "mw"}
dmc = {key: v for key, v in aff.items() if key[2] == "dmc"}
red_mw = [float(v["shiter"]["red"]) for v in mw.values()]
red_dmc = [float(v["shiter"]["red"]) for v in dmc.values()]
check("MW affected cells", "all 19 affected Meta-World cells", f"{len(mw)} affected, {sum(r >= 0.5 for r in red_mw)} with red>=0.5", "E1_tdmpc2_cells.csv")
check("MW median red", "median 95\\%", f(100 * statistics.median(red_mw)), "E1_tdmpc2_cells.csv")
wk = min(mw.items(), key=lambda kv: float(kv[1]["shiter"]["red"]))
check("weakest MW cell", "keeps 47\\% of the damage (window-open, 4 bits)",
      f"{wk[0][0]} k={wk[0][1]} red={float(wk[1]['shiter']['red']):.3f}", "E1_tdmpc2_cells.csv")
check("DMC red>=0.5", "in only 5 of 12 cells", f"{sum(r >= 0.5 for r in red_dmc)}/{len(red_dmc)}", "E1_tdmpc2_cells.csv")
dmax = max(by.items(), key=lambda kv: float(kv[1]["det"]["D"]) if kv[1]["det"]["confirmatory"] == "True" else -1)
check("max det damage", "up to 100 points (mw-button-press at 2 bits)",
      f"{dmax[0][0]} k={dmax[0][1]} D={100 * float(dmax[1]['det']['D']):.1f}", "E1_tdmpc2_cells.csv")
pp = by[("mw-pick-place", 3, "mw")]
check("pick-place example errors", "the error is 0.22 for \\dtm and 0.39 for \\shiter",
      f"{float(pp['det']['pt_err1h']):.2f} / {float(pp['shiter']['pt_err1h']):.2f}", "E1_tdmpc2_cells.csv")
check("pick-place example damage", "yet \\dtm loses 37.8 points and \\shiter 7.8",
      f"{100 * float(pp['det']['D']):.1f} / {100 * float(pp['shiter']['D']):.1f}", "E1_tdmpc2_cells.csv")
wa = by[("walker-walk", 3, "dmc")]
check("walker k=3 harmful", "78.9\\% of return lost vs.\\ 56.9\\% under \\dtm",
      f"sh-iter {100 * float(wa['shiter']['D']):.1f} / det {100 * float(wa['det']['D']):.1f}", "E1_tdmpc2_cells.csv")
h1_td = sum(float(v["det"]["pt_err1h"]) <= float(v["shiter"]["pt_err1h"]) for v in aff.values())
rew_mw = sum(float(v["det"]["pt_errr"]) > float(v["shiter"]["pt_errr"]) for v in mw.values())
check("reward error MW", "larger under \\dtm in all 19 affected Meta-World cells", f"{rew_mw}/{len(mw)}", "E1_tdmpc2_cells.csv (pt_errr) [post hoc]")
ch4 = by[("cheetah-run", 4, "dmc")]
check("uniform 4-bit cheetah", "against 71.6\\% on cheetah-run with a uniform 4-bit grid", f(100 * float(ch4["det"]["D"]), 1), "E1_tdmpc2_cells.csv")

means = {v: 100 * statistics.mean(float(x[v]["D"]) for x in mw.values()) for v in ("det", "ind", "shiter", "shstep", "shep", "detshift")}
check("variant mean drops (MW)", "the mean drop is 50.1 points under \\shep, 34.9 under \\shstep, 29.2 under \\ind and 4.0 under \\shiter, against 44.6 under \\dtm",
      ", ".join(f"{k} {v:.1f}" for k, v in means.items()), "E1_tdmpc2_cells.csv")
check("det+shift mean drop (MW)", "also loses only 4.0", f"{means['detshift']:.1f}", "E1_tdmpc2_cells.csv")

# ---- E1 DINO-WM ---------------------------------------------------------------------------------------------
dw = {}
for r in csv.DictReader(open(os.path.join(RES, "E1", "e1_dinowm_cells.csv"))):
    if r["role"] == "confirmatory":
        dw.setdefault((r["task"], int(r["k"])), {})[r["var"]] = r
dw_aff = {key: v for key, v in dw.items() if v["det"]["drop_real_holm"] == "True"}
dw_red = sum(float(v["shiter"]["red"]) >= 0.5 for v in dw_aff.values())
check("DINO-WM red>=0.5", "in 4 of 5 DINO-WM cells", f"{dw_red}/{len(dw_aff)}", "e1_dinowm_cells.csv")
h1_dw = sum(float(v["det"]["rel_err1"]) <= float(v["shiter"]["rel_err1"]) for v in dw_aff.values())
check("H1 count 5M", "In 29 of the 36 affected 5M cells", f"{h1_td + h1_dw}/{len(aff) + len(dw_aff)}", "E1 cells (pt_err1h, rel_err1)")
p5 = dw[("pusht", 5)]
check("PushT k=5 det damage", "PushT at 5 bits, where \\dtm's damage is only 5 points", f(100 * float(p5["det"]["D"])), "e1_dinowm_cells.csv")
p4 = dw[("pusht", 4)]
check("PushT 5-step error", "five steps: 0.25 for \\dtm vs.\\ 0.04 for \\shiter",
      f"{float(p4['det']['rel_err5']):.2f} / {float(p4['shiter']['rel_err5']):.2f}", "e1_dinowm_cells.csv")

# ---- E10 317M -----------------------------------------------------------------------------------------------
e10 = [r for r in csv.DictReader(open(os.path.join(EXT, "E10_mt317m", "results", "analysis", "e10_eval.csv")))]
e10a = [r for r in e10 if r["drop_real"] == "True"]
ratios = [float(r["pt_err1h_det"]) / float(r["pt_err1h_sh"]) for r in e10a]
check("317M affected and red>=0.5", "all 8 affected tasks of a 317M",
      f"{len(e10a)} affected, {sum(float(r['red_point']) >= 0.5 for r in e10a)} with red>=0.5", "e10_eval.csv")
check("317M latent error ratio", "0.55--0.69 times that of \\shiter", f"{min(ratios):.2f}--{max(ratios):.2f}", "e10_eval.csv")
check("317M det damage range", "loses 12--99 points",
      f"{100 * min(float(r['D_det']) for r in e10a):.0f}--{100 * max(float(r['D_det']) for r in e10a):.0f}", "e10_eval.csv")
check("317M red range", "removes 75--104\\% of the damage",
      f"{100 * min(float(r['red_point']) for r in e10a):.0f}--{100 * max(float(r['red_point']) for r in e10a):.0f}", "e10_eval.csv")
peg = [r for r in e10 if r["task"] == "mw-peg-insert-side"][0]
check("317M peg example", "98.9 points lost under \\dtm, 4.4 under \\shiter",
      f"{100 * float(peg['D_det']):.1f} / {100 * float(peg['D_sh']):.1f}", "e10_eval.csv")
upper = [float(r["Dsh_minus_Ddet_ci"].strip("[]").replace(";", ",").split(",")[1]) for r in e10a]
check("317M sh-iter significantly less damaging", "lies below zero in every task", f"max CI upper {max(upper):.3f}", "e10_eval.csv")
rew317 = sum(float(r["pt_errr_det"]) > float(r["pt_errr_sh"]) for r in e10a)
check("317M reward error", "all 8 tasks at 317M", f"{rew317}/{len(e10a)}", "e10_eval.csv [post hoc]")
check("H1 incl. 317M", "in 37 of 44 affected cells", f"{h1_td + h1_dw + sum(x <= 1 for x in ratios)}/{len(aff) + len(dw_aff) + len(e10a)}", "E1 + E10")

# ---- E2 / E1 DINO-WM mechanism --------------------------------------------------------------------------------
lb = json.load(open(os.path.join(RES, "E2_tdmpc2", "e2_tdmpc2_summary.json")))["layer_b"]
tasks = ("mw-push", "mw-pick-place", "cheetah-run")
pt = json.load(open(os.path.join(RES, "E1", "e1_dinowm_v52.json")))["b2_H2a_prime_pusht_k4"]
r_shiter = [lb[t]["b2_H2a_prime"]["r_shiter"]["r"] for t in tasks] + [pt["r_shiter"]]
check("sh-iter drift ratios", "(0.10, 0.24, 0.03 and $-0.01$ of it)", ", ".join(f"{x:.2f}" for x in r_shiter), "E2 summary, e1_dinowm_v52")
r_ds = [lb[t]["b2_H2a_prime"]["r_detshift"]["r"] for t in tasks] + [pt["r_detshift"]]
check("det+shift drift max", "keeps at most 15\\%", f"max {max(r_ds):.3f}", "E2 summary, e1_dinowm_v52")
r_pc = [lb[t]["b2_H2a_prime"][v]["r"] for t in tasks for v in ("r_shstep", "r_shep")]
check("positive controls", "brings back 70--130\\% of \\dtm's drift", f"{100 * min(r_pc):.0f}--{100 * max(r_pc):.0f}", "E2 summary")
check("PushT sh-plan", "(ratio 0.01)", f"{pt['positive_control_r_shplan']:.3f}", "e1_dinowm_v52")
e2 = {(r["task"], r["setting"], r["variant"]): r for r in csv.DictReader(open(os.path.join(RES, "E2_tdmpc2", "e2_tdmpc2_cells.csv")))}
d12 = {v: [100 * float(e2[(t, "it12", v)]["D"]) for t in tasks] for v in ("det", "shstep", "shiter")}
check("12 iterations", "its damage is 0--11\\% while \\dtm's stays at 43--47\\% and \\shstep's at 18--47\\%",
      "; ".join(f"{v} {min(x):.1f}-{max(x):.1f}" for v, x in d12.items()), "e2_tdmpc2_cells.csv")
d1 = {v: [100 * float(e2[(t, "it1", v)]["D"]) for t in tasks] for v in ("det", "shiter")}
check("1 iteration", "With one MPPI iteration \\shiter is no better than \\dtm",
      " ".join(f"{t}: det {a:.1f} sh-iter {b:.1f}" for t, a, b in zip(tasks, d1["det"], d1["shiter"])), "e2_tdmpc2_cells.csv")

# ---- E3 DMC explanation ---------------------------------------------------------------------------------------
dx = json.load(open(os.path.join(RES, "E3_tdmpc2", "e3_tdmpc2_final.json")))["DMC_explanation"]
s_mw = [r["S_det"] for r in dx["rows"] if r["task"].startswith("mw-")]
s_dmc = [r["S_det"] for r in dx["rows"] if not r["task"].startswith("mw-")]
check("persistent share", "is 0.33--0.52 on DMC and 0.54--0.73 on Meta-World",
      f"DMC {min(s_dmc):.2f}-{max(s_dmc):.2f}, MW {min(s_mw):.2f}-{max(s_mw):.2f}", "e3_tdmpc2_final.json")
check("DMC rule Spearman", "(Spearman 0.83, new seeds)", f"{dx['spearman_S_red']:.3f}", "e3_tdmpc2_final.json")

# ---- E8 latency ------------------------------------------------------------------------------------------------
cells = [r for r in csv.DictReader(open(os.path.join(EXT, "E8_latency", "results", "analysis", "e8_cells.csv")))]
g = [r for r in cells if r["mode"] == "graph"]
sp = [float(r["T_fp32_ms"]) / float(r["T_int_det_ms"]) for r in g]
check("graph speed-up vs FP32", "1.9--3.3$\\times$ faster than FP32", f"{min(sp):.2f}--{max(sp):.2f} ({sum(x >= 1.25 for x in sp)}/12 at ratio<=0.8)", "e8_cells.csv")
fast = [1 / float(r["int_det_over_minfast"]) for r in g]
passing = [x for x in fast if x >= 1 / 0.9 - 1e-9]
check("graph speed-up vs fastest float", "1.19--1.48$\\times$ faster than the faster of TF32 and BF16 in 10 of 12",
      f"{min(passing):.2f}--{max(passing):.2f} in {len(passing)}/12", "e8_cells.csv")
ov = [100 * float(r["overhead_median"]) for r in g]
check("dither overhead", "$-0.2$ to $+2.1\\%$", f"{min(ov):.2f} to {max(ov):.2f}", "e8_cells.csv")
st = [r for r in cells if r["mode"] == "step" and r["size"] == "317"][0]
check("eager full step 317M", "1.35$\\times$ over FP32 and 1.05$\\times$ over the fastest float format",
      f"{1 / float(st['int_det_over_fp32']):.2f} / {1 / float(st['int_det_over_minfast']):.2f}", "e8_cells.csv (step)")

# ---- E8b, E9 ---------------------------------------------------------------------------------------------------
e9 = json.load(open(os.path.join(EXT, "E9_fp4", "results", "e9_summary.json")))
check("E9 summary keys", "NVFP4 fails F1 (2/12), MXFP4 holds (4/12)", ", ".join(sorted(e9.keys()))[:120], "e9_summary.json (inspect)")
check("NVFP4 cells", "(cheetah-run 9.9\\%, walker 1.7\\%)", "see E9 REPORT §1", "E9_fp4/REPORT.md")
check("MXFP4 cells", "(cheetah-run 31.3\\%, window-open 10.0 points)", "see E9 REPORT §1", "E9_fp4/REPORT.md")
check("F3 ratio", "0.48--0.63 times that of \\shiter", "see LOG 11:20Z (verify_e9.py)", "rss_ext/LOG.md")
check("E8b NVFP4", "up to 5.3$\\times$ faster than BF16", "L1 M8192 0.19 of BF16", "E8_latency/REPORT.md §7.1")

# ---- report-only numbers ---------------------------------------------------------------------------------------
check("theory simulation", "922k simulated MPPI and CEM runs (fitted slopes 0.92--1.05)", "report", "main/reports/E4 (JKAIA wording)")
check("E7 eager slowdown", "it was 18\\% slower than FP32", "report: 17.5-18.2%", "FINAL_A §4.1(7)")
check("E7 overhead", "holds (max 6.9\\%)", "report", "FINAL_A §3 H5")
check("H3b", "(10 of 19 cells)", "report", "FINAL_A §3 H3b")
check("Wall exploratory", "(+37 points)", "report", "FINAL_A §4.1(4), E3D §6")

bad = 0
for label, text, computed, source in checks:
    ok = text in TEX
    bad += not ok
    print(f"[{'OK ' if ok else 'MISSING'}] {label}: text \"{text[:60]}\" | computed {computed} | {source}")
print(f"\n{len(checks) - bad}/{len(checks)} quoted strings found; compare 'computed' with the text by eye for report checks.")
sys.exit(1 if bad else 0)
