"""E9 figure: per task, the det and sh-iter drops of NVFP4 and MXFP4 (95% CI) next to E1's uniform k = 4 det drop.
Usage: python plot_e9.py --summary <outdir>/e9_summary.json --out <outdir>/e9_drops.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
S = json.load(open(a.summary))
C = S["cells"]
tasks = []
for k in C:
    t = k.split(":")[0]
    if t not in tasks:
        tasks.append(t)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [8, 4]})
cols = {"nvfp4_det": "#c0392b", "nvfp4_shiter": "#f1948a", "mxfp4_det": "#1f4e79", "mxfp4_shiter": "#85c1e9",
        "e1_k4": "#7f7f7f"}
for ax, fam in zip(axes, ("mw", "dmc")):
    ts = [t for t in tasks if (t.startswith("mw-") == (fam == "mw"))]
    x = np.arange(len(ts))
    w = 0.16
    for j, (lab, key, ci_key) in enumerate([("NVFP4 det", "nvfp4:D_det", "nvfp4:D_det_ci"),
                                             ("NVFP4 sh-iter", "nvfp4:D_shiter", "nvfp4:D_shiter_ci"),
                                             ("MXFP4 det", "mxfp4:D_det", "mxfp4:D_det_ci"),
                                             ("MXFP4 sh-iter", "mxfp4:D_shiter", "mxfp4:D_shiter_ci")]):
        fmt, fld = key.split(":")
        v = np.array([100 * C[f"{t}:{fmt}"][fld] for t in ts])
        ci = np.array([[100 * c for c in C[f"{t}:{fmt}"][ci_key.split(":")[1]]] for t in ts])
        ax.bar(x + (j - 2) * w, v, w, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=2,
               color=cols[f"{fmt}_{'det' if 'det' in fld else 'shiter'}"], label=lab)
    e1 = [100 * C[f"{t}:nvfp4"].get("e1_k4_D_det", np.nan) if C[f"{t}:nvfp4"].get("e1_k4_D_det") is not None
          else np.nan for t in ts]
    ax.bar(x + 2 * w, e1, w, color=cols["e1_k4"], label="E1 uniform k=4 det (other seeds)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("mw-", "") for t in ts], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("success-rate drop (pp)" if fam == "mw" else "relative return drop (%)")
    ax.set_title("Meta-World" if fam == "mw" else "DMC")
axes[0].legend(fontsize=7, ncol=3, loc="upper left")
fig.suptitle("E9: block-scaled FP4 activations (TD-MPC2 5M, 90 pairs per cell, 95% paired-bootstrap CI)", fontsize=10)
fig.tight_layout()
fig.savefig(a.out, dpi=110)
print("saved", a.out)
