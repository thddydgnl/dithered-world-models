"""Coordinator descriptive item (DESIGN_tmlr §7): folded per-channel integer D minus E12 fake-quant per-channel D.
Independent samples (different pods): CI of the difference from two independent paired bootstraps."""
import csv, sys
import numpy as np
K, T = sys.argv[1], sys.argv[2]
NB = 10000
MW = {"mw-push", "mw-pick-place"}
def load_k():
    d = {}
    for r in csv.DictReader(open(K)):
        if r["prec"] in ("fp32", "intpc"):
            key = (r["task"], r["prec"], int(r["k"]), r["variant"])
            d.setdefault(key, {})[(int(r["ckpt"]), int(r["seed"]))] = (float(r["success"]), float(r["ret"]))
    return d
def load_t():
    d = {}
    for r in csv.DictReader(open(T)):
        if r["ptag"] != "base":
            continue
        d.setdefault((r["task"], r["cell"]), {})[(int(r["ckpt"]), int(r["seed"]))] = (float(r["success"]), float(r["ret"]))
    return d
def boot_D(task, none, v, seed):
    keys = sorted(set(none) & set(v))
    a = np.array([none[k] for k in keys]); b = np.array([v[k] for k in keys])
    idx = np.random.RandomState(seed).randint(0, len(keys), size=(NB, len(keys)))
    if task in MW:
        D = float(np.mean(a[:, 0] - b[:, 0])); Db = (a[idx, 0] - b[idx, 0]).mean(1)
    else:
        D = float(1 - b[:, 1].sum() / a[:, 1].sum()); Db = 1 - b[idx, 1].sum(1) / a[idx, 1].sum(1)
    return D, Db, len(keys)
kd, td = load_k(), load_t()
print("task k var | D_int_pc [CI] (n) | D_fq_pc [CI] (n) | diff [CI]")
for task in ("mw-push", "mw-pick-place", "cheetah-run", "walker-walk"):
    for k in (2, 3):
        for var in ("det", "shiter"):
            Di, Dib, ni = boot_D(task, kd[(task, "fp32", 0, "none")], kd[(task, "intpc", k, var)], 0)
            tn = td.get((task, "none")) or td.get((task, "none_it6")) or td.get((task, "E11_none"))
            if tn is None:
                cands = [c for (t, c) in td if t == task and "none" in c]
                tn = td[(task, cands[0])]
            tv = td.get((task, f"PCk{k}_{var}"))
            if tv is None:
                print(task, k, var, "no fake-quant cell"); continue
            Df, Dfb, nf = boot_D(task, tn, tv, 1)
            diff = Dib - Dfb
            print(f"{task:13s} k{k} {var:6s} | {Di:+.3f} [{np.percentile(Dib,2.5):+.3f},{np.percentile(Dib,97.5):+.3f}] ({ni}) | "
                  f"{Df:+.3f} [{np.percentile(Dfb,2.5):+.3f},{np.percentile(Dfb,97.5):+.3f}] ({nf}) | {Di-Df:+.3f} [{np.percentile(diff,2.5):+.3f},{np.percentile(diff,97.5):+.3f}]")
