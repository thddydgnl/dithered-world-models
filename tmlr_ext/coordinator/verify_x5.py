"""Coordinator recomputation of X5 (D6): k*(det+noise) per (task, range) from td.7 raw episodes, and the counts
against E12's k* (td_e12_cells.csv). Point estimates only (k* uses point D, as in E12)."""
import csv, glob, gzip, json, sys, collections
base = sys.argv[1]  # .../TD_tdmpc2/results
TAU, TOL = 0.10, 1e-9
S = collections.defaultdict(dict)  # (task, cell) -> {(ckpt, seed): success}
for f in glob.glob(f"{base}/raw/td.7/jobs/episodes/ev_*.json.gz"):
    d = json.load(gzip.open(f)); sp = d["spec"]
    if not d.get("ok", True) or sp.get("ptag") != "base":
        continue
    cell = "none" if sp["variant"] == "none" else f"{sp['rk']}k{sp['k']}_{sp['variant']}"
    for e in d["episodes"]:
        S[(sp["task"], cell)][(sp["ckpt"], e["seed"])] = float(e["success"])
tasks = sorted({t for t, _ in S})
e12 = collections.defaultdict(dict)
for r in csv.DictReader(open(f"{base}/analysis/td_e12_cells.csv")):
    e12[(r["task"], r["range"], r["rounding"])][int(r["k"])] = float(r["D"])
def kstar(kd):
    ks = sorted(kd)
    for k in ks:
        if all(kd[j] <= TAU + TOL for j in ks if j >= k):
            return k
    return 9
cnt = collections.Counter()
for rng in ("PT", "PC"):
    for t in tasks:
        nn = S[(t, "none")]
        kd = {}
        for (tt, cell), m in S.items():
            if tt == t and cell.startswith(f"{rng}k") and cell.endswith("_detnoise"):
                k = int(cell[3:cell.index("_")])
                keys = sorted(set(m) & set(nn))
                kd[k] = sum(nn[x] - m[x] for x in keys) / len(keys)
        kn = kstar(kd)
        ke = {v: kstar(e12[(t, rng, v)]) for v in ("det", "detbc", "shiter", "detshift")}
        g = min(ke["det"], ke["detbc"]) - kn
        a = kn <= ke["shiter"]; b = kn <= min(ke["shiter"], ke["detshift"])
        cnt[(rng, "le_shiter")] += a; cnt[(rng, "le_best")] += b; cnt[(rng, "G>=1")] += g >= 1
        print(f"{t:24s} {rng} k*(noise)={kn} E12 det/bc/shiter/shift={ke['det']}/{ke['detbc']}/{ke['shiter']}/{ke['detshift']} G_noise={g} "
              + " ".join(f"k{k}:{kd[k]:+.3f}" for k in sorted(kd)))
for rng in ("PT", "PC"):
    print(rng, "noise<=sh-iter", cnt[(rng, "le_shiter")], "noise<=min", cnt[(rng, "le_best")], "G_noise>=1", cnt[(rng, "G>=1")])
