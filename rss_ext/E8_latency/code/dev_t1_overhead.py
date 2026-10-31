"""E8 development check run on the pod (09:22Z and again after the dither change, before the smoke): det vs sh-iter
graph-WM rollout time at 5M and 317M, N 512, GEMM a and b (4 rounds x 300 / 60 replays), and the counter states.
First run: per-program Philox draw (about +5%); second run: u drawn once, one kernel ahead (about +1%)."""
import time, torch, numpy as np, e8lib as E
torch.backends.cuda.matmul.allow_tf32 = False
pool = E.EventPool(3000)
for size in (5, 317):
    agent, cfg = E.build_agent(size); m = agent.model; qr = E.calib_ranges(m)
    inp = E.wm_inputs(m, 512)
    gs = {}
    for impl in ("a", "b"):
        for p in ("int-det", "int-shiter"):
            wm = E.make_wm(m, p, qr, impl); wm.new_episode(5)
            gs[(impl, p)] = (wm, E.GraphRunner(lambda w=wm: E.wm_rollout(w, *inp)))
    wm = E.make_wm(m, "fp32"); gs[("-", "fp32")] = (wm, E.GraphRunner(lambda w=wm: E.wm_rollout(w, *inp)))
    res = {k: [] for k in gs}
    for rd in range(4):
        for k in gs:
            ms, _ = E.time_block(gs[k][1], 20, 300 if size == 5 else 60, pool)
            res[k].append(float(np.median(ms)))
    for k, v in res.items():
        print(size, k, ["%.3f" % x for x in v])
    for impl in ("a", "b"):
        w = gs[(impl, "int-shiter")][0]
        print(size, impl, "dyn state", w.dyn.state.tolist(), "rew state", w.rew.state.tolist())
