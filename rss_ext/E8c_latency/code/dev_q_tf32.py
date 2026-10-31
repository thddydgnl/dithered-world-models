"""E8c development check (pod, after the smoke calibration, before the main run; numbers informal, descriptive):
why does TF32 in the rest of the planner save little at 317M N 8192? Eager timing (median of 5 calls after 2 warmup
calls) of the planner's rest modules at 317M with TF32 off / on: Q-ensemble (8 heads, return_type='avg', vmap),
policy prior, and a plain fp32 mm of the Q layer-1 size."""
import time, torch, numpy as np, e8lib as E
torch.backends.cuda.matmul.allow_tf32 = False
agent, cfg = E.build_agent(317)
m = agent.model
for N in (512, 8192):
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    z = m.encode(torch.randn(N, E.OBS_DIM, device="cuda", generator=g), None)
    a = torch.rand(N, E.ACT_DIM, device="cuda", generator=g) * 2 - 1
    x = torch.randn(N, 4096, device="cuda", generator=g); w = torch.randn(8 * 4096, 4096, device="cuda", generator=g)
    fns = {"Q_avg": lambda: m.Q(z, a, None, return_type="avg"), "Q_all": lambda: m._Qs(torch.cat([z, a], -1)),
           "pi": lambda: m._pi(z), "mm_8x4096": lambda: x @ w.t()}
    for name, fn in fns.items():
        res = {}
        for tf in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf
            with torch.no_grad():
                for _ in range(2):
                    fn()
                torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1e3 * (time.perf_counter() - t0))
            res[tf] = float(np.median(ts))
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"N {N} {name}: fp32 {res[False]:.2f} ms, tf32 {res[True]:.2f} ms, ratio {res[True] / res[False]:.3f}", flush=True)
with torch.no_grad(), torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    torch.backends.cuda.matmul.allow_tf32 = True
    N = 8192
    g = torch.Generator(device="cuda"); g.manual_seed(0)
    z = m.encode(torch.randn(N, E.OBS_DIM, device="cuda", generator=g), None)
    a = torch.rand(N, E.ACT_DIM, device="cuda", generator=g) * 2 - 1
    m.Q(z, a, None, return_type="avg"); torch.cuda.synchronize()
    torch.backends.cuda.matmul.allow_tf32 = False
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=70))
