"""E8 development check run on the pod at 09:20Z (before the smoke): every precision path runs at 5M, and a short
graph / eager rollout timing (20 calls; numbers informal). First dither implementation (per-program Philox draw)."""
import time, torch, e8lib as E
t=time.time()
agent, cfg = E.build_agent(5)
m = agent.model
qr = E.calib_ranges(m)
print("ranges", qr, time.time()-t)
x = torch.cat([m.encode(torch.randn(512, 39, device="cuda"), None), torch.rand(512, 4, device="cuda")*2-1], -1)
for impl in ("a", "b"):
    for d in (False, True):
        mm = E.GenIntMLP(m._dynamics, qr["dyn0"], qr["dyn1"], d, 0, True, impl)
        q0, q1, o = mm._forward(x); torch.cuda.synchronize()
        print(impl, d, q0.shape, q0.dtype, int(q0.min()), int(q0.max()), q1.float().mean().item(), o.shape, o.sum().item(), mm.state.tolist())
        mr = E.GenIntMLP(m._reward, qr["rew0"], qr["rew1"], d, 4, False, impl)
        q0, q1, o = mr._forward(x); torch.cuda.synchronize()
        print(" rew", o.shape, mr.state.tolist())
import e7dev
for d in (False, True):
    me = e7dev.E7IntMLP(m._dynamics, qr["dyn0"], qr["dyn1"], d, 0, True)
    q0, q1, o = me._forward(x); torch.cuda.synchronize(); print("e7", d, o.shape, me.state.tolist())
inp = E.wm_inputs(m, 512)
for p in ("fp32","tf32","bf16","int-det","int-shiter","e7-det"):
    wm = E.make_wm(m, p, qr, "b")
    g = E.GraphRunner(lambda: E.wm_rollout(wm, *inp))
    pool = E.EventPool(100)
    ms,_ = E.time_block(g, 5, 20, pool)
    ms2,_ = E.time_block(lambda: E.wm_rollout(wm, *inp), 5, 20, pool)
    print(p, "graph %.3f eager %.3f" % (sorted(ms)[10], sorted(ms2)[10]))
print("total", time.time()-t)
