"""DESIGN_tmlr 0.7 check 2: unit checks of the new variants on real activations (pod only), plus the planner checks
the E16 CEM-like setting needs. Uses the smoke prep (calibration rows and bias vectors) of mw-push s1 (PT k = 3, PC
k = 2) and cheetah-run s1 (PT k = 5); the activations are the unquantized layer outputs of 2,000 calibration rows x 3
horizon steps, fed to the quantizer hook directly.
  1 det+BC     the 4th pass of the 3-sweep procedure (residual after the 3rd sweep): mean |residual_j| / delta_j < 0.01
               for every layer, every range kind and k of the prep jobs (reported; the E11/E12 cells need all)
  2 det+shift-fixed  the offset table does not change inside an episode (begin_step + forward calls), changes with the
               episode seed, is reproducible from it; element-wise error mean / variance equal det+shift's
               (128 tables vs 128 fresh draws; |mean diff| < 0.02 sd, |var ratio - 1| < 0.02)
  3 sh-iter+b  same generator seed: output - sh-iter output == b (fp32 rounding); error mean - sh-iter's error mean == b;
               error variance equal sh-iter's (rel. diff < 0.02)
  4 det+noise  added noise variance delta^2 / 12 (rel. diff < 0.02); det code bit-identical to det's (the noise is
               replayed from the seed and removed)
  5 global RNG the CPU / CUDA RNG states are unchanged by the quantizer for every new variant (hook calls), and one
               planning step (agent.act) leaves the planner's RNG in the same state as with none
  6 planner    TDMPC2._plan source lines; with temperature 0 the elite weights passed to the final Gumbel draw are all
               exactly 1/num_elites; num_pi_trajs 0 gives no 24-row calls
Usage: python test_variants.py --out <json> --prep_out <smoke out dir> --big <smoke big dir>
"""
import argparse
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import qlib as L  # noqa: E402
import runner as R  # noqa: E402
import tdjobs as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--prep_out", required=True)
ap.add_argument("--big", required=True)
a = ap.parse_args()
torch.set_num_threads(1)
RES = {"checks": {}, "info": {}}


def check(name, ok, **info):
    RES["checks"][name] = {"pass": bool(ok), **info}
    print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(info, default=float)[:400]}", flush=True)


def acts_of(agent, quant, task, ckpt, rows=2000):
    z0, aa = T.calib_rows(a.big, task, ckpt, [0, 1, 2, 3, 4], L.DEV)
    return T.record_acts(L, agent.model, quant, z0[:rows], aa[:rows])


def hook_out(quant, name, out):
    return quant._hook(name)(quant.mods[name], (None,), out)


def rng_states():
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()


def same_rng(s1, s2):
    return bool(torch.equal(s1[0], s2[0]) and torch.equal(s1[1], s2[1]))


with torch.no_grad():
    # ---------------------------------------------------------------- 1 det+BC residual (prep results)
    import glob
    import gzip
    worst = {}
    for f in sorted(glob.glob(f"{a.prep_out}/jobs/prep/prep_*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        RES["info"][f"ranges_{d['spec']['task']}"] = d["ranges"]
    for f in sorted(glob.glob(f"{a.prep_out}/jobs/bias/bias_*.json.gz")):
        d = json.load(gzip.open(f, "rt"))
        for rk, byk in d["bias"].items():
            for k, x in byk.items():
                worst[f"{d['spec']['task']}|{rk}|k{k}"] = x["resid_mean_abs_rel_delta_max_layer"]
        RES["info"][f"bias_rule_{d['spec']['task']}"] = d.get("rule")
    if worst:
        check("detbc_residual_lt_1pct_delta", max(worst.values()) < 0.01, worst_max=max(worst.values()),
              n=len(worst), worst_cells=sorted(worst.items(), key=lambda kv: -kv[1])[:4])
    else:
        print("PENDING detbc_residual_lt_1pct_delta (no bias job results yet: rule pending)", flush=True)
        RES["checks"]["detbc_residual_lt_1pct_delta"] = {"pass": None, "pending": True}

    for task, k, rk in (("mw-push", 3, "PT"), ("mw-push", 2, "PC"), ("cheetah-run", 5, "PT")):
        tag = f"{task}|{rk}k{k}"
        agent, cfg = L.build_agent(task, 1)
        quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
        X = acts_of(agent, quant, task, 1)
        quant.qrange = R.load_ranges(a.prep_out, task, 1, "e1" if rk == "PT" else "pc")
        if os.path.exists(R.bias_path(a.prep_out, task, 1)):
            quant.bias = R.load_bias(a.prep_out, task, 1, "e1" if rk == "PT" else "pc", k)
        else:       # mechanics checks of sh-iter+b only need some b: a fixed random vector
            g0 = torch.Generator(device=L.DEV)
            g0.manual_seed(5)
            quant.bias = {n: 0.1 * torch.randn(quant.mods[n].out_features, generator=g0, device=L.DEV)
                          for n in L.LAYERS}
        nl = (1 << k) - 1
        # ------------------------------------------------------------ 2 det+shift-fixed
        quant.set("detshiftfix", k, seed=11)
        T0 = {n: quant.table[n].clone() for n in L.LAYERS}
        for _ in range(5):
            quant.begin_step()
            for n in L.LAYERS:
                hook_out(quant, n, X[n][:512])
        const = all(torch.equal(T0[n], quant.table[n]) for n in L.LAYERS)
        quant.set("detshiftfix", k, seed=11)
        repro = all(torch.equal(T0[n], quant.table[n]) for n in L.LAYERS)
        quant.set("detshiftfix", k, seed=12)
        changes = all(not torch.equal(T0[n], quant.table[n]) for n in L.LAYERS)
        check(f"detshiftfix_table_{tag}", const and repro and changes, constant_in_episode=const,
              reproducible=repro, changes_across_episodes=changes, table_shape=list(T0["dyn0"].shape))
        stats = {}
        for n in L.LAYERS:
            lo, hi = quant.qrange[n]
            dl = (hi - lo) / nl
            acc = {}
            for v, base in (("detshiftfix", 1000), ("detshift", 2000)):
                s1 = s2 = 0.0
                cnt = 0
                for dd in range(128):
                    quant.set(v, k, seed=base + dd)
                    quant.hforce = 0 if v == "detshiftfix" else None
                    e = (hook_out(quant, n, X[n]) - X[n]).double()
                    s1 += float(e.sum())
                    s2 += float((e ** 2).sum())
                    cnt += e.numel()
                quant.off()
                m = s1 / cnt
                acc[v] = (m, s2 / cnt - m * m)
            (mf, vf), (ms, vs) = acc["detshiftfix"], acc["detshift"]
            dlm = float(dl.mean()) if torch.is_tensor(dl) else float(dl)
            stats[n] = {"mean_fixed": mf, "mean_shift": ms, "var_fixed": vf, "var_shift": vs,
                        "mean_diff_over_sd": abs(mf - ms) / np.sqrt(vs), "var_rel_diff": abs(vf / vs - 1),
                        "var_over_delta2_12": vs / (dlm ** 2 / 12)}
        ok = all(s["mean_diff_over_sd"] < 0.02 and s["var_rel_diff"] < 0.02 for s in stats.values())
        check(f"detshiftfix_error_moments_{tag}", ok, layers=stats)
        # ------------------------------------------------------------ 3 sh-iter+b
        st = {}
        for n in L.LAYERS:
            m1 = m2 = 0.0
            dmax = 0.0
            s1a = s1b = s2a = s2b = 0.0
            cnt = 0
            for dd in range(32):
                quant.set("shiter", k, seed=3000 + dd)
                q1 = hook_out(quant, n, X[n])
                quant.set("shiterb", k, seed=3000 + dd)
                q2 = hook_out(quant, n, X[n])
                dmax = max(dmax, float((q2 - q1 - quant.bias[n]).abs().max()))
                e1, e2 = (q1 - X[n]).double(), (q2 - X[n]).double()
                s1a += e1.sum(0)
                s1b += e2.sum(0)
                s2a += float((e1 ** 2).sum())
                s2b += float((e2 ** 2).sum())
                cnt += e1.shape[0]
            quant.off()
            mu1, mu2 = s1a / cnt, s1b / cnt                       # per-unit error means (W,)
            ntot = cnt * X[n].shape[1]
            va = s2a / ntot - float(mu1.mean()) ** 2
            vb_ = s2b / ntot - float(mu2.mean()) ** 2
            # variances about the per-unit means (the bias b only shifts the mean)
            va_c = s2a / ntot - float((mu1 ** 2).mean())
            vb_c = s2b / ntot - float((mu2 ** 2).mean())
            st[n] = {"max_abs(out_b - out - b)": dmax,
                     "max_abs(mean_err_b - mean_err - b)": float((mu2 - mu1 - quant.bias[n].double()).abs().max()),
                     "mean_abs_b": float(quant.bias[n].abs().mean()), "mean_err_shiter": float(mu1.mean()),
                     "var_centered_shiter": va_c, "var_centered_shiterb": vb_c, "var_rel_diff": abs(vb_c / va_c - 1),
                     "raw_var_shiter": va, "raw_var_shiterb": vb_}
        ok = all(s["max_abs(out_b - out - b)"] < 1e-4 and s["max_abs(mean_err_b - mean_err - b)"] < 1e-4
                 and s["var_rel_diff"] < 0.02 for s in st.values())
        check(f"shiterb_mean_b_var_{tag}", ok, layers=st)
        # ------------------------------------------------------------ 4 det+noise
        st = {}
        for n in L.LAYERS:
            lo, hi = quant.qrange[n]
            dl = (hi - lo) / nl
            quant.set("det", k)
            qd = hook_out(quant, n, X[n])
            code_d = torch.round((X[n].clamp(lo, hi) - lo) * (1.0 / dl)).clamp(0, nl)
            s2 = 0.0
            cnt = 0
            same_code = True
            maxdev = 0.0
            for dd in range(64):
                sd = 4000 + dd
                quant.set("detnoise", k, seed=sd)
                qn = hook_out(quant, n, X[n])
                g = torch.Generator(device=L.DEV)
                g.manual_seed(sd)
                nz = torch.rand((1, X[n].shape[-1]), generator=g, device=L.DEV, dtype=X[n].dtype) - 0.5
                code_n = torch.round((qn - lo) / dl - nz)
                same_code &= bool(torch.equal(code_n, code_d))
                maxdev = max(maxdev, float((qn - dl * nz - qd).abs().max()))
                added = ((qn - qd) / dl).double()        # = noise in grid units (up to fp rounding)
                s2 += float((added ** 2).sum())
                cnt += added.numel()
            quant.off()
            var_units = s2 / cnt                        # E[n^2], n uniform on [-1/2, 1/2): 1/12
            st[n] = {"added_var_over_1_12": var_units * 12, "det_code_identical": same_code,
                     "max_abs(out - delta n - det_out)": maxdev}
        ok = all(abs(s["added_var_over_1_12"] - 1) < 0.02 and s["det_code_identical"] for s in st.values())
        check(f"detnoise_var_detpart_{tag}", ok, layers=st)
        # ------------------------------------------------------------ 5 global RNG
        rs = {}
        for v in L.TMLR_VARIANTS:
            torch.manual_seed(123)
            s0 = rng_states()
            quant.set(v, k, seed=77)
            for n in L.LAYERS:
                hook_out(quant, n, X[n][:512])
            quant.off()
            rs[v] = same_rng(s0, rng_states())
        check(f"global_rng_hook_{tag}", all(rs.values()), per_variant=rs)
        env = L.make_env(cfg)
        obs = env.reset()
        post = {}
        for v in ("none",) + L.TMLR_VARIANTS:
            L.seed_all(5)
            quant.set(v, k, seed=99)
            quant.begin_step()
            agent.act(obs, t0=True, eval_mode=True)
            quant.off()
            post[v] = rng_states()
        same = {v: same_rng(post["none"], post[v]) for v in L.TMLR_VARIANTS}
        check(f"global_rng_planner_{tag}", all(same.values()), per_variant=same)
        quant.remove()
        del agent

    # ---------------------------------------------------------------- 6 planner source and temperature 0
    from tdmpc2 import TDMPC2
    src = inspect.getsource(TDMPC2._plan)
    keep = [ln.strip() for ln in src.splitlines() if any(s in ln for s in ("temperature", "score", "num_pi_trajs",
                                                                           "min_std", "gumbel", "elite"))]
    RES["info"]["plan_source_lines"] = keep
    print("TDMPC2._plan lines:\n  " + "\n  ".join(keep), flush=True)
    agent, cfg = L.build_agent("mw-push", 1)
    quant = L.Quantizer(agent.model, horizon=int(cfg.horizon))
    L.set_planner(agent, {"temperature": 0.0, "min_std": 0.01, "num_pi_trajs": 0})
    caught = []
    orig = L._tdmpc2_module.math.gumbel_softmax_sample

    def grab(p, *aa, **kw):
        caught.append(p.detach().clone())
        return orig(p, *aa, **kw)

    L._tdmpc2_module.math.gumbel_softmax_sample = grab
    calls = {}
    hs = [quant.mods["dyn0"].register_forward_hook(lambda m, i, o: calls.__setitem__(int(o.shape[0]),
                                                                                    calls.get(int(o.shape[0]), 0) + 1))]
    env = L.make_env(cfg)
    obs = env.reset()
    L.seed_all(7)
    for t in range(3):
        a_ = agent.act(obs, t0=(t == 0), eval_mode=True)
        obs, _, _, _ = env.step(a_)
    hs[0].remove()
    L._tdmpc2_module.math.gumbel_softmax_sample = orig
    ne = int(agent.cfg.num_elites)
    eq = all(bool(torch.all(p == p.reshape(-1)[0])) and float(p.reshape(-1)[0]) == 1.0 / ne for p in caught)
    check("cem_equal_elite_weights", eq and len(caught) == 3, n_draws=len(caught), num_elites=ne,
          weight=float(caught[0].reshape(-1)[0]) if caught else None, min_std=float(agent.cfg.min_std),
          dyn0_rows_calls=calls)
    check("cem_no_policy_prior_calls", 24 not in calls and set(calls) == {int(agent.cfg.num_samples)},
          dyn0_rows_calls=calls)
    L.set_planner(agent, None)
    dp = agent._default_planner
    check("planner_reset_default", all(getattr(agent.cfg, kk) == vv for kk, vv in dp.items()),
          default_planner={kk: (float(vv) if isinstance(vv, float) else vv) for kk, vv in dp.items()})

RES["all_pass"] = all(c["pass"] for c in RES["checks"].values() if c["pass"] is not None)
RES["pending"] = [k for k, c in RES["checks"].items() if c["pass"] is None]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(RES, open(a.out, "w"), indent=1, default=float)
print(f"TEST_VARIANTS_DONE all_pass={RES['all_pass']} n={len(RES['checks'])} pending={RES['pending']}", flush=True)
