"""E4 (paper A): simulation library. Quadratic objectives, structured cost-error models and faithful re-implementations
of the two planners of the main experiments. Runs on the server CPU only (numpy, one thread per process).

Planners (read on the server 2026-10-22, see worklogs/E4_theory.md):
  mppi_plan  TD-MPC2 TDMPC2._plan (tdmpc2 @ 332f19e): num_pi_trajs policy-prior proposals drawn once per plan and
             re-evaluated in every iteration; num_samples - num_pi_trajs samples mean + std * randn clamped to [-1, 1];
             top-num_elites by value V = -Jhat; score = exp(temperature (V - max V)) / sum; mean, std = score-weighted
             elite mean / std, std clamped to [min_std, max_std]; std starts at max_std, mean at 0 (t0) or warm start
             (prev[1:], 0); executed plan = elite drawn from Categorical(score) of the last iteration
             (gumbel_softmax_sample); eval_mode (no action noise); the next warm start uses the final mean.
  cem_plan   DINO-WM CEMPlanner.plan (dino_wm @ 0a9492f; batched in main/code/dinowm/qlib.run_cem): mu = 0, sigma =
             var_scale = 1; each step N samples randn * sigma + mu with sample 0 = mu; top-k lowest cost; mu, sigma =
             mean, unbiased std of the top-k; open loop; returns mu.
OEI exactly as the experiments define it:
  MPPI (DESIGN_main 0.4): e(executed elite, the realized error of the last-iteration evaluation) - mean e of the
       iteration-0 sampled candidates (policy-prior proposals excluded).
  CEM (DESIGN_main 5, oei_ref_mf): e(final elite mean mu; fresh-noise models re-evaluated with one independent draw,
       plan-fixed / deterministic models with their own error) - mean e of iteration-0 candidates 1..N-1.
Decomposition (coordinator request 2026-10-22T06:58Z): OEI = sel_last + drift,
  sel_last = e(executed) - mean e(last-iteration population), drift = mean e(last population) - mean e(iteration 0);
  both are recorded with the same-draw error of the executed plan and with an independent re-evaluation.
"""
import math
import numpy as np

import e4_formulas as F

# ================================================================================================ objectives
class Quadratic:
    """J(a) = 1/2 (a - a*)^T H (a - a*), a in R^d. regime 'sep': eigenvalues log-uniform in [1, 10] (well separated);
    'tie': half of the eigen-directions nearly flat ([0.01, 0.1]) so candidates that differ along them are near-ties at
    every population scale. a* ~ U[-0.5, 0.5]^d (inside MPPI's action box, so J* = 0)."""

    def __init__(self, rng, d, regime, h_scale=1.0, astar_range=0.5):
        self.d = int(d)
        if regime == 'sep':
            lam = np.exp(rng.uniform(0.0, math.log(10.0), d))
        elif regime == 'tie':
            nf = max(1, d // 2)
            lam = np.concatenate([np.exp(rng.uniform(0.0, math.log(10.0), d - nf)),
                                  np.exp(rng.uniform(math.log(0.01), math.log(0.1), nf))])
        else:
            raise ValueError(regime)
        Q, R = np.linalg.qr(rng.standard_normal((d, d)))
        Q = Q * np.sign(np.diag(R))
        self.lam = h_scale * lam
        self.Q = Q
        self.H = (Q * self.lam) @ Q.T
        self.Hinv = (Q / self.lam) @ Q.T
        self.a_star = rng.uniform(-astar_range, astar_range, d)

    def J(self, A):
        D = A - self.a_star
        return 0.5 * np.einsum('nd,nd->n', D @ self.H, D)


# ================================================================================================ error models
class RFF:
    """Stationary random field with unit variance and squared-exponential covariance of length ell (random Fourier
    features, omega ~ N(0, I/ell^2))."""

    def __init__(self, rng, d, ell, F=128):
        self.W = rng.standard_normal((F, d)) / ell
        self.p = rng.uniform(0.0, 2 * math.pi, F)
        self.s = math.sqrt(2.0 / F)

    def __call__(self, A):
        return self.s * np.cos(A @ self.W.T + self.p).sum(1)

    def smoothed(self, mu, sig):
        """value, gradient, Hessian of E_z[f(mu + sig * z)] (sig: per-coordinate std)."""
        att = np.exp(-0.5 * (self.W ** 2) @ (np.asarray(sig, float) ** 2))
        arg = self.W @ mu + self.p
        c, s = att * np.cos(arg), att * np.sin(arg)
        val = self.s * c.sum()
        grad = -self.s * (s @ self.W)
        hess = -self.s * (self.W.T * c) @ self.W
        return val, grad, hess


class ErrModel:
    """Interface. begin_plan(rng) once per plan (plan-level tables), begin_iter() before every model evaluation
    (fresh draws), __call__(A) errors of the candidates under the current draw, same_draw(A, rng_aux) errors at new
    points under the current draw (independent values for per-candidate noise), redraw(A, rng) errors under an
    independent draw (= the persistent error for persistent models), mean_field(A, rng, n) E over draws."""
    persistent = False
    name = 'base'

    def begin_plan(self, rng):
        self.rng = rng

    def begin_iter(self):
        pass

    def same_draw(self, A, rng_aux):
        return self(A)

    def redraw(self, A, rng):
        return self(A)

    def mean_field(self, A, rng, n=64):
        if self.persistent:
            return self(A)
        acc = np.zeros(len(A))
        for _ in range(n):
            acc += self.redraw(A, rng)
        return acc / n


class NoErr(ErrModel):
    persistent = True
    name = 'none'

    def __call__(self, A):
        return np.zeros(len(A))


class PersistentField(ErrModel):
    """(1) persistent field e(a) = amp * b(a), b a fixed random field (det analogue)."""
    persistent = True
    name = 'M1'

    def __init__(self, field, amp):
        self.f, self.amp = field, float(amp)

    def __call__(self, A):
        return self.amp * self.f(A)


class WhiteNoise(ErrModel):
    """(2) fresh independent noise: i.i.d. N(0, amp^2) per candidate and per evaluation (ind analogue)."""
    name = 'M2'

    def __init__(self, amp):
        self.amp = float(amp)

    def __call__(self, A):
        return self.amp * self.rng.standard_normal(len(A))

    def same_draw(self, A, rng_aux):
        return self.amp * rng_aux.standard_normal(len(A))

    def redraw(self, A, rng):
        return self.amp * rng.standard_normal(len(A))


class FreshField(ErrModel):
    """(3) fresh shared noise: per evaluation a new random field F_t (length ell) plus a common shock xi_t shared by all
    candidates: e = amp (sqrt(rho_c) xi_t + sqrt(1 - rho_c) F_t(a)) (sh-iter analogue)."""
    name = 'M3'

    def __init__(self, amp, ell, rho_c, d, F=128):
        self.amp, self.ell, self.rho_c, self.d, self.F = float(amp), float(ell), float(rho_c), int(d), int(F)

    def _draw(self, rng):
        return RFF(rng, self.d, self.ell, self.F), float(rng.standard_normal())

    def begin_iter(self):
        self.cur, self.xi = self._draw(self.rng)

    def _eval(self, A, f, xi):
        return self.amp * (math.sqrt(self.rho_c) * xi + math.sqrt(1.0 - self.rho_c) * f(A))

    def __call__(self, A):
        return self._eval(A, self.cur, self.xi)

    def redraw(self, A, rng):
        f, xi = self._draw(rng)
        return self._eval(A, f, xi)


# ------------------------------------------------------------------------------------------------ rounding toy
class RoundToy(ErrModel):
    """Toy quantized model ('quantization with offsets', report section 2.5). K = n_tok x n_unit quantized entries with
    pre-quantization values in grid units
        y_{tau,j}(a) = nu * w_j . (a - c0) / s0 + beta_j + eta_{tau,j}
    and cost error e(a) = amp / K * sum_{tau,j} G_{tau,j}(a) r_{tau,j}(a),
        G_{tau,j}(a) = 1 + gamma * s_j(a) + gamma_tok * t_{tau,j}(a)    (linearized downstream sensitivity),
        s_j(a) = (sqrt(kappa_s) vbar + sqrt(1 - kappa_s) v_j) . (a - c0) / s0,  t_{tau,j}(a) = v_{tau,j} . (a - c0)/s0.
    Rounding error r (grid units) by variant:
        det       round(y) - y                                    (no offset; deterministic)
        ind       floor(y + u) - y, u fresh per candidate, entry and evaluation
        shiter    floor(y + u) - y, u per entry shared by the candidates, fresh per evaluation
        fixed     floor(y + u_g) - y, one u per sharing group g, drawn once per plan (sh-step / sh-ep)
        detshift  round(y - o) + o - y, o ~ U[-1/2, 1/2) per entry shared by the candidates, fresh per evaluation
    Sharing groups of 'fixed': tok_block consecutive tokens x unit_block consecutive units share one offset;
    n_eff = (n_tok / tok_block) (n_unit / unit_block)."""
    name = 'R'

    def __init__(self, struct, variant, amp, tok_block=1, unit_block=1):
        self.st, self.var, self.amp = struct, variant, float(amp)
        self.persistent = variant in ('det', 'fixed')
        T, U = struct['n_tok'], struct['n_unit']
        self.T, self.U, self.K = T, U, T * U
        self.tb, self.ub = int(tok_block), int(unit_block)
        assert T % self.tb == 0 and U % self.ub == 0
        self.n_eff = (T // self.tb) * (U // self.ub)
        self.name = f'R-{variant}'

    # -- structure
    @staticmethod
    def make_struct(rng, d, n_tok=1, n_unit=512, nu=3.0, gamma=1.0, gamma_tok=0.0, kappa_s=0.0, eta_spread=0.0,
                    phase_conc=1.0, kappa_w=0.0, c0=None, s0=1.0):
        w = rng.standard_normal((n_unit, d))
        if kappa_w > 0:
            wbar = rng.standard_normal(d)
            w = math.sqrt(kappa_w) * wbar[None] / np.linalg.norm(wbar) * math.sqrt(d) + math.sqrt(1 - kappa_w) * w
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        v = rng.standard_normal((n_unit, d)) / math.sqrt(d)
        if kappa_s > 0:
            vbar = rng.standard_normal(d) / math.sqrt(d)
            v = math.sqrt(kappa_s) * vbar[None] + math.sqrt(1 - kappa_s) * v
        vt = rng.standard_normal((n_tok, n_unit, d)) / math.sqrt(d) if gamma_tok > 0 else None
        return dict(d=d, n_tok=n_tok, n_unit=n_unit, nu=nu, gamma=gamma, gamma_tok=gamma_tok,
                    w=w, beta=rng.uniform(0.0, phase_conc, n_unit),
                    eta=rng.uniform(0.0, eta_spread, (n_tok, n_unit)) if eta_spread > 0 else np.zeros((n_tok, n_unit)),
                    v=v, vt=vt, c0=np.zeros(d) if c0 is None else c0, s0=s0)

    def _yG(self, A):
        st = self.st
        Ap = (A - st['c0']) / st['s0']
        yu = st['nu'] * (Ap @ st['w'].T) + st['beta'][None]                         # (N, U)
        y = yu[:, None, :] + st['eta'][None]                                         # (N, T, U)
        G = 1.0 + st['gamma'] * (Ap @ st['v'].T)[:, None, :]                         # (N, 1, U)
        if st['vt'] is not None:
            G = G + st['gamma_tok'] * np.einsum('nd,tud->ntu', Ap, st['vt'])
        return y, G

    def _offsets(self, rng, kind):
        if kind == 'shared':
            return rng.random((1, self.T, self.U))
        if kind == 'group':
            g = rng.random((self.T // self.tb, self.U // self.ub))
            return np.repeat(np.repeat(g, self.tb, 0), self.ub, 1)[None]
        raise ValueError(kind)

    def _r(self, y, u):
        v = self.var
        if v == 'det':
            return np.round(y) - y
        if v in ('ind', 'shiter', 'fixed'):
            return np.floor(y + u) - y
        if v == 'detshift':
            return np.round(y - u) + u - y
        raise ValueError(v)

    def _e(self, A, u):
        y, G = self._yG(A)
        if self.var == 'ind' and u is None:
            u = self.rng.random(y.shape)
        return self.amp / self.K * (G * self._r(y, u)).sum((1, 2))

    def begin_plan(self, rng):
        self.rng = rng
        self.u = self._offsets(rng, 'group') if self.var == 'fixed' else None

    def begin_iter(self):
        if self.var == 'shiter':
            self.u = self._offsets(self.rng, 'shared')
        elif self.var == 'detshift':
            self.u = self._offsets(self.rng, 'shared') - 0.5

    def __call__(self, A):
        return self._e(A, None if self.var in ('det', 'ind') else self.u)

    def same_draw(self, A, rng_aux):
        if self.var == 'ind':
            y, G = self._yG(A)
            return self.amp / self.K * (G * self._r(y, rng_aux.random(y.shape))).sum((1, 2))
        return self(A)

    def redraw(self, A, rng):
        v = self.var
        if v in ('det', 'fixed'):
            return self(A)
        y, G = self._yG(A)
        if v == 'ind':
            u = rng.random(y.shape)
        elif v == 'shiter':
            u = rng.random((1, self.T, self.U))
        else:
            u = rng.random((1, self.T, self.U)) - 0.5
        return self.amp / self.K * (G * self._r(y, u)).sum((1, 2))


# ------------------------------------------------------------------------------------------------ Jensen (nonlinear) toy
class NLToy(ErrModel):
    """Fresh dither passing through a nonlinear head before the cost (coordinator request 2). U hidden units with
    activations x_j(a) = sx * w_j . (a - c0)/s0 + x0_j in [0, 1] (bits k: grid Delta = 1/(2^k - 1), y = x / Delta),
    rounding errors eps_j = Delta r_j (r as in RoundToy, per-unit offsets), latent error dz = Wz eps / sqrt(U) (m dims),
    residual rho(a) = P (a - a*) (first-order sensitivity vanishes at the optimum), and
        e(a) = amp [ 2 rho(a) . dz(a) + kappa ||dz(a)||^2 ].
    E_u[e] = amp kappa Delta^2 / U sum_j ||Wz_j||^2 v_j(a): v = f(1-f) (stochastic rounding = non-subtractive dither:
    an input-dependent persistent field), v = 1/12 (det+shift = subtractive dither: constant)."""

    def __init__(self, struct, variant, amp, kappa, bits, a_star):
        self.st, self.var, self.amp, self.kappa = struct, variant, float(amp), float(kappa)
        self.delta = 1.0 / (2 ** int(bits) - 1)
        self.persistent = variant in ('det', 'fixed')
        self.a_star = a_star
        self.U = struct['n_unit']
        self.name = f'NL-{variant}'

    @staticmethod
    def make_struct(rng, d, n_unit=256, m=8, sx=0.3, c0=None, s0=1.0):
        w = rng.standard_normal((n_unit, d))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        return dict(d=d, n_unit=n_unit, m=m, sx=sx, w=w, x0=rng.uniform(0.15, 0.85, n_unit),
                    Wz=rng.standard_normal((m, n_unit)), P=rng.standard_normal((m, d)) / math.sqrt(d),
                    c0=np.zeros(d) if c0 is None else c0, s0=s0)

    def _y(self, A):
        st = self.st
        x = st['sx'] * (((A - st['c0']) / st['s0']) @ st['w'].T) + st['x0'][None]
        return x / self.delta

    def _r(self, y, u):
        v = self.var
        if v == 'det':
            return np.round(y) - y
        if v in ('ind', 'shiter', 'fixed'):
            return np.floor(y + u) - y
        return np.round(y - u) + u - y

    def _e_from_r(self, A, r):
        st = self.st
        dz = self.delta * (r @ st['Wz'].T) / math.sqrt(self.U)                # (N, m)
        rho = (A - self.a_star) @ st['P'].T
        return self.amp * (2.0 * (rho * dz).sum(1) + self.kappa * (dz ** 2).sum(1))

    def _u(self, rng, N=None):
        if self.var == 'ind':
            return rng.random((N, self.U))
        if self.var == 'detshift':
            return rng.random((1, self.U)) - 0.5
        return rng.random((1, self.U))

    def begin_plan(self, rng):
        self.rng = rng
        self.u = self._u(rng) if self.var == 'fixed' else None

    def begin_iter(self):
        if self.var in ('shiter', 'detshift'):
            self.u = self._u(self.rng)

    def __call__(self, A):
        y = self._y(A)
        if self.var == 'det':
            return self._e_from_r(A, self._r(y, None))
        u = self._u(self.rng, len(A)) if self.var == 'ind' else self.u
        return self._e_from_r(A, self._r(y, u))

    def same_draw(self, A, rng_aux):
        if self.var == 'ind':
            return self._e_from_r(A, self._r(self._y(A), rng_aux.random((len(A), self.U))))
        return self(A)

    def redraw(self, A, rng):
        if self.var in ('det', 'fixed'):
            return self(A)
        return self._e_from_r(A, self._r(self._y(A), self._u(rng, len(A))))

    def bias_exact(self, A):
        """Exact E_u[e(a)] for the dithered variants (first-order term vanishes; E||dz||^2 = Delta^2/U sum ||Wz_j||^2
        v_j)."""
        y = self._y(A)
        f = np.mod(y, 1.0)
        v = np.full_like(f, 1.0 / 12.0) if self.var == 'detshift' else f * (1 - f)
        wz2 = (self.st['Wz'] ** 2).sum(0)
        return self.amp * self.kappa * self.delta ** 2 / self.U * (v @ wz2)


# ================================================================================================ planners
def _corr(x, y):
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _gap(J):
    if len(J) < 2:
        return np.nan
    p = np.partition(J, 1)[:2]
    return float(p[1] - p[0])


def mppi_plan(prob, err, cfg, rng_plan, rng_err, rng_pick, rng_aux, prior_center, rec=(1, 3, 6, 12), mean0=None,
              keep_traj=True):
    """TD-MPC2 MPPI on Jhat = J + e. Returns dict(rec={n: metrics}, traj=[per-iteration stats], sig=[sampling std
    vectors], it0={...})."""
    N, k, npi, T, temp = int(cfg['N']), int(cfg['k']), int(cfg['npi']), int(cfg['iters']), float(cfg['temp'])
    d, Ad = prob.d, int(cfg['A'])
    mean = np.zeros(d) if mean0 is None else np.array(mean0, float)
    std = np.full(d, float(cfg['max_std']))
    prior = (np.clip(prior_center[None] + cfg['pi_std'] * rng_plan.standard_normal((npi, d)), -1, 1)
             if npi > 0 else np.zeros((0, d)))
    err.begin_plan(rng_err)
    out = dict(rec={}, traj=[], sig=[])
    e0 = None
    for it in range(T):
        err.begin_iter()
        samp = np.clip(mean + std * rng_plan.standard_normal((N - npi, d)), -1, 1)
        A = np.concatenate([prior, samp]) if npi > 0 else samp
        J = prob.J(A)
        e = err(A)
        V = -(J + e)
        idx = np.argpartition(-V, k - 1)[:k] if k < N else np.arange(N)
        ev = V[idx]
        sc = np.exp(temp * (ev - ev.max()))
        sc = sc / sc.sum()
        el = A[idx]
        out['sig'].append(std.copy())
        mean = (sc @ el) / (sc.sum() + 1e-9)
        std = np.clip(np.sqrt((sc @ (el - mean) ** 2) / (sc.sum() + 1e-9)), cfg['min_std'], cfg['max_std'])
        Js, es = J[npi:], e[npi:]
        sJ, se = float(Js.std()), float(es.std())
        rho = _corr(Js, es)
        sV = float(V[npi:].std())
        er = float(es.mean())
        jt = int(idx[np.argmax(ev)])
        st = dict(sJ=sJ, se=se, rho=rho, e_rand=er, theta=temp * sV, keff=float(1.0 / (sc ** 2).sum()),
                  e_w=float(sc @ e[idx]), e_top1=float(e[jt]), std=float(out['sig'][-1].mean()),
                  J_rand=float(Js.mean()))
        if keep_traj:
            out['traj'].append(st)
        if it == 0:
            e0 = er
            out['it0'] = dict(sJ=sJ, se=se, rho=rho, e0=er, gap=_gap(Js), theta=temp * sV, oei_first=st['e_w'] - er)
        if (it + 1) in rec:
            u = rng_pick.random()
            j = min(int(np.searchsorted(np.cumsum(sc), u)), k - 1)
            ch = int(idx[j])
            a_exec = A[ch].copy()
            e_ex = float(e[ch])
            e_rd = float(err.redraw(a_exec[None], rng_aux)[0])
            stein = F.stein_selection(J + e, k, se, temp)
            out['rec'][it + 1] = dict(
                oei=e_ex - e0, oei_rd=e_rd - e0, sel_last=e_ex - er, sel_last_rd=e_rd - er, drift=er - e0,
                stein_pred=stein,
                oei_w=st['e_w'] - e0, oei_top1=st['e_top1'] - e0, sel_w=st['e_w'] - er,
                J_exec=float(J[ch]), J_mean=float(prob.J(mean[None])[0]), a1=a_exec[:Ad], a_exec=a_exec,
                mean=mean.copy(), sJ=sJ, se=se, rho=rho, theta=st['theta'], keff=st['keff'], N=N, k=k)
    return out


def cem_plan(prob, err, cfg, rng_plan, rng_err, rng_aux, rec=(5, 10, 30), keep_traj=True):
    """DINO-WM CEM on Jhat = J + e (open loop). Returns dict(rec={n: metrics}, traj, sig, it0)."""
    N, k, T = int(cfg['N']), int(cfg['k']), int(cfg['iters'])
    d, Ad = prob.d, int(cfg['A'])
    mu = np.zeros(d)
    sig = np.full(d, float(cfg['var_scale']))
    err.begin_plan(rng_err)
    out = dict(rec={}, traj=[], sig=[])
    e0 = None
    for i in range(T):
        err.begin_iter()
        A = rng_plan.standard_normal((N, d)) * sig + mu
        A[0] = mu
        J = prob.J(A)
        e = err(A)
        idx = np.argsort(J + e, kind='stable')[:k]
        out['sig'].append(sig.copy())
        el = A[idx]
        mu = el.mean(0)
        sig = el.std(0, ddof=1)
        Jr, er_ = J[1:], e[1:]
        sJ, se = float(Jr.std()), float(er_.std())
        rho = _corr(Jr, er_)
        er = float(er_.mean())
        st = dict(sJ=sJ, se=se, rho=rho, e_rand=er, e_elite=float(e[idx].mean()), e_argmin=float(e[idx[0]]),
                  std=float(out['sig'][-1].mean()), J_rand=float(Jr.mean()))
        if keep_traj:
            out['traj'].append(st)
        if i == 0:
            e0 = er
            out['it0'] = dict(sJ=sJ, se=se, rho=rho, e0=er, gap=_gap(Jr))
        if (i + 1) in rec:
            e_rd = float(err.redraw(mu[None], rng_aux)[0])
            e_sd = float(err.same_draw(mu[None], rng_aux)[0])
            out['rec'][i + 1] = dict(
                stein_pred_elite=F.stein_selection(J + e, k, se, 0.0),
                ntc_pred_argmin=F.argmin_near_tie(Jr, se) if se > 0 else 0.0,
                oei_mf=e_rd - e0, sel_last_mf=e_rd - er, sel_last_mf_sd=e_sd - er, oei_mf_sd=e_sd - e0, drift=er - e0,
                oei_elite=st['e_elite'] - e0, sel_last_elite=st['e_elite'] - er, oei_argmin=st['e_argmin'] - e0,
                sel_last_argmin=st['e_argmin'] - er, J_mf=float(prob.J(mu[None])[0]), a1=mu[:Ad].copy(),
                mean=mu.copy(), sJ=sJ, se=se, rho=rho, N=N, k=k)
    return out


# ================================================================================================ SAA prediction
def saa_track(prob, field, amp, sig_traj, box=None, newton=25):
    """Graduated-smoothing SAA prediction for a persistent random field: follow the minimizer of
    J(mu) + amp * E_z[b(mu + sigma_t z)] as sigma_t shrinks along the planner's sampling-std trajectory, starting at a*.
    Returns the predicted mean after every iteration."""
    mu = prob.a_star.copy()
    preds = []
    for sig in sig_traj:
        for _ in range(newton):
            _, g, Hs = field.smoothed(mu, sig)
            grad = prob.H @ (mu - prob.a_star) + amp * g
            Hm = prob.H + amp * Hs
            try:
                ev = np.linalg.eigvalsh(Hm)
                step = np.linalg.solve(Hm, grad) if ev.min() > 1e-6 else np.linalg.solve(prob.H, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.solve(prob.H, grad)
            nrm = np.linalg.norm(step)
            if nrm > 0.2:
                step *= 0.2 / nrm
            mu = mu - step
            if box is not None:
                mu = np.clip(mu, -box, box)
            if nrm < 1e-9:
                break
        preds.append(mu.copy())
    return preds


# ================================================================================================ closed loop (warm start)
class LQLoop:
    """Closed-loop block for MPPI warm start (report section 3.7). State x in R^2, action u in [-1, 1]^2,
    x_{t+1} = x_t + 0.5 u_t, stage cost x^T x + 0.1 u^T u, horizon 3, terminal value x^T P x (discrete Riccati), so the
    planning objective in the action sequence a = (u_0, u_1, u_2) is quadratic: J_t(a) = 1/2 a^T H a + g_t^T a + c_t."""

    def __init__(self, H_steps=3, dx=2, b=0.5, ru=0.1, gamma=0.99):
        self.Hs, self.dx, self.b, self.ru, self.gamma = H_steps, dx, b, ru, gamma
        P = np.eye(dx)
        for _ in range(500):
            P = np.eye(dx) + gamma * (P - gamma * b * b * P @ np.linalg.inv(ru * np.eye(dx) + gamma * b * b * P) @ P)
        self.P = P
        # quadratic form: x_h = x0 + b sum_{i<h} u_i
        n = H_steps * dx
        Hq = np.zeros((n, n))
        L = []
        for h in range(H_steps + 1):
            Lh = np.zeros((dx, n))
            for i in range(h):
                Lh[:, i * dx:(i + 1) * dx] = b * np.eye(dx)
            L.append(Lh)
        self.L = L
        Hq = np.zeros((n, n))
        for h in range(H_steps):
            w = gamma ** h
            Hq += 2 * w * L[h].T @ L[h]
            Hq[h * dx:(h + 1) * dx, h * dx:(h + 1) * dx] += 2 * w * ru * np.eye(dx)
        Hq += 2 * gamma ** H_steps * L[H_steps].T @ P @ L[H_steps]
        self.Hq = Hq
        self.d = n

    def objective(self, x0):
        """Quadratic object with a_star and H for the planning problem at state x0 (J shifted so that min = 0)."""
        n, dx = self.d, self.dx
        g = np.zeros(n)
        for h in range(self.Hs):
            g += 2 * self.gamma ** h * self.L[h].T @ x0
        g += 2 * self.gamma ** self.Hs * self.L[self.Hs].T @ self.P @ x0
        a_star = -np.linalg.solve(self.Hq, g)
        q = Quadratic.__new__(Quadratic)
        q.d, q.H, q.Hinv, q.a_star = n, self.Hq, np.linalg.inv(self.Hq), a_star
        q.lam = np.linalg.eigvalsh(self.Hq)
        return q

    def step(self, x, u):
        return x + self.b * u, float(x @ x + self.ru * u @ u)


class JointField(ErrModel):
    """Persistent error over (state, action sequence): e = amp * b([x / ell_x_scale, a]) (det / sh-ep analogue in the
    closed loop); with per_step=True a new field is drawn at every control step (sh-step analogue)."""

    def __init__(self, rng_struct, d_x, d_a, ell, amp, per_step=False, F=128):
        self.dx, self.da, self.ell, self.amp, self.per_step, self.F = d_x, d_a, ell, float(amp), per_step, F
        self.field = RFF(rng_struct, d_x + d_a, ell, F)
        self.persistent = True
        self.name = 'Mstep' if per_step else 'M1j'

    def set_state(self, x, rng=None):
        self.x = np.asarray(x, float)
        if self.per_step and rng is not None:
            self.field = RFF(rng, self.dx + self.da, self.ell, self.F)

    def __call__(self, A):
        X = np.broadcast_to(self.x, (len(A), self.dx))
        return self.amp * self.field(np.concatenate([X, A], 1))
