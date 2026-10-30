"""E4 (paper A, EXPERIMENT_DESIGN 4.4 / 6.4): closed-form predictions used by the simulation analysis and the E2/E3
overlay hooks. Pure numpy/scipy; no simulation here (the expensive Monte Carlo parts are tabulated once on the server
by build_tables()).

Notation (report reports/E4_theory.md, section 2):
  candidates i of one planner evaluation, true cost J_i, cost error e_i, model cost Jh_i = J_i + e_i.
  sigma_J, sigma_e = across-candidate std of J and e in that population (common mode removed), rho = corr(J, e).
  S_op = selection intensity of the planner's selection operator in standard-deviation units of Jh:
         argmin            S = E[max of N std normals]                         (exp_max)
         top-k mean        S = mean of the top-k expected order statistics      (topk_intensity)
         MPPI              S = E[sum_j w_j Z_(j)], top-k then w ~ exp(theta Z)  (mppi_intensity, theta = temperature *
                               sigma_Jh); the TD-MPC2 Gumbel pick has the same expectation as the weighted mean.
  Winner's curse (bivariate normal, exact): E[e_sel] - E[e] = -(sigma_e^2 + rho sigma_J sigma_e) / sigma_Jh * S_op.
"""
import math
import numpy as np
from scipy import special, stats

# ---------------------------------------------------------------------------------------------- order statistics
_ZG = np.linspace(-9.0, 9.0, 36001)
_LPHI = stats.norm.logcdf(_ZG)
_LSF = stats.norm.logsf(_ZG)
_LPDF = stats.norm.logpdf(_ZG)


def order_stat_means(N):
    """E[Z_(r)] for r = 1..N where Z_(1) >= Z_(2) >= ... (descending), N iid standard normals (trapezoid quadrature on a
    fine grid, |error| < 1e-6)."""
    N = int(N)
    out = np.empty(N)
    lc = special.gammaln(N + 1) - special.gammaln(np.arange(1, N + 1)) - special.gammaln(N - np.arange(1, N + 1) + 1)
    for s in range(0, N, 64):
        r = np.arange(s + 1, min(N, s + 64) + 1)
        # density of the r-th largest: N!/((r-1)!(N-r)!) Phi^(N-r) (1-Phi)^(r-1) phi
        lf = lc[r - 1][:, None] + (N - r)[:, None] * _LPHI[None] + (r - 1)[:, None] * _LSF[None] + _LPDF[None]
        f = np.exp(lf)
        out[s:s + len(r)] = np.trapezoid(f * _ZG[None], _ZG, axis=1)
    return out


_OS_CACHE = {}


def _os(N):
    if N not in _OS_CACHE:
        _OS_CACHE[N] = order_stat_means(N)
    return _OS_CACHE[N]


_EMAX = {}


def exp_max(N):
    """E[max of N iid N(0,1)] (argmin selection intensity); single quadrature, cached."""
    N = int(N)
    if N not in _EMAX:
        lf = math.log(N) + (N - 1) * _LPHI + _LPDF
        _EMAX[N] = float(np.trapezoid(np.exp(lf) * _ZG, _ZG))
    return _EMAX[N]


def topk_intensity(N, k):
    """Mean of the top-k expected order statistics of N standard normals (elite-mean selection intensity)."""
    return float(_os(int(N))[:int(k)].mean())


def topk_intensity_asym(q):
    """Large-N limit phi(z_q)/q of the top-q-fraction mean."""
    z = stats.norm.isf(q)
    return float(stats.norm.pdf(z) / q)


def mppi_intensity(N, k, thetas, reps=20000, seed=12345, chunk=1000):
    """Monte Carlo selection intensity of TD-MPC2's MPPI update (top-k elites, weights softmax(theta Z) over the elites;
    Z in standard-deviation units of the model value). Returns S(theta) for an array of thetas, plus the effective
    elite count k_eff = 1/sum w^2 (mean over reps). theta -> 0: top-k mean; theta -> inf: argmax."""
    thetas = np.atleast_1d(np.asarray(thetas, float))
    rng = np.random.default_rng(seed)
    S = np.zeros(len(thetas))
    K = np.zeros(len(thetas))
    n = 0
    for s in range(0, reps, chunk):
        m = min(chunk, reps - s)
        Z = rng.standard_normal((m, int(N)))
        top = -np.sort(-Z, axis=1)[:, :int(k)]                      # descending
        for i, th in enumerate(thetas):
            w = np.exp(th * (top - top[:, :1]))
            w /= w.sum(1, keepdims=True)
            S[i] += (w * top).sum(1).sum()
            K[i] += (1.0 / (w ** 2).sum(1)).sum()
        n += m
    return S / n, K / n


def _mppi_all_k(N, ks, thetas, reps=20000, seed=0, chunk=500):
    """mppi_intensity for several elite counts from one Monte Carlo sample (sorted once)."""
    rng = np.random.default_rng(seed)
    kmax = max(ks)
    Sk = {k: np.zeros(len(thetas)) for k in ks}
    Kk = {k: np.zeros(len(thetas)) for k in ks}
    n = 0
    for s in range(0, reps, chunk):
        m = min(chunk, reps - s)
        Z = rng.standard_normal((m, int(N)))
        top_all = -np.sort(-Z, axis=1)[:, :kmax]
        for k in ks:
            top = top_all[:, :k]
            for i, th in enumerate(thetas):
                w = np.exp(th * (top - top[:, :1]))
                w /= w.sum(1, keepdims=True)
                Sk[k][i] += (w * top).sum()
                Kk[k][i] += (1.0 / (w ** 2).sum(1)).sum()
        n += m
    return {k: v / n for k, v in Sk.items()}, {k: v / n for k, v in Kk.items()}


def winners_curse(sig_e, sig_J, rho, S):
    """Expected error of the selected candidate(s) minus the population mean error (negative = favourable error picked).
    Exact for jointly normal (J, e); S = selection intensity of the operator."""
    sig_e, sig_J, rho = np.asarray(sig_e, float), np.asarray(sig_J, float), np.asarray(rho, float)
    var_h = sig_J ** 2 + sig_e ** 2 + 2 * rho * sig_J * sig_e
    return -(sig_e ** 2 + rho * sig_J * sig_e) / np.sqrt(np.maximum(var_h, 1e-300)) * S


def regret_selection(sig_e, sig_J, rho, S):
    """Expected true cost of the selected candidate(s) minus the population mean of J (attenuated selection on J)."""
    sig_e, sig_J, rho = np.asarray(sig_e, float), np.asarray(sig_J, float), np.asarray(rho, float)
    var_h = sig_J ** 2 + sig_e ** 2 + 2 * rho * sig_J * sig_e
    return -(sig_J ** 2 + rho * sig_J * sig_e) / np.sqrt(np.maximum(var_h, 1e-300)) * S


def argmin_cond_exact(J, sig_e, n_cap=128, ngrid=121):
    """Exact E[e_argmin | J] - 0 for errors e_i iid N(0, sig_e^2) on a FIXED population of true costs J (any
    distribution): sum_i int e phi_s(e) prod_{j != i} Phi((J_j - J_i - e) / s) de, restricted to the n_cap lowest J
    (candidates far above the minimum cannot win). Refines winners_curse when J is far from normal."""
    s = float(sig_e)
    if s <= 0:
        return 0.0
    J = np.sort(np.asarray(J, float))
    n_in = int(np.sum(J - J[0] < 10.0 * s))           # candidates that can still win (adaptive cap)
    J = J[:max(2, min(len(J), max(n_cap, n_in)))]
    z = np.linspace(-7, 7, ngrid)
    e = s * z
    w = stats.norm.pdf(z) * (z[1] - z[0])
    D = (J[None, :] - J[:, None])                                   # D[i, j] = J_j - J_i
    tot = 0.0
    for g in range(ngrid):
        L = stats.norm.logcdf((D - e[g]) / s)                        # (i, j)
        np.fill_diagonal(L, 0.0)
        tot += w[g] * e[g] * np.exp(L.sum(1)).sum()
    return float(tot)


def stein_selection(Jh, k, sig_e_perp, temp=0.0):
    """Stein-identity form of the selection term for errors independent across candidates with variance
    sig_e_perp^2 (common mode removed), any distribution of J:
        E[sum_i w_i e_i] = -sig^2 E[sum_i dw_i/dJh_i] = -sig^2 [temp (1 - sum w^2) + N f(t_k) w_k],
    w = top-k elites with softmax(-temp Jh) weights, t_k = k-th smallest model cost, f = density of Jh at t_k
    (k-NN estimate), w_k = weight of the marginal elite. temp = 0: top-k mean, -sig^2 f(t_k) / q. No boundary term
    when k = N."""
    Jh = np.sort(np.asarray(Jh, float))
    N = len(Jh)
    k = int(k)
    el = Jh[:k]
    w = np.exp(-temp * (el - el[0]))
    w /= w.sum()
    smooth = temp * (1.0 - float((w ** 2).sum()))
    if k >= N:
        return -sig_e_perp ** 2 * smooth
    m = max(2, int(math.sqrt(N) / 2))
    lo, hi = max(0, k - 1 - m), min(N - 1, k - 1 + m)
    f = (hi - lo) / (N * max(Jh[hi] - Jh[lo], 1e-300))
    return -sig_e_perp ** 2 * (smooth + N * f * float(w[-1]))


def near_tie_count(J, sig_e):
    """Effective number of candidates competing with the best: n_c = sum_j 2 Phi(-(J_j - J_min) / (sqrt 2 sig_e))."""
    J = np.asarray(J, float)
    return float(np.sum(2 * stats.norm.cdf(-(J - J.min()) / (math.sqrt(2) * sig_e))))


def argmin_near_tie(J, sig_e):
    """Near-tie approximation of the argmin winner's curse: -sig_e * E[max of n_c normals] (n_c from near_tie_count,
    log-linear interpolation of the expected maximum between integers)."""
    n = max(1.0, near_tie_count(J, sig_e))
    lo = int(math.floor(n))
    hi = lo + 1
    elo = exp_max(lo) if lo > 1 else 0.0
    ehi = exp_max(hi)
    t = (math.log(n) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return -sig_e * ((1 - t) * elo + t * ehi)


# ---------------------------------------------------------------------------------------------- rounding with offsets
def bb_kernel(y1, y2):
    """E_u[r(y1;u) r(y2;u)] for the stochastic-rounding error r(y;u) = floor(y + u) - y with ONE shared offset
    u ~ U[0,1): the Brownian-bridge kernel min(f1, f2) - f1 f2 of the fractional parts."""
    f1, f2 = np.mod(y1, 1.0), np.mod(y2, 1.0)
    return np.minimum(f1, f2) - f1 * f2


def V_offset(sigma_y, mmax=400):
    """Expected across-candidate variance (over a population with y ~ N(ybar, sigma_y^2), grid units) of the rounding
    error of ONE entry with a fixed random offset, averaged over the offset:
        V(s) = 1/12 - sum_m exp(-4 pi^2 m^2 s^2) / (2 pi^2 m^2);  V(0) = 0, V(s) ~ s/sqrt(pi) (s << 1), V(inf) = 1/12."""
    s = np.asarray(sigma_y, float)
    m = np.arange(1, mmax + 1)
    return 1.0 / 12.0 - (np.exp(-4 * np.pi ** 2 * m[None] ** 2 * s.reshape(-1, 1) ** 2) / (2 * np.pi ** 2 * m[None] ** 2)).sum(1).reshape(s.shape)


def rho_cand_shared(sigma_y, mmax=400):
    """Across-candidate correlation of the error of one entry under a SHARED fresh offset (sh-iter), constant
    sensitivity, uniform phase: rho = 1/2 + 3 sum_m exp(-4 pi^2 m^2 s^2)/(pi^2 m^2). rho(0) = 1, rho(inf) = 1/2.
    The exploitable (non-common) share is 1 - rho = 6 V(s); independent dither (ind) has share 1 (no common mode)."""
    return 1.0 - 6.0 * V_offset(sigma_y, mmax)


def var_cand_neff(n_eff, A, B):
    """Two-term law of the across-candidate variance of a fixed shared dither table with n_eff independent offset
    groups: coherent part A / n_eff + incoherent floor B (report section 2.5)."""
    return A / np.asarray(n_eff, float) + B


# ---------------------------------------------------------------------------------------------- persistent smooth field
def rff_saa_moments(sig_b, ell, Hinv, sigma_pop, d):
    """Weak-field SAA predictions for a stationary random field b with variance sig_b^2 and squared-exponential
    covariance of length ell (random Fourier features omega ~ N(0, I/ell^2)), planner population std sigma_pop
    (isotropic) and objective Hessian inverse Hinv (d x d). Returns
      shift2  = E||delta||^2,            delta = -H^{-1} grad b_sigma(a*)
      oei_mf  = E[b(mu*) - E_p0 b]      = -E[grad b . H^{-1} grad b_sigma]
      regret  = E[1/2 delta^T H delta]  = 1/2 E[grad b_sigma^T H^{-1} grad b_sigma]
    using E[omega omega^T exp(-c ||omega||^2)] = (I/ell^2) (1 + 2c/ell^2)^(-d/2-1)."""
    tH = float(np.trace(Hinv))
    tH2 = float(np.trace(Hinv @ Hinv))
    s2 = sigma_pop ** 2
    f_ss = (1 + 2 * s2 / ell ** 2) ** (-d / 2 - 1)        # both factors smoothed
    f_s1 = (1 + s2 / ell ** 2) ** (-d / 2 - 1)            # one factor smoothed
    return dict(shift2=sig_b ** 2 * tH2 / ell ** 2 * f_ss,
                oei_mf=-sig_b ** 2 * tH / ell ** 2 * f_s1,
                regret=0.5 * sig_b ** 2 * tH / ell ** 2 * f_ss)


# ---------------------------------------------------------------------------------------------- fresh noise (SA)
def fresh_white_stationary(sig_eps, S, k_eff, sigma, H):
    """Stationary covariance and regret of the planner mean under fresh i.i.d. noise (noise-dominated selection,
    population std sigma per coordinate, linear response): delta_{t+1} = (I - alpha H) delta_t + xi_t,
    alpha = sigma^2 S / sig_eps, Var(xi) = sigma^2 / k_eff. Stationary Var = sig_eps / (2 S k_eff) H^-1,
    regret of the mean = d sig_eps / (4 S k_eff); relaxation time 1 / (alpha lambda_min) iterations."""
    d = H.shape[0]
    lam = np.linalg.eigvalsh(H)
    alpha = sigma ** 2 * S / sig_eps
    return dict(var_scale=sig_eps / (2 * S * k_eff), regret_mean=d * sig_eps / (4 * S * k_eff),
                tau_max=1.0 / (alpha * lam.min()), alpha=alpha)


def ar1_transient(lams, alphas, vs, delta0):
    """Mean-square deviation along eigen-directions after T iterations of delta_{t+1} = (1 - alpha_t lam) delta_t + xi_t,
    Var(xi_t) = v_t: returns (bias^2 part, variance part) per direction."""
    lams = np.asarray(lams, float)
    b = np.asarray(delta0, float) ** 2
    v = np.zeros_like(lams)
    for a, vt in zip(alphas, vs):
        c = (1 - a * lams) ** 2
        b = b * c
        v = v * c + vt
    return b, v


# ---------------------------------------------------------------------------------------------- Jensen (2nd order)
def jensen_bias_field(delta, h_diag, frac, subtractive=False):
    """Leading-order bias of the model cost under unbiased dither of hidden activations:
        b2(a) = 1/2 sum_i Delta_i^2 h_ii(a) v_i(a),  v_i = f_i (1 - f_i) (non-subtractive / stochastic rounding),
                                                      v_i = 1/12           (subtractive dither = det+shift),
    h_ii = d^2 Jhat / dx_i^2 at the unquantized activations, f_i = fractional grid position of x_i. Cross terms vanish
    for offsets independent across entries (ind and sh-iter alike: within one candidate the entries' offsets are
    independent). Scales as Delta^2 ~ 4^-k."""
    v = np.full_like(np.asarray(frac, float), 1.0 / 12.0) if subtractive else frac * (1 - frac)
    return 0.5 * np.sum(np.asarray(delta) ** 2 * np.asarray(h_diag) * v, axis=-1)


def jensen_drift_ratio(delta, h_over_g, ell_ratio=1.0):
    """Predicted drift of the Jensen field relative to det's drift (incoherent errors, SAA regime, same planner):
    (sigma_b2 / sigma_det)^2 (ell_det / ell_b2)^2 with sigma_b2 / sigma_det = Delta (h/g) / (2 sqrt(15)). ~ 4^-k."""
    r = np.asarray(delta, float) * np.asarray(h_over_g, float) / (2 * np.sqrt(15.0))
    return r ** 2 * ell_ratio ** 2


# ---------------------------------------------------------------------------------------------- tables
def build_tables(Ns=(64, 128, 256, 300, 512, 1024), qs=(0.02, 0.10, 0.125, 0.25), kfix=(30, 64),
                 thetas=None, reps=20000):
    """Selection-intensity tables used by the simulation workers and the analysis:
      exp_max[N], topk[N][k] (exact quadrature), mppi[N][k](theta) (Monte Carlo, reps)."""
    if thetas is None:
        thetas = np.concatenate([[0.0], np.logspace(-3, 2.5, 56)])
    tab = dict(thetas=thetas.tolist(), Ns=list(Ns), exp_max={}, topk={}, mppi={}, mppi_keff={})
    for N in Ns:
        os_ = _os(N)
        tab['exp_max'][str(N)] = float(os_[0])
        ks = sorted({max(2, int(round(q * N))) for q in qs} | {min(kf, N) for kf in kfix} | {1, 8, 16, 32, 64, 128})
        ks = [k for k in ks if 1 <= k <= N]
        tab['topk'][str(N)] = {str(k): float(os_[:k].mean()) for k in ks}
        tab['mppi'][str(N)] = {}
        tab['mppi_keff'][str(N)] = {}
        Sk, Kk = _mppi_all_k(N, ks, thetas, reps, seed=1000 + N)
        for k in ks:
            tab['mppi'][str(N)][str(k)] = Sk[k].tolist()
            tab['mppi_keff'][str(N)][str(k)] = Kk[k].tolist()
    return tab


class Tables:
    """Lookup of selection intensities with interpolation in theta (log scale) and on-demand fallbacks."""

    def __init__(self, tab):
        self.t = tab
        self.th = np.asarray(tab['thetas'], float)
        self._lth = np.log(np.maximum(self.th, 1e-4))

    def exp_max(self, N):
        v = self.t['exp_max'].get(str(int(N)))
        return v if v is not None else exp_max(N)

    def topk(self, N, k):
        v = self.t['topk'].get(str(int(N)), {}).get(str(int(k)))
        return v if v is not None else topk_intensity(N, k)

    def mppi(self, N, k, theta):
        row = self.t['mppi'].get(str(int(N)), {}).get(str(int(k)))
        if row is None:
            return float(mppi_intensity(N, k, [theta], reps=4000)[0][0])
        row = np.asarray(row)
        x = np.log(np.maximum(theta, 1e-4))
        return float(np.interp(x, self._lth, row))

    def mppi_keff(self, N, k, theta):
        row = self.t['mppi_keff'].get(str(int(N)), {}).get(str(int(k)))
        if row is None:
            return float(mppi_intensity(N, k, [theta], reps=4000)[1][0])
        return float(np.interp(np.log(max(theta, 1e-4)), self._lth, np.asarray(row)))


if __name__ == '__main__':
    import json
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 's_tables.json'
    tab = build_tables()
    json.dump(tab, open(out, 'w'))
    print('wrote', out, {N: tab['exp_max'][str(N)] for N in tab['Ns']})
