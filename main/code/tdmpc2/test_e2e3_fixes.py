"""Unit checks of the 2026-10-23 analysis fixes (synthetic numbers only)."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/pmx/A/tdmpc2_main/code"))
import numpy as np
from scipy.stats import spearmanr
import v5_common as C
rng = np.random.default_rng(1)
# 1. spearman_rows == scipy spearmanr per row, also with heavy ties
for L in (4, 5, 12):
    X = rng.integers(0, 4, size=(2000, L)).astype(float)
    y = np.arange(L, dtype=float)
    got = C.spearman_rows(X, y)
    ref = np.array([spearmanr(x, y).correlation if np.ptp(x) > 0 else np.nan for x in X])
    ok = np.isfinite(ref)
    assert np.array_equal(np.isfinite(got), ok), "NaN pattern"
    assert np.max(np.abs(got[ok] - ref[ok])) < 1e-12, np.max(np.abs(got[ok] - ref[ok]))
    Y = rng.normal(size=(2000, L))
    got2 = C.spearman_rows2(X, Y)
    ref2 = np.array([spearmanr(x, yy).correlation if np.ptp(x) > 0 else np.nan for x, yy in zip(X, Y)])
    ok2 = np.isfinite(ref2)
    assert np.max(np.abs(got2[ok2] - ref2[ok2])) < 1e-12
print("spearman_rows / rows2 == scipy (ties averaged): OK")
# 2. thresholds: exact grid values pass, the next grid value does not
four = [1, 2, 3, 4]
assert C.le(C.spearman(four, [4, 3, 1, 2]), -0.8) and C.le(C.spearman(four, [4, 3, 2, 1]), -0.8)
assert not C.le(C.spearman(four, [3, 4, 1, 2]), -0.8)     # rho = -0.6
five = [1, 2, 3, 4, 5]
assert C.ge(C.spearman(five, [2, 1, 3, 5, 4]), 0.8) and not C.ge(C.spearman(five, [3, 1, 2, 4, 5]), 0.8)
t12 = list(range(12))
y = list(range(12)); y[0], y[11] = y[11], y[0]
print("12-point swap of the ends:", C.spearman(t12, y), C.ge(C.spearman(t12, y), 0.5))
assert not C.ge(float("nan"), 0.5) and not C.le(float("nan"), -0.8)
print("threshold tolerance: OK (exact -0.8 / 0.8 pass, -0.6 / 0.7 fail, NaN fails)")
b = C.boot_rho_summary(np.array([-1.0, -0.7999999999999999, -0.6, np.nan]), -0.8, "le")
assert b["share_le_-0.8"] == 2 / 3 and b["n_defined"] == 3, b
print("boot_rho_summary: OK", b)
