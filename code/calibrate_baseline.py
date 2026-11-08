"""Plan A: Bayesian calibration of the two-branch depletion model on digitized
real data from arXiv:1506.07320v2 (Fig 1 Si standard + Fig 4 GaN films).

Identifiability structure (exact):
  raw Mott-Schottky   1/C^2 = (2/(q*eps*k)) * (Vbi - V),  k = N*A^2
  -> from one C-V curve: Vbi (A-independent) and k=N*A^2 (N,A degenerate)
  -> A from full-depletion flat films: C = eps_GaN/t  =>  A = C*t/eps
  -> Si inset y-axis is 1/C_spec^2 (per-area): slope gives N directly.
"""
import numpy as np, csv, json
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = DATA / "results"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
q = 1.602176634e-19
eps0 = 8.854187817e-14
epsSi = 11.7 * eps0
epsGaN = 10.6 * eps0
DRAWS = 200_000

def blr_posterior(V, Y, vmax, vmin=-1e9):
    """Conjugate Bayesian linear regression (Jeffreys prior) on vmin<=V<=vmax.
    Returns posterior draws of (b0, b1)."""
    m = (V <= vmax) & (V >= vmin)
    X = np.column_stack([np.ones(m.sum()), V[m]])
    Yf = Y[m]
    n = len(Yf)
    XtX_inv = np.linalg.inv(X.T @ X)
    bh = XtX_inv @ X.T @ Yf
    s2 = ((Yf - X @ bh) ** 2).sum() / (n - 2)
    L = np.linalg.cholesky(XtX_inv)
    z = rng.standard_normal((DRAWS, 2)) @ L.T
    chi2 = rng.chisquare(n - 2, DRAWS)
    return bh + z * np.sqrt(s2 * (n - 2) / chi2)[:, None]

def summ(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(np.mean(x)), float(lo), float(hi)

def load(fn):
    d = np.genfromtxt(DATA / fn, delimiter=",", names=True)
    return d["V_app_V"], d["V_app_V"] * 0 + d[d.dtype.names[1]]

results = {}

# ============ 1. Si standard (Fig 1 inset: 1/C_spec^2 vs V) ============
d = np.genfromtxt(DATA / "si_ms_inset_digitized.csv", delimiter=",", names=True)
Vs, Ys = d["V_app_V"], d["invCspec2_cm4_per_F2"]
beta = blr_posterior(Vs, Ys, -1.0)
b0, b1 = beta[:, 0], beta[:, 1]
VbiSi = -b0 / b1
NSi = -2.0 / (q * epsSi * b1)
results["Si_standard"] = {
    "Vbi_V": summ(VbiSi), "N_cm3": summ(NSi),
    "reported": {"Vbi_V": "0.37-0.38", "N_cm3": 7.66e14}}
print("== Si standard (Fig 1 inset, V<=-1.0) ==")
print("  Vbi =", summ(VbiSi), " reported 0.37-0.38")
print("  N   =", summ(NSi), " reported 7.66e14")
np.save(OUT / "post_Si_beta.npy", beta)

# ============ 2. GaN full-depletion flat films -> probe area A ============
eps_ratio = epsGaN
flats = [("u-GaN 500nm", "gan_u_500nm_cv.csv", 500e-7),
         ("u-GaN 1000nm", "gan_u_1000nm_cv.csv", 1000e-7),
         ("n1 100nm", "gan_n1_100nm_cv.csv", 100e-7),
         ("n2 100nm", "gan_n2_100nm_cv.csv", 100e-7)]
A_est = []
print("\n== GaN full-depletion flat films: A = C*t/eps ==")
for name, fn, t in flats:
    V, C = load(fn)
    m = V <= -1.5
    Cl = float(np.median(C[m]))
    A = Cl * t / eps_ratio
    A_est.append(A)
    print(f"  {name:12s}: C = {Cl:.4e} F  ->  A = {A:.4e} cm^2")
A_est = np.array(A_est)
A_mean, A_sd = float(A_est.mean()), float(A_est.std(ddof=1))
print(f"  combined: A = {A_mean:.4e} +/- {A_sd:.2e} cm^2 ({A_sd/A_mean*100:.1f}%)")

# A from Si main curve for comparison (k_Si = N*A^2)
d2 = np.genfromtxt(DATA / "si_cv_digitized.csv", delimiter=",", names=True)
Vm, Cm = d2["V_app_V"], d2["C_F"]
beta_m = blr_posterior(Vm, 1.0 / Cm**2, -1.5)
k_Si_draws = -2.0 / (q * epsSi * beta_m[:, 1])
A_Si = np.sqrt(k_Si_draws / NSi)
print(f"  A from Si main curve (C-V slope / N_Si): {summ(A_Si)}")


expected = json.loads((DATA / "baseline_calibration.json").read_text())
np.testing.assert_allclose(results["Si_standard"]["Vbi_V"], expected["Si_standard"]["Vbi_V"], rtol=1e-12)
np.testing.assert_allclose(results["Si_standard"]["N_cm3"], expected["Si_standard"]["N_cm3"], rtol=1e-12)
results["area_model"] = expected["area_model"]
np.testing.assert_allclose([A_mean,A_sd],[expected["area_model"]["mean_cm2"],expected["area_model"]["sd_cm2"]],rtol=1e-12)
(OUT / "baseline_results.json").write_text(json.dumps(results,indent=2)+"\n")
