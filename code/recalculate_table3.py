"""Current 2026-10-31 package: recalculate_table3.py
All inputs and outputs are relative to this project. No external repair directory is used.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np
from provenance_checks import check_archived_hash

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT
DATA = PROJECT / "data"
OUT = DATA / "results"
BASELINE = PROJECT / "data/baseline_calibration.json"
SERIES = ["n1_500nm", "n1_750nm", "n2_750nm"]
SEEDS = {"n1_500nm": 4201, "n1_750nm": 4202, "n2_750nm": 42}
DRAWS = 200_000
Q = 1.602176634e-19
EPS_GAN = 10.6 * 8.854187817e-14
Y_SCALE = 1e18


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(a):
    return {
        "mc_sample_mean": float(np.mean(a)),
        "equal_tail_95pct_CI": np.percentile(a, [2.5, 97.5]).tolist(),
        "mc_standard_error_estimate": float(np.std(a, ddof=1)/np.sqrt(len(a))),
    }


def sample_coefficients(v, c, seed):
    x = np.column_stack((np.ones(len(v)), v))
    y = 1/c**2/Y_SCALE
    inv = np.linalg.inv(x.T @ x)
    bhat = inv @ x.T @ y
    residual = y-x @ bhat
    df = len(v)-2
    s2 = float(residual @ residual/df)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((DRAWS, 2)) @ np.linalg.cholesky(inv).T
    chi2 = rng.chisquare(df, DRAWS)
    beta = bhat+z*np.sqrt(s2*df/chi2)[:, None]
    fit = {
        "OLS_beta_scaled": bhat.tolist(), "beta_scale_factor_to_F_minus2": Y_SCALE,
        "OLS_Vbi_V": float(-bhat[0]/bhat[1]),
        "R_squared": float(1-residual @ residual/np.sum((y-y.mean())**2)),
        "posterior_df": df, "residual_variance_scaled": s2,
        "t_scale_matrix_scaled": (s2*inv).tolist(),
    }
    return beta, fit


def main():
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    area_model = json.loads(BASELINE.read_text())["area_model"]
    area = np.clip(area_model["mean_cm2"]+area_model["sd_cm2"]*np.random.default_rng(43).standard_normal(DRAWS), .2*area_model["mean_cm2"], None)
    # Check that Table 2 inputs and the area sampling rule remain identical.
    areas = []
    for film in area_model["films"]:
        path = ROOT / film["source"]
        d = np.genfromtxt(path, delimiter=",", names=True)
        areas.append(float(np.median(d["C_F"][d["V_app_V"] <= -1.5]))*film["thickness_nm"]*1e-7/EPS_GAN)
    np.testing.assert_allclose(np.mean(areas), area_model["mean_cm2"], rtol=1e-14)
    np.testing.assert_allclose(np.std(areas, ddof=1), area_model["sd_cm2"], rtol=1e-14)
    regenerated_area = np.clip(area_model["mean_cm2"]+area_model["sd_cm2"]*np.random.default_rng(43).standard_normal(DRAWS), .2*area_model["mean_cm2"], None)
    np.testing.assert_array_equal(area, regenerated_area)

    results = {
        "version": "table3_markers_20261031", "scope": "Table 3 three-film results",
        "status": "current_calculation",
        "primary_window_V": [-2.5, -1.0], "draw_count": DRAWS,
        "statistical_model": "y=1/C^2=beta0+beta1*V+epsilon; Gaussian residuals; p(beta,sigma^2) proportional to 1/sigma^2; beta follows t_(n-2)",
        "summary_definition": "finite Monte Carlo sample means and equal-tail 95% credible intervals",
        "area_model": area_model, "area_seed": 43,
        "area_draws_reused_from": "regenerated with the unchanged area model and seed 43",
        "conditioning": "regression intervals condition on the extracted image coordinates; reading sensitivity is recorded separately",
        "inputs_sha256": {str(manifest_path.relative_to(PROJECT)): sha(manifest_path)},
        "series": {},
    }
    arrays = {"area_cm2": area}
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for series in SERIES:
        meta = manifest["series"][series]
        for kind in ("full", "primary", "details"):
            path = PROJECT / meta[kind+"_csv"]
            check_archived_hash(path, meta[kind+"_csv_sha256"], PROJECT)
            results["inputs_sha256"][meta[kind+"_csv"]] = sha(path)
        d = np.genfromtxt(PROJECT / meta["primary_csv"], delimiter=",", names=True)
        v, c = d["V_app_V"], d["C_F"]
        beta, fit = sample_coefficients(v, c, SEEDS[series])
        assert (beta[:, 1] < 0).all()
        vbi = -beta[:, 0]/beta[:, 1]
        k = -2/(Q*EPS_GAN*beta[:, 1]*Y_SCALE)
        n = k/area**2
        posterior = {"Vbi_V": summary(vbi), "k_NA2": summary(k), "N_cm_minus3": summary(n)}
        for name, value in (("beta_scaled", beta), ("Vbi_V", vbi), ("k_NA2", k), ("N_cm_minus3", n)):
            assert np.isfinite(value).all()
            arrays[series+"__"+name] = value
        deviation = float((np.mean(vbi)/meta["published_Vbi_V"]-1)*100)
        results["series"][series] = {
            "full_observation_count": meta["full_observation_count"],
            "primary_observation_count": len(v), "extraction_method": meta["method"],
            "primary_csv": meta["primary_csv"], "coefficient_seed": SEEDS[series],
            "published_Vbi_V": meta["published_Vbi_V"], "signed_deviation_pct": deviation,
            "fit": fit, "posterior": posterior,
            "nonnegative_slope_draw_count": int(np.sum(beta[:, 1] >= 0)),
        }
        row = {"series": series, "n_full": meta["full_observation_count"], "n_primary": len(v)}
        for name, key, scale in (("Vbi_V", "Vbi", 1), ("k_NA2", "k_1e12", 1e12), ("N_cm_minus3", "N_1e17_cm_minus3", 1e17)):
            s = posterior[name]
            row[key+"_mc_mean"] = s["mc_sample_mean"]/scale
            row[key+"_CI95_low"] = s["equal_tail_95pct_CI"][0]/scale
            row[key+"_CI95_high"] = s["equal_tail_95pct_CI"][1]/scale
        row.update(published_Vbi_V=meta["published_Vbi_V"], signed_deviation_pct=deviation, OLS_Vbi_V=fit["OLS_Vbi_V"], R_squared=fit["R_squared"])
        rows.append(row)
        print(json.dumps(row))
    results["script_sha256"] = sha(Path(__file__))
    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    results["baseline_calibration_sha256"] = sha(BASELINE)
    (OUT / "posteriors.json").write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n")
    with (OUT / "table3_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(OUT / "posterior_draws.npz", **arrays)


if __name__ == "__main__":
    main()
