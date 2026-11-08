"""Posterior credible and predictive capacitance intervals for Table 5 and Figure 4.
All inputs and outputs are relative to this project.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np
from provenance_checks import check_archived_hash

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = DATA / "coverage_results"
ORDER = ["n1_500nm", "n1_750nm", "n2_750nm"]
Q = 1.602176634e-19
EPS = 10.6 * 8.854187817e-14  # F/cm
Y_SCALE = 1e18
VERSION = "coverage_20261102_full123"
PREDICTIVE_SEEDS = {"n1_500nm": 19100, "n1_750nm": 19101, "n2_750nm": 19103}
STABILITY_INDEX = {"n1_500nm": 0, "n1_750nm": 1, "n2_750nm": 3}
GRID_SEED = 19801
THICKNESS_CM = {"n1_500nm": 500e-7, "n1_750nm": 750e-7,
                "n2_750nm": 750e-7}
BAND_KEYS = ["C_median_F", "C_credible_lower_F", "C_credible_upper_F",
             "C_predictive_median_F", "C_predictive_lower_F", "C_predictive_upper_F",
             "relative_halfwidth_credible", "relative_halfwidth_predictive"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regenerate_beta_sigma(fit, seed, draw_count):
    """Reproduce the calibration RNG stream to recover paired sigma draws."""
    df = fit["posterior_df"]
    s2 = fit["residual_variance_scaled"]
    bhat = np.asarray(fit["OLS_beta_scaled"])
    inv = np.asarray(fit["t_scale_matrix_scaled"])/s2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((draw_count, 2)) @ np.linalg.cholesky(inv).T
    chi2 = rng.chisquare(df, draw_count)
    scale = np.sqrt(s2*df/chi2)
    return bhat+z*scale[:, None], scale


def intervals(v, vbi, k, area, beta, sigma, thickness_cm, rng):
    """Pointwise intervals from paired joint draws; no secant or smoothness remainder."""
    v = np.atleast_1d(v)
    n_density = k/area**2
    transition = Q*n_density*thickness_cm**2/(2*EPS)
    values = {key: [] for key in BAND_KEYS}
    diagnostics = {"invalid_model_draw_count": [], "nonpositive_predictive_y_count": [],
                   "minimum_transition_margin_V": []}
    for x in v:
        b = vbi-x
        valid = np.isfinite(k) & np.isfinite(area) & np.isfinite(b) & np.isfinite(transition) & (k > 0) & (area > 0) & (b > 0) & (b < transition)
        if not valid.all():
            raise RuntimeError(f"Invalid physical model draws at V={x}: {np.sum(~valid)}; no samples discarded")
        c_model = np.sqrt(EPS*Q*k/(2*b))
        cred_lo, c_med, cred_hi = np.quantile(c_model, [.025, .5, .975])
        y_tilde = beta[:, 0]+beta[:, 1]*x+sigma*rng.standard_normal(len(sigma))
        valid_y = np.isfinite(y_tilde) & (y_tilde > 0)
        if not valid_y.all():
            raise RuntimeError(f"Nonpositive or nonfinite predictive y at V={x}: {np.sum(~valid_y)}; no samples discarded")
        c_pred = 1/np.sqrt(Y_SCALE*y_tilde)
        pred_lo, pred_med, pred_hi = np.quantile(c_pred, [.025, .5, .975])
        point = {"C_median_F": c_med, "C_credible_lower_F": cred_lo,
                 "C_credible_upper_F": cred_hi, "C_predictive_median_F": pred_med,
                 "C_predictive_lower_F": pred_lo, "C_predictive_upper_F": pred_hi,
                 "relative_halfwidth_credible": (cred_hi-cred_lo)/(2*c_med),
                 "relative_halfwidth_predictive": (pred_hi-pred_lo)/(2*c_med)}
        for key in BAND_KEYS:
            values[key].append(float(point[key]))
        diagnostics["invalid_model_draw_count"].append(int((~valid).sum()))
        diagnostics["nonpositive_predictive_y_count"].append(int((y_tilde <= 0).sum()))
        diagnostics["minimum_transition_margin_V"].append(float(np.min(transition-b)))
    values.update({key: np.asarray(value) for key, value in diagnostics.items()})
    return values


def point_rows_for(series, observation_ids, v, c, fitted, bands):
    rows = []
    for j, i in enumerate(observation_ids):
        row = {"series": series, "observation_id": int(i), "V_app_V": float(v[i]),
               "C_observed_F": float(c[i]), "used_in_primary_fit": int(fitted[i])}
        for key in BAND_KEYS:
            row[key] = float(bands[key][j])
        row["inside_credible"] = int(bands["C_credible_lower_F"][j] <= c[i] <= bands["C_credible_upper_F"][j])
        row["inside_predictive"] = int(bands["C_predictive_lower_F"][j] <= c[i] <= bands["C_predictive_upper_F"][j])
        row["relative_obs_minus_model"] = float(c[i]/bands["C_median_F"][j]-1)
        row["invalid_model_draw_count"] = int(bands["invalid_model_draw_count"][j])
        row["nonpositive_predictive_y_count"] = int(bands["nonpositive_predictive_y_count"][j])
        row["minimum_transition_margin_V"] = float(bands["minimum_transition_margin_V"][j])
        rows.append(row)
    return rows


def summarize(series, evaluation_set, rows):
    return {"series": series, "evaluation_set": evaluation_set, "n_test_points": len(rows),
            "inside_credible": int(sum(r["inside_credible"] for r in rows)),
            "coverage_credible": float(np.mean([r["inside_credible"] for r in rows])),
            "inside_predictive": int(sum(r["inside_predictive"] for r in rows)),
            "coverage_predictive": float(np.mean([r["inside_predictive"] for r in rows])),
            "mean_relative_halfwidth_credible": float(np.mean([r["relative_halfwidth_credible"] for r in rows])),
            "mean_relative_halfwidth_predictive": float(np.mean([r["relative_halfwidth_predictive"] for r in rows])),
            "median_relative_halfwidth_credible": float(np.median([r["relative_halfwidth_credible"] for r in rows])),
            "median_relative_halfwidth_predictive": float(np.median([r["relative_halfwidth_predictive"] for r in rows]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    current_path = DATA/"CURRENT_TABLE3.json"
    current = json.loads(current_path.read_text())
    draws_path = PROJECT/current["draws_npz"]
    assert sha(draws_path) == current["draws_npz_sha256"]
    assert sha(PROJECT/current["results_json"]) == current["results_json_sha256"]
    arrays = np.load(draws_path)
    posteriors = json.loads((PROJECT/current["results_json"]).read_text())
    inputs = {str(current_path.relative_to(PROJECT)): sha(current_path),
              current["draws_npz"]: sha(draws_path),
              current["results_json"]: sha(PROJECT/current["results_json"])}
    all_points, membership, stability, segments = [], [], [], []
    series_details, cache = {}, {}
    for si, series in enumerate(ORDER):
        meta = current["series"][series]
        for kind in ("full", "primary"):
            path = PROJECT/meta[kind+"_csv"]
            check_archived_hash(path, meta[kind+"_csv_sha256"], PROJECT)
            inputs[meta[kind+"_csv"]] = sha(path)
        full = np.genfromtxt(PROJECT/meta["full_csv"], delimiter=",", names=True)
        primary = np.genfromtxt(PROJECT/meta["primary_csv"], delimiter=",", names=True)
        v, c = full["V_app_V"], full["C_F"]
        fitted = np.isin(v, primary["V_app_V"])
        assert fitted.sum() == len(primary)
        np.testing.assert_array_equal(c[fitted], primary["C_F"])
        vbi, k, area = arrays[series+"__Vbi_V"], arrays[series+"__k_NA2"], arrays["area_cm2"]
        saved_beta = arrays[series+"__beta_scaled"]
        fit = posteriors["series"][series]["fit"]
        seed = posteriors["series"][series]["coefficient_seed"]
        beta, sigma = regenerate_beta_sigma(fit, seed, len(vbi))
        np.testing.assert_allclose(beta, saved_beta, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(vbi, -beta[:, 0]/beta[:, 1], rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(k, -2/(Q*EPS*beta[:, 1]*Y_SCALE), rtol=2e-13, atol=2e-13)
        ids = np.arange(len(v))
        assert len(np.unique(v)) == len(v), "Observation voltages must be unique"
        series_segments = []
        # Contiguous membership runs in sorted observation order, not a min/max
        # range that would hide disconnected non-fitting voltage segments.
        sorted_ids = np.argsort(v)
        runs = np.split(sorted_ids, np.flatnonzero(np.diff(fitted[sorted_ids].astype(int)))+1)
        for segment_id, run in enumerate(runs, 1):
            segment = dict(series=series, segment_id=segment_id,
                           evaluation_set="fitting" if fitted[run[0]] else "out-of-window",
                           used_in_primary_fit=int(fitted[run[0]]), n_points=len(run),
                           V_min_V=float(v[run].min()), V_max_V=float(v[run].max()),
                           first_observation_id=int(run[0]), last_observation_id=int(run[-1]))
            series_segments.append(segment)
            segments.append(segment)
        bands = intervals(v[ids], vbi, k, area, beta, sigma, THICKNESS_CM[series],
                          np.random.default_rng(PREDICTIVE_SEEDS[series]))
        rows = point_rows_for(series, ids, v, c, fitted, bands)
        all_points.extend(rows)
        for i in range(len(v)):
            membership.append({"series": series, "observation_id": i, "V_app_V": float(v[i]),
                               "C_F": float(c[i]), "used_in_primary_fit": int(fitted[i]),
                               "included_in_coverage": 1,
                               "reason": "all_observed_points",
                               "evaluation_set": "fitting" if fitted[i] else "out-of-window"})
        fit_rows = [r for r in rows if r["used_in_primary_fit"]]
        out_rows = [r for r in rows if not r["used_in_primary_fit"]]
        series_details[series] = {
            "full_observation_count": len(v), "primary_fit_count": len(primary),
            "posterior_draw_count": len(vbi), "predictive_seed": PREDICTIVE_SEEDS[series],
            "voltage_segments": series_segments, "evaluated_V_range": [float(v[ids].min()), float(v[ids].max())],
            "selected_observation_ids": ids.tolist(), "fitting_summary": summarize(series, "fitting", fit_rows),
            "out_of_window_summary": summarize(series, "out-of-window", out_rows),
            "invalid_model_draw_count": int(sum(r["invalid_model_draw_count"] for r in rows)),
            "nonpositive_predictive_y_count": int(sum(r["nonpositive_predictive_y_count"] for r in rows)),
            "negative_credible_lower_bound_count": int(sum(r["C_credible_lower_F"] < 0 for r in rows)),
            "negative_predictive_lower_bound_count": int(sum(r["C_predictive_lower_F"] < 0 for r in rows)),
            "minimum_transition_margin_V": float(min(r["minimum_transition_margin_V"] for r in rows)),
            "mean_out_of_window_relative_obs_minus_model": float(np.mean([r["relative_obs_minus_model"] for r in out_rows])),
            "missed_observation_ids_credible": [r["observation_id"] for r in rows if not r["inside_credible"]],
            "missed_observation_ids_predictive": [r["observation_id"] for r in rows if not r["inside_predictive"]],
        }
        # Five disjoint posterior subsets check Monte Carlo sensitivity.
        for batch, idx in enumerate(np.array_split(np.arange(len(vbi)), 5), 1):
            sub = intervals(v[ids], vbi[idx], k[idx], area[idx], beta[idx], sigma[idx],
                            THICKNESS_CM[series], np.random.default_rng(22000+100*STABILITY_INDEX[series]+batch))
            subrows = point_rows_for(series, ids, v, c, fitted, sub)
            for evaluation_set, selected_rows in (("fitting", [r for r in subrows if r["used_in_primary_fit"]]),
                                                  ("out-of-window", [r for r in subrows if not r["used_in_primary_fit"]])):
                s = summarize(series, evaluation_set, selected_rows)
                stability.append({"series": series, "evaluation_set": evaluation_set, "batch": batch,
                                  "draw_count": len(idx), "predictive_seed": 22000+100*STABILITY_INDEX[series]+batch, "coverage_credible": s["coverage_credible"],
                                  "coverage_predictive": s["coverage_predictive"],
                                  "mean_relative_halfwidth_credible": s["mean_relative_halfwidth_credible"],
                                  "mean_relative_halfwidth_predictive": s["mean_relative_halfwidth_predictive"]})
        cache[series] = (v[ids], vbi, k, area, beta, sigma)
        print(json.dumps(series_details[series]["fitting_summary"]), flush=True)
        print(json.dumps(series_details[series]["out_of_window_summary"]), flush=True)

    assert len(all_points) == len(membership) == 123
    assert sum(r["used_in_primary_fit"] for r in all_points) == 43
    assert len({(r["series"], r["observation_id"]) for r in all_points}) == len(all_points)
    summaries = []
    for series in ORDER+["pooled"]:
        rows = all_points if series == "pooled" else [r for r in all_points if r["series"] == series]
        for evaluation_set, selected_rows in (("fitting", [r for r in rows if r["used_in_primary_fit"]]),
                                              ("out-of-window", [r for r in rows if not r["used_in_primary_fit"]])):
            summaries.append(summarize(series, evaluation_set, selected_rows))
    write_csv(OUT/"table5_summary.csv", summaries)
    write_csv(OUT/"coverage_points.csv", all_points)
    write_csv(OUT/"point_membership.csv", membership)
    write_csv(OUT/"voltage_segments.csv", segments)
    write_csv(OUT/"sampling_stability.csv", stability)

    vt, vbi, k, area, beta, sigma = cache[ORDER[0]]
    grid = np.linspace(vt.min(), vt.max(), 181)
    grid_bands = intervals(grid, vbi, k, area, beta, sigma, THICKNESS_CM[ORDER[0]],
                           np.random.default_rng(GRID_SEED))
    write_csv(OUT/"figure4_n1_500_grid.csv",
              [dict(V_app_V=float(x), **{key: float(grid_bands[key][j]) for key in BAND_KEYS})
               for j, x in enumerate(grid)])
    pooled_fit, pooled_out = summaries[-2], summaries[-1]
    results = {
        "version": VERSION, "status": "current_calculation",
        "source_data_version": current["version"],
        "evaluation_rule": "Every digitized observation in each full CSV is evaluated exactly once. Actual primary CSV membership defines fitting versus non-fitting groups; no voltage-grid filtering is used",
        "credible_interval_definition": "pointwise equal-tail 95% of C(V;Vbi,k)=sqrt(eps*q*k/(2*(Vbi-V))) from paired joint posterior draws",
        "predictive_interval_definition": "pointwise equal-tail 95% of C_tilde=(y_tilde*1e18)^(-1/2), y_tilde=beta0+beta1*V+sigma*z from paired beta,sigma draws",
        "relative_halfwidth_definition": "(upper-lower)/(2*posterior median model capacitance) for both interval types",
        "physical_domain": "finite k>0, area>0, and 0<Vbi-V<V_T using paired area draws; any invalid model or nonpositive/nonfinite predictive y aborts rather than truncating samples",
        "physical_domain_scope": "Checks cover the finite saved joint draws, not the entire support of the untruncated Student-t posterior",
        "joint_draws_reused": True, "draw_count_per_film": len(area),
        "point_count": len(all_points), "fit_count": pooled_fit["n_test_points"],
        "out_of_window_count": pooled_out["n_test_points"], "samples_discarded": 0,
        "h_used_in_interval_construction": False, "predictive_seeds": PREDICTIVE_SEEDS,
        "figure_grid_predictive_seed": GRID_SEED,
        "inclusion_rate_interpretation": "Descriptive observed-point inclusion. Fitting points reuse calibration observations; non-fitting points are other points of the same digitized curves, not an independent experimental validation. Model credible-band inclusion is not nominal predictive calibration",
        "includes_primary_fit_points": True, "independent_holdout_validation": False,
        "series": series_details,
        "pooled": {"fitting": pooled_fit, "out_of_window": pooled_out},
        "inputs_sha256": inputs, "script_sha256": sha(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    write_json(OUT/"coverage_results.json", results)
    preview = ["补充结果：联合后验可信区间与预测区间", "补充直接区间计算；论文表5采用割线构造。", "",
               "薄膜 | 评价集合 | 点数 | 可信区间观测落入率 | 预测区间观测落入率 | 可信RHW | 预测RHW"]
    for row in summaries:
        preview.append(f"{row['series']} | {row['evaluation_set']} | {row['n_test_points']} | {100*row['coverage_credible']:.1f}% | {100*row['coverage_predictive']:.1f}% | {100*row['mean_relative_halfwidth_credible']:.3f}% | {100*row['mean_relative_halfwidth_predictive']:.3f}%")
    preview += ["", "计算口径：", "1. 复用表3主窗口的20万组联合后验，不重新拟合。",
                "2. 评价全部123个数字化点；按实际主拟合CSV成员分为43个拟合点和80个非主拟合点，h不参与评价点筛选或这两类区间构造。",
                "3. 可信区间直接来自联合参数样本的逐点电容；预测区间进一步加入变换空间的新观测噪声。",
                "4. 两类相对半宽统一以后验中位数模型电容为分母；按实际成员分组，非主拟合点不构成独立实验验证。",
                "5. 本次全部评价点均无无效物理样本、非正预测值或负电容下界。"]
    (OUT/"表5预览与计算说明.txt").write_text("\n".join(preview)+"\n")
    outputs = {str(path.relative_to(PROJECT)): sha(path) for path in sorted(OUT.iterdir())
               if path.is_file() and path.name in {"table5_summary.csv", "coverage_points.csv", "point_membership.csv", "voltage_segments.csv", "sampling_stability.csv", "figure4_n1_500_grid.csv", "coverage_results.json", "表5预览与计算说明.txt"}}
    write_json(OUT/"manifest.json", {"version": VERSION, "status": results["status"],
               "inputs_sha256": inputs, "script_sha256": sha(Path(__file__)), "outputs_sha256": outputs})
    pointer = DATA/"CURRENT_COVERAGE.json"
    state = {"version": VERSION, "status": results["status"],
             "directory": str(OUT.relative_to(PROJECT)), "source": current["draws_npz"],
             "source_sha256": sha(draws_path), "results_json": str((OUT/"coverage_results.json").relative_to(PROJECT)),
             "results_json_sha256": sha(OUT/"coverage_results.json"),
             "table5_csv": str((OUT/"table5_summary.csv").relative_to(PROJECT)),
             "code": "code/recalculate_coverage.py", "figure_code": "paper/make_paper_figs.py"}
    write_json(pointer, state)
    print(json.dumps(pooled_fit), flush=True)
    print(json.dumps(pooled_out), flush=True)


if __name__ == "__main__":
    main()
