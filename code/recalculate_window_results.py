"""Current 2026-10-31 package: recalculate_window_results.py
All inputs and outputs are relative to this project. No external repair directory is used.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform

import numpy as np
from provenance_checks import check_archived_hash

from recalculate_table3 import DRAWS, SEEDS, sample_coefficients, summary

PROJECT = Path(__file__).resolve().parents[1]
WINDOWS = [
    ("shifted", -2.0, -0.5),
    ("primary", -2.5, -1.0),
    ("deep", -3.0, -1.5),
    ("near_zero", -1.0, 0.0),
]
SERIES = ["n1_500nm", "n1_750nm", "n2_750nm"]
BATCHES = 10
VERSION = "windows_20261031"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_summary(beta):
    values = -beta[:, 0]/beta[:, 1]
    assert np.isfinite(values).all()
    return {
        **summary(values),
        "median": float(np.median(values)),
        "nonnegative_slope_draws": int(np.sum(beta[:, 1] >= 0)),
        "min_draw": float(values.min()), "max_draw": float(values.max()),
    }


def main():
    current_path = PROJECT/"data/CURRENT_TABLE3.json"
    current = json.loads(current_path.read_text())
    paths = {key: PROJECT/current[key] for key in
             ("extraction_manifest", "results_json", "draws_npz")}
    for key, path in paths.items():
        assert sha(path) == current[key+"_sha256"], path
    extraction = json.loads(paths["extraction_manifest"].read_text())
    primary = json.loads(paths["results_json"].read_text())
    saved = np.load(paths["draws_npz"])
    output = PROJECT/current["data_directory"]/"window_results"
    output.mkdir(exist_ok=True)
    inputs = {str(path.relative_to(PROJECT)): sha(path) for path in paths.values()}
    datasets = {}
    for series in SERIES:
        meta = extraction["series"][series]
        for key in ("full_csv", "primary_csv"):
            path = PROJECT/meta[key]
            check_archived_hash(path, meta[key+"_sha256"], PROJECT)
            inputs[meta[key]] = sha(path)
        datasets[series] = np.genfromtxt(PROJECT/meta["full_csv"], delimiter=",", names=True)

    results = {
        "version": VERSION, "source_data_version": current["version"],
        "scope": "Table 4 four windows for three partial-depletion films",
        "status": "current_calculation",
        "draws_per_fit": DRAWS,
        "model": primary["statistical_model"],
        "summary_definition": primary["summary_definition"],
        "endpoint_rule": extraction["endpoint_rule"],
        "source_coordinates_modified": False,
        "primary_draws_reused": True,
        "draws_filtered_or_truncated": False,
        "stability_batches": BATCHES,
        "stability_seed_rule": "base seed + batch index * 1000000; each batch has 200000 draws",
        "series_order": SERIES,
        "windows": [], "inputs_sha256": inputs,
    }
    detail_rows, membership_rows, stability_rows, table_rows = [], [], [], []
    arrays = {}
    for window_index, (window_id, lo, hi) in enumerate(WINDOWS):
        window = {"id": window_id, "nominal_bounds_V": [lo, hi], "series": {}}
        table_row = {"window_id": window_id, "V_min_nominal": lo, "V_max_nominal": hi}
        for series in SERIES:
            meta = extraction["series"][series]
            data = datasets[series]
            tolerance = meta["half_pixel_endpoint_tolerance_V"]
            ids = np.flatnonzero((data["V_app_V"] >= lo-tolerance) &
                                (data["V_app_V"] <= hi+tolerance))
            v, c = data["V_app_V"][ids], data["C_F"][ids]
            assert len(v) > 2
            seed = SEEDS[series] if window_id == "primary" else SEEDS[series]+10000*(window_index+1)
            regenerated, fit = sample_coefficients(v, c, seed)
            if window_id == "primary":
                assert ids.tolist() == meta["primary_observation_ids"]
                subset = np.genfromtxt(PROJECT/meta["primary_csv"], delimiter=",", names=True)
                np.testing.assert_array_equal(v, subset["V_app_V"])
                np.testing.assert_array_equal(c, subset["C_F"])
                beta = saved[series+"__beta_scaled"]
                np.testing.assert_array_equal(regenerated, beta)
                np.testing.assert_array_equal(-beta[:, 0]/beta[:, 1], saved[series+"__Vbi_V"])
                assert summary(-beta[:, 0]/beta[:, 1]) == primary["series"][series]["posterior"]["Vbi_V"]
            else:
                beta = regenerated
            stats = draw_summary(beta)
            arrays[window_id+"__"+series+"__beta_scaled"] = beta
            batches = []
            for batch in range(BATCHES):
                batch_seed = seed+batch*1000000
                other = beta if batch == 0 else sample_coefficients(v, c, batch_seed)[0]
                bs = draw_summary(other)
                batch_row = {
                    "window_id": window_id, "series": series, "batch": batch,
                    "seed": batch_seed, "draws": DRAWS,
                    "Vbi_mc_mean_V": bs["mc_sample_mean"], "Vbi_median_V": bs["median"],
                    "CI95_low_V": bs["equal_tail_95pct_CI"][0],
                    "CI95_high_V": bs["equal_tail_95pct_CI"][1],
                    "nonnegative_slope_draws": bs["nonnegative_slope_draws"],
                    "minimum_draw_V": bs["min_draw"], "maximum_draw_V": bs["max_draw"],
                }
                batches.append(batch_row)
                stability_rows.append(batch_row)
            means = [row["Vbi_mc_mean_V"] for row in batches]
            medians = [row["Vbi_median_V"] for row in batches]
            stability = {
                "mean_min_V": min(means), "mean_max_V": max(means),
                "mean_range_V": max(means)-min(means),
                "median_min_V": min(medians), "median_max_V": max(medians),
                "mean_range_above_0_01V": bool(max(means)-min(means) > .01),
                "observed_batches_only_not_a_convergence_guarantee": True,
            }
            published = meta["published_Vbi_V"]
            window["series"][series] = {
                "n_observations": len(ids), "source_observation_ids": ids.tolist(),
                "actual_range_V": [float(v.min()), float(v.max())],
                "endpoint_tolerance_V": tolerance, "seed": seed,
                "fit": fit, "Vbi_V": stats, "stability": stability,
                "published_Vbi_V": published,
                "absolute_deviation_pct": abs(stats["mc_sample_mean"]-published)/published*100,
                "published_in_CI95": bool(stats["equal_tail_95pct_CI"][0] <= published <= stats["equal_tail_95pct_CI"][1]),
            }
            detail = {
                "window_id": window_id, "series": series,
                "nominal_V_min": lo, "nominal_V_max": hi,
                "n_observations": len(ids), "actual_V_min": float(v.min()), "actual_V_max": float(v.max()),
                "endpoint_tolerance_V": tolerance, "seed": seed, "draws": DRAWS,
                "Vbi_mc_mean_V": stats["mc_sample_mean"], "Vbi_median_V": stats["median"],
                "CI95_low_V": stats["equal_tail_95pct_CI"][0], "CI95_high_V": stats["equal_tail_95pct_CI"][1],
                "OLS_Vbi_V": fit["OLS_Vbi_V"], "R_squared": fit["R_squared"],
                "posterior_df": fit["posterior_df"], "published_Vbi_V": published,
                "nonnegative_slope_draws": stats["nonnegative_slope_draws"],
                "batch_mean_min_V": min(means), "batch_mean_max_V": max(means),
                "mean_stability_note": "batch range > 0.01 V; inspect stability results" if stability["mean_range_above_0_01V"] else "",
            }
            detail_rows.append(detail)
            table_row[series+"_Vbi_mc_mean_V"] = stats["mc_sample_mean"]
            for obs_id, vv, cc in zip(ids, v, c):
                membership_rows.append({"window_id": window_id, "series": series,
                                        "source_observation_id": int(obs_id), "V_app_V": float(vv), "C_F": float(cc)})
        results["windows"].append(window)
        table_rows.append(table_row)

    by_id = {w["id"]: w for w in results["windows"]}
    results["four_window_mean_ranges_V"] = {
        s: max(w["series"][s]["Vbi_V"]["mc_sample_mean"] for w in results["windows"])-
           min(w["series"][s]["Vbi_V"]["mc_sample_mean"] for w in results["windows"])
        for s in SERIES
    }
    results["script_sha256"] = sha(Path(__file__))
    results["sampler_script_sha256"] = sha(PROJECT/"code/recalculate_table3.py")
    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    write_json(output/"posteriors.json", results)
    write_csv(output/"table4_values.csv", table_rows)
    write_csv(output/"window_details.csv", detail_rows)
    write_csv(output/"window_point_membership.csv", membership_rows)
    write_csv(output/"sampling_stability.csv", stability_rows)
    np.savez_compressed(output/"posterior_draws.npz", **arrays)
    note = """三组数据的四窗口重算结果（2026-11-02）
用于论文表4。三个系列采用与1102相同的四个窗口、半像素端点容差、校准坐标和随机种子。
主窗口复用表3的200000组联合样本；另外三个窗口各用200000次抽样。
window_details.csv含12项结果，sampling_stability.csv含每项10个独立种子批次。
中心量为固定批次Monte Carlo样本均值；未删极端样本、未截断斜率。
深窗口中 n2 750 nm 的观测范围仅约[-2,-1.5] V，共6点；实际范围及批次稳定性另列。
"""
    (output/"结果说明.txt").write_text(note)
    files = ["posteriors.json", "table4_values.csv", "window_details.csv", "window_point_membership.csv",
             "sampling_stability.csv", "posterior_draws.npz", "结果说明.txt"]
    manifest = {"version": VERSION, "source_data_version": current["version"],
                "files_sha256": {f: sha(output/f) for f in files}, "inputs_sha256": inputs,
                "script_sha256": results["script_sha256"], "sampler_script_sha256": results["sampler_script_sha256"]}
    write_json(output/"manifest.json", manifest)
    write_json(PROJECT/"data/CURRENT_WINDOWS.json", {
        "version": VERSION, "source_data_version": current["version"],
        "status": "current_calculation", "directory": str(output.relative_to(PROJECT)),
        "manifest": str((output/"manifest.json").relative_to(PROJECT)),
        "manifest_sha256": sha(output/"manifest.json"),
        "table4_values": str((output/"table4_values.csv").relative_to(PROJECT)),
        "detailed_results": str((output/"posteriors.json").relative_to(PROJECT)),
        "code": "code/recalculate_window_results.py",
    })
    print(json.dumps(table_rows, indent=2))
    print("Outputs:", output)


if __name__ == "__main__":
    main()
