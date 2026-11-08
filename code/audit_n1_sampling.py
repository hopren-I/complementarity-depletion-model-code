"""Current 2026-10-31 package: audit_n1_sampling.py
All inputs and outputs are relative to this project. No external repair directory is used.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

from prepare_table3_data import color_mask

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT
OUT = PROJECT / "data/digitization/n1_sampling_audit"
SOURCE = PROJECT / "data/digitization"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(v, c):
    x = np.column_stack([np.ones(len(v)), v])
    y = 1 / c**2 / 1e18
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    res = y - x @ beta
    return {"n": len(v), "OLS_Vbi_V": float(-beta[0]/beta[1]),
            "R_squared": float(1 - res @ res / np.sum((y-y.mean())**2))}


def isolated_components(mask, size_range):
    labs, count = label(mask, np.ones((3, 3)))
    components = []
    for k in range(1, count+1):
        ys, xs = np.where(labs == k)
        if size_range[0] <= len(xs) <= size_range[1]:
            components.append((xs, ys))
    return sorted(components, key=lambda a: a[0].mean())


def centroids(mask, centers, width, shift=0):
    ys, xs = np.where(mask)
    pts = []
    for center in centers+shift:
        keep = (xs >= center-width/2) & (xs < center+width/2)
        assert keep.sum() > 2
        pts.append([xs[keep].mean(), ys[keep].mean()])
    return np.array(pts)


def circle_contour_diagnostic(mask, cx):
    """Fit free centers to a union of ellipses; no equal-spacing penalty.

    Ellipse size comes from ten isolated markers. The existing cell centers
    initialize the overlapping segment. Its marker count is NOT independently
    inferred by this fit. Multiple starts expose dependence on initialization.
    """
    comps = isolated_components(mask, (35, 80))
    observations = []
    for xs, ys in comps:
        xx = np.unique(xs)
        observations.append((xx, np.array([ys[xs == x].min()-.5 for x in xx]),
                             np.array([ys[xs == x].max()+.5 for x in xx])))
    p0 = np.ravel([[xs.mean(), ys.mean()] for xs, ys in comps]).tolist()+[3.65, 4.7]

    def isolated_residual(par):
        rx, ry = par[-2:]
        residual = []
        for j, (xx, top, bottom) in enumerate(observations):
            x, y = par[2*j:2*j+2]
            off = ry*np.sqrt(np.maximum(1-((xx-x)/rx)**2, 1e-8))
            residual.extend(y-off-top)
            residual.extend(y+off-bottom)
        return residual

    allowance = np.ones(len(p0))
    shape_fit = least_squares(isolated_residual, p0,
                              bounds=(np.array(p0)-allowance, np.array(p0)+allowance))
    rx, ry = shape_fit.x[-2:]
    nominal = np.arange(-28, -6)/10
    grid = (nominal-cx[1])/cx[0]
    initial = centroids(mask, grid, .1/cx[0])
    xx = np.arange(int(np.ceil(grid[1])), int(np.floor(grid[-2]))+1)
    top = np.array([np.flatnonzero(mask[:, x]).min()-.5 for x in xx])
    bottom = np.array([np.flatnonzero(mask[:, x]).max()+.5 for x in xx])

    def residual(par):
        xy = par.reshape(-1, 2)
        dx = (xx[:, None]-xy[None, :, 0])/rx
        covered = np.abs(dx) <= 1
        off = ry*np.sqrt(np.maximum(1-dx**2, 0))
        upper = np.min(np.where(covered, xy[:, 1]-off, 1e3), axis=1)
        lower = np.max(np.where(covered, xy[:, 1]+off, -1e3), axis=1)
        return np.r_[upper-top, lower-bottom]

    lower = (initial-np.array([2.4, 3])).ravel()
    upper = (initial+np.array([2.4, 3])).ravel()
    solutions = []
    for offset in [-1, 0, 1]:
        start = initial.copy()
        start[:, 0] += offset
        sol = least_squares(residual, start.ravel(), bounds=(lower, upper),
                            max_nfev=500, diff_step=1e-4)
        solutions.append(sol)
    best = min(solutions, key=lambda s: s.cost)
    main = (nominal >= -2.5) & (nominal <= -1)
    points = best.x.reshape(-1, 2)[main]
    return points, {
        "model": "union of ellipses, size learned from isolated markers; free centers, fixed count",
        "not_an_independent_marker_count": True,
        "ellipse_radii_px": [float(rx), float(ry)],
        "isolated_boundary_RMSE_px": float(np.sqrt(np.mean(shape_fit.fun**2))),
        "overlapping_boundary_RMSE_px": float(np.sqrt(np.mean(best.fun**2))),
        "multi_start_boundary_RMSE_px": [float(np.sqrt(np.mean(s.fun**2))) for s in solutions],
        "primary_max_grid_center_offset_px": float(np.max(np.abs(points[:, 0]-grid[main]))),
        "primary_fitted_spacing_V": float(np.polyfit(np.arange(len(points)),
                                                     np.polyval(cx, points[:, 0]), 1)[0]),
        "center_bounds_px": [2.4, 3.0],
        "points_px": points.tolist(),
    }


def main():
    OUT.mkdir(exist_ok=True)
    image_path = SOURCE / "source_figure4.png"
    image = np.asarray(Image.open(image_path).convert("RGB"))
    cal = json.loads((SOURCE / "calibration.json").read_text())["b"]
    cx, cy = np.array(cal["x_affine"]), np.array(cal["y_affine"])
    nominal = np.arange(-25, -9)/10
    grid = (nominal-cx[1])/cx[0]
    step = .1/cx[0]
    result = {
        "scope": "0.1 V n1 digitization-cell audit for the Table 3 primary window",
        "source_image_sha256": sha(image_path),
        "pixel_scale_V": float(cx[0]), "cell_width_px": float(step),
        "primary_nominal_V": nominal.tolist(),
        "reference_text": "No voltage sampling increment is specified in the source paper text. Its reported D<0.1 is a dissipation factor, not a voltage increment.",
        "series": {},
    }
    visual = []
    for series, color, size_range in [("n1_500nm", "green", (35, 80)),
                                       ("n1_750nm", "maroon", (20, 50))]:
        mask = color_mask(image, "b", color)
        comps = isolated_components(mask, size_range)
        isolated = np.array([[xs.mean(), ys.mean()] for xs, ys in comps])
        spacing = np.polyfit(np.arange(len(isolated)), np.polyval(cx, isolated[:, 0]), 1)[0]
        baseline = centroids(mask, grid, step)
        current = np.genfromtxt(PROJECT/f"data/gan_{series}_primary_cv.csv",
                                delimiter=",", names=True)
        v, c = np.polyval(cx, baseline[:, 0]), np.polyval(cy, baseline[:, 1])
        np.testing.assert_array_equal(v, current["V_app_V"])
        np.testing.assert_array_equal(c, current["C_F"])
        base_fit = fit(v, c)
        shifts = []
        for shift in [-1, -.5, 0, .5, 1]:
            points = centroids(mask, grid, step, shift)
            f = fit(np.polyval(cx, points[:, 0]), np.polyval(cy, points[:, 1]))
            shifts.append({"boundary_shift_px": shift, **f,
                           "delta_Vbi_V": f["OLS_Vbi_V"]-base_fit["OLS_Vbi_V"]})
        diagnostic = {
            "isolated_marker_count": len(comps),
            "isolated_markers_V_range": np.polyval(cx, isolated[[0, -1], 0]).tolist(),
            "isolated_marker_spacing_V": float(spacing),
            "isolated_markers_px": isolated.tolist(),
            "baseline": base_fit, "boundary_shift_check": shifts,
            "every_second_observation": [fit(v[phase::2], c[phase::2]) for phase in [0, 1]],
        }
        if color == "green":
            points, shape = circle_contour_diagnostic(mask, cx)
            alternative = fit(np.polyval(cx, points[:, 0]), np.polyval(cy, points[:, 1]))
            diagnostic["circle_contour_diagnostic"] = shape
            diagnostic["alternative"] = {"method": "free-center ellipse-contour diagnostic", **alternative}
        else:
            # Triangular marker width peaks are clearly resolved in this window.
            # No 0.1 V spacing or target peak count is passed to find_peaks.
            xx = np.flatnonzero(mask.any(axis=0))
            counts = mask[:, xx].sum(axis=0)
            signal = counts-savgol_filter(counts, 21, 2)
            peak_indices, _ = find_peaks(signal, prominence=.8, distance=3)
            peaks = xx[peak_indices]
            peak_v = np.polyval(cx, peaks)
            # +/- 0.05 V attaches detected markers to the two endpoint labels;
            # this diagnostic association is not the production endpoint rule.
            select = (peak_v >= -2.55) & (peak_v <= -.95)
            indices = np.flatnonzero(select)
            ys, xs = np.where(mask)
            points = []
            for j in indices:
                lo, hi = (peaks[j-1]+peaks[j])/2, (peaks[j]+peaks[j+1])/2
                keep = (xs >= lo) & (xs < hi)
                points.append([xs[keep].mean(), ys[keep].mean()])
            points = np.array(points)
            robustness = []
            for prominence in [.4, .8, 1.2, 1.6, 2]:
                inds, _ = find_peaks(signal, prominence=prominence, distance=3)
                vv = np.polyval(cx, xx[inds])
                robustness.append({"prominence_px": prominence,
                                   "primary_count": int(np.sum((vv >= -2.55) & (vv <= -.95)))})
            diagnostic["independent_width_peaks"] = {
                "primary_count": len(indices), "primary_x_px": peaks[select].tolist(),
                "spacing_V": float(np.polyfit(np.arange(len(indices)), peak_v[select], 1)[0]),
                "max_grid_center_offset_px": float(np.max(np.abs(peaks[select]-grid))),
                "prominence_sensitivity": robustness,
            }
            diagnostic["alternative"] = {"method": "centroids between independently detected marker midpoints",
                                           **fit(np.polyval(cx, points[:, 0]), np.polyval(cy, points[:, 1]))}
        diagnostic["alternative"]["delta_Vbi_V"] = diagnostic["alternative"]["OLS_Vbi_V"]-base_fit["OLS_Vbi_V"]
        result["series"][series] = diagnostic
        visual.append((series, baseline, points))
    result["interpretation"] = {
        "spacing": "0.1 V is supported by image marker spacing, not prescribed by the Bayesian or depletion model.",
        "current_digitization": "Supported as an approximate raster-digitization rule for the Table 3 primary n1 windows.",
        "not_proven": ["the instrument voltage increment", "exact original measurement coordinates", "independent measurement errors", "credible intervals including image-reading uncertainty"],
        "triangle_bbox_note": "A triangle bounding-box midpoint is not its area centroid; the earlier bbox variant is not an equivalent center convention.",
    }
    result["audit_script_sha256"] = sha(Path(__file__))
    (OUT/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")

    os.environ.setdefault("MPLCONFIGDIR", str(OUT/"mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    for ax, (series, base, alternative) in zip(axes, visual):
        ax.imshow(image, interpolation="nearest")
        for x in grid-step/2:
            ax.axvline(x, color="0.5", alpha=.4, linewidth=.6)
        ax.scatter(base[:, 0], base[:, 1], s=33, marker="+", c="black", label="Current cell centroid")
        ax.scatter(alternative[:, 0], alternative[:, 1], s=27, marker="o", facecolors="none",
                   edgecolors="#1565c0", linewidths=.8, label="Contour / adaptive check")
        ax.set_xlim(grid[0]-5, grid[-1]+5)
        ax.set_ylim(base[:, 1].max()+8, base[:, 1].min()-8)
        ax.set_title(series.replace("_", " "))
        ax.set_xlabel("Source-image x (pixels)")
        ax.set_ylabel("Source-image y (pixels)")
        ax.legend(fontsize=8, loc="upper right")
    fig.savefig(OUT/"primary_marker_check.png", dpi=200)
    plt.close(fig)
    print(json.dumps({s: {"spacing_V": d["isolated_marker_spacing_V"],
                          "baseline": d["baseline"], "alternative": d["alternative"],
                          "shift_range_V": [min(t["OLS_Vbi_V"] for t in d["boundary_shift_check"]),
                                            max(t["OLS_Vbi_V"] for t in d["boundary_shift_check"])]}
                      for s, d in result["series"].items()}, indent=2))


if __name__ == "__main__":
    main()
