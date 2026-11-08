"""Current 2026-10-31 package: prepare_table3_data.py
All inputs and outputs are relative to this project. No external repair directory is used.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import label

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT
VERSION = "table3_markers_20261031"
DATA = PROJECT / "data"
AUDIT = PROJECT / "data/digitization"
OUT = PROJECT / "data/digitization"
APPROVED = DATA  # Retain the already adopted n2-750 CSV formatting.
SPECS = [
    ("n1_500nm", "b", "green", 1.03),
    ("n1_750nm", "b", "maroon", 1.05),
    ("n2_750nm", "c", "pink", 1.80),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def color_mask(image, panel, color):
    r, g, b = image.astype(int).transpose(2, 0, 1)
    yy, xx = np.indices(r.shape)
    masks = {
        "green": (g > 90) & (g-r > 30) & (g-b > 30),
        "maroon": (r > 70) & (r < 185) & (g < 75) & (b < 75) & (r-g > 40),
        "teal": (g > 110) & (b > 110) & (r < 110),
        "pink": (r > 170) & (b > 110) & (g < 120) & (r-g > 80),
    }
    m = masks[color]
    if panel == "b":
        m &= (xx >= 577) & (xx <= 852) & (yy >= 50) & (yy <= 360)
        m &= ~((xx < 690) & (yy < 140))  # Exclude the legend.
    else:
        m &= (xx >= 1000) & (xx < 1350)
        if color == "pink":
            m &= yy >= 260
        else:
            m &= (yy >= 65) & (yy <= 218) & ~((xx < 1150) & (yy < 135))
    return m


def ols(v, c):
    x = np.column_stack((np.ones(len(v)), v))
    y = 1/c**2/1e18
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(-beta[0]/beta[1])


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    image_path = AUDIT / "source_figure4.png"
    image = np.asarray(Image.open(image_path).convert("RGB"))
    cal = json.loads((AUDIT / "calibration.json").read_text())
    manifest = {
        "version": VERSION,
        "scope": "three partial-depletion series for Table 3 only",
        "source_pdf": "data/source_evidence/chandrasekar2015.pdf",
        "source_pdf_sha256": sha(PROJECT / "data/source_evidence/chandrasekar2015.pdf"),
        "source_raster": str(image_path.relative_to(ROOT)),
        "source_raster_sha256": sha(image_path),
        "source_raster_size": [image.shape[1], image.shape[0]],
        "source_pdf_page_index": 4,
        "calibration": {key: cal[key] for key in ("b", "c")},
        "primary_window_V": [-2.5, -1.0],
        "endpoint_rule": "closed interval with half a source x pixel tolerance; preserve calibrated coordinates",
        "n1_extraction_rule": "one colored-pixel centroid per 0.1 V cell, centers -5..0 V fixed from axis ticks and visible marker spacing; overlapping marks are reconstructed cells",
        "n2_extraction_rule": "one unweighted colored-pixel centroid per isolated marker component",
        "fit_results_used_for_extraction": False,
        "series": {},
    }
    details_all, sensitivity, plotdata = [], [], []
    for series, panel, color, published in SPECS:
        cx = np.array(cal[panel]["x_affine"])
        cy = np.array(cal[panel]["y_affine"])
        tol = abs(cx[0])/2
        mask = color_mask(image, panel, color)
        ys, xs = np.where(mask)
        cells = []
        if panel == "c":
            labs, count = label(mask, np.ones((3, 3)))
            for j in range(1, count+1):
                sy, sx = np.where(labs == j)
                if len(sx) >= 10:
                    cells.append((sx, sy, None))
            cells.sort(key=lambda cell: cell[0].mean())
            assert len(cells) == 21, (series, len(cells))
            method = "isolated_marker_centroid"
        else:
            xstep = 0.1/cx[0]
            for nominal_v in np.arange(-50, 1)/10:
                center_x = (nominal_v-cx[1])/cx[0]
                use = (xs >= center_x-xstep/2) & (xs < center_x+xstep/2)
                # A missing/occluded cell is an error, not an invented point.
                assert use.sum() > 2, (series, nominal_v, "unresolved cell")
                cells.append((xs[use], ys[use], float(nominal_v)))
            method = "overlapping_marker_cell_centroid"
        rows, details = [], []
        for j, (sx, sy, nominal_v) in enumerate(cells):
            px, py = float(sx.mean()), float(sy.mean())
            v, c = float(np.polyval(cx, px)), float(np.polyval(cy, py))
            rows.append({"V_app_V": v, "C_F": c})
            details.append({
                "series": series, "observation_id": j, "method": method,
                "nominal_grid_V": nominal_v if nominal_v is not None else "",
                "source_x_px": px, "source_y_px": py, "V_app_V": v, "C_F": c,
                "color_pixel_count": len(sx), "bbox_xmin_px": int(sx.min()),
                "bbox_xmax_px": int(sx.max()), "bbox_ymin_px": int(sy.min()),
                "bbox_ymax_px": int(sy.max()),
                "bbox_center_V": float(np.polyval(cx, (sx.min()+sx.max())/2)),
                "bbox_center_C_F": float(np.polyval(cy, (sy.min()+sy.max())/2)),
            })
        v = np.array([r["V_app_V"] for r in rows])
        c = np.array([r["C_F"] for r in rows])
        assert np.all(np.diff(v) > 0) and np.all(c > 0)
        keep = (v >= -2.5-tol) & (v <= -1+tol)
        for detail, selected in zip(details, keep):
            detail["in_primary_window"] = int(selected)
        full_path = DATA / f"gan_{series}_cv.csv"
        primary_path = DATA / f"gan_{series}_primary_cv.csv"
        if series == "n2_750nm":
            # Preserve the already approved 21/11-point files byte for byte.
            old = np.genfromtxt(APPROVED / full_path.name, delimiter=",", names=True)
            np.testing.assert_array_equal(v, old["V_app_V"])
            np.testing.assert_array_equal(c, old["C_F"])
            primary = np.genfromtxt(APPROVED / primary_path.name, delimiter=",", names=True)
            np.testing.assert_array_equal(v[keep], primary["V_app_V"])
            np.testing.assert_array_equal(c[keep], primary["C_F"])
        else:
            save_csv(full_path, rows)
            save_csv(primary_path, [row for row, selected in zip(rows, keep) if selected])
        assert int(keep.sum()) == (16 if panel == "b" else 11)
        detail_path = DATA / f"gan_{series}_marker_details.csv"
        save_csv(detail_path, details)
        manifest["series"][series] = {
            "panel": panel, "method": method, "full_observation_count": len(v),
            "primary_observation_count": int(keep.sum()), "published_Vbi_V": published,
            "half_pixel_endpoint_tolerance_V": tol,
            "primary_observation_ids": np.flatnonzero(keep).tolist(),
            "actual_primary_range_V": [float(v[keep].min()), float(v[keep].max())],
            "full_csv": str(full_path.relative_to(PROJECT)),
            "primary_csv": str(primary_path.relative_to(PROJECT)),
            "details_csv": str(detail_path.relative_to(PROJECT)),
            "full_csv_sha256": sha(full_path), "primary_csv_sha256": sha(primary_path),
            "details_csv_sha256": sha(detail_path),
        }
        variants = {
            "centroid": (v, c),
            "bbox_center": (np.array([d["bbox_center_V"] for d in details]), np.array([d["bbox_center_C_F"] for d in details])),
        }
        for dx in (-0.5, 0.5):
            for dy in (-0.5, 0.5):
                variants[f"calibration_shift_x{dx}_y{dy}_px"] = (v+cx[0]*dx, c+cy[0]*dy)
        for name, (vv, cc) in variants.items():
            sensitivity.append({"series": series, "reading_variant": name, "primary_count": int(keep.sum()), "OLS_Vbi_V": ols(vv[keep], cc[keep])})
        details_all.extend(details)
        plotdata.append((series, panel, details, cx))
        print(series, "full", len(v), "primary", int(keep.sum()), "OLS Vbi", ols(v[keep], c[keep]))
    save_csv(DATA / "all_marker_details.csv", details_all)
    save_csv(OUT / "reading_sensitivity.csv", sensitivity)
    manifest["preparation_script_sha256"] = sha(Path(__file__))
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")

    # Internal verification images: source figure with a single cross per output.
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, (series, panel, details, cx) in zip(axes.flat, plotdata):
        ax.imshow(image, interpolation="nearest")
        px = np.array([d["source_x_px"] for d in details])
        py = np.array([d["source_y_px"] for d in details])
        ax.scatter(px, py, marker="+", c="black", s=30, linewidths=.8)
        ax.set_xlim(px.min()-10, px.max()+10)
        ax.set_ylim(py.max()+15, py.min()-15)
        ax.set_title(series+f": {len(details)} observations")
        ax.set_xlabel("Source x (px)"); ax.set_ylabel("Source y (px)")
    fig.savefig(OUT / "source_overlay_full.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, (series, panel, details, cx) in zip(axes.flat, plotdata):
        selected = [d for d in details if d["in_primary_window"]]
        px = np.array([d["source_x_px"] for d in selected]); py = np.array([d["source_y_px"] for d in selected])
        ax.imshow(image, interpolation="nearest")
        if panel == "b":
            for edge in np.arange(-25.5, -9.4)/10:
                ax.axvline((edge-cx[1])/cx[0], lw=.6, color="#555555", alpha=.5)
        ax.scatter(px, py, marker="+", c="black", s=45, linewidths=1)
        ax.set_xlim(px.min()-5, px.max()+5); ax.set_ylim(py.max()+7, py.min()-7)
        ax.set_title(series+f": {len(selected)} primary observations")
        ax.set_xlabel("Source x (px)"); ax.set_ylabel("Source y (px)")
    fig.savefig(OUT / "source_overlay_primary.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
