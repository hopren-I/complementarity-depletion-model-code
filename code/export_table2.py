"""Export the four Table 2 plateau measurements from the packaged CSVs."""
from pathlib import Path
import csv,json,numpy as np
P=Path(__file__).resolve().parents[1]
a=json.loads((P/"data/baseline_calibration.json").read_text())["area_model"]
rows=[]
for f in a["films"]:
    d=np.genfromtxt(P/f["source"],delimiter=",",names=True)
    c=float(np.median(d["C_F"][d["V_app_V"]<=-1.5]))
    area=c*f["thickness_nm"]*1e-7/(10.6*8.854187817e-14)
    np.testing.assert_allclose(area,f["area_cm2"],rtol=1e-14)
    rows.append(dict(series=Path(f["source"]).stem,thickness_nm=f["thickness_nm"],median_capacitance_F=c,area_cm2=area))
with (P/"data/results/table2_summary.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
