# A Complementarity Depletion Model for Uniformly Doped Semiconductor Films with Capacitance Bounds and Bayesian Calibration

Experimental code for the paper by Siyuan Chen and Lipu Zhang.

The experiments cover deterministic capacitance bounds for the one-dimensional
obstacle model, Bayesian parameter calibration, voltage-window sensitivity,
Si/GaN extrapolation, and InSb prediction across measurement conditions.

## Requirements

Python 3.12 or newer. Package versions are pinned in `requirements.txt`; the
reference calculations used Python 3.12.14. Numerical experiments and figure
generation do not require LaTeX. All analysis inputs are included.

## Run

From this directory on macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py
```

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproduce.py
```

On macOS or Linux, `sh setup.sh` followed by `sh run_all.sh` is an alternative.
`ECS_PYTHON=/path/to/python sh run_all.sh` selects an existing environment.
The first dependency installation requires internet access. Experiment runs
use local data.

The default command recalculates all results, checks them against the numerical
reference CSVs, and produces five figures in each language. To calculate without
rendering figures:

```sh
python reproduce.py --skip-figures
```

After a completed calculation, use:

```sh
python reproduce.py --mode figures
python reproduce.py --mode verify
```

Run the full calculation before invoking individual plotting scripts. The
repository contains input data and reference CSVs; posterior samples and other
large derived outputs are generated locally.

## Repository contents

| Path | Contents |
| --- | --- |
| `code/` | Data extraction, numerical experiments, statistical analysis, plotting, and checks |
| `data/` | Fixed inputs: digitized readings, source image and PDF, InSb archive, calibration definitions, and protocols |
| `reference_results/` | Numerical reference CSVs and figure hashes used for comparison |
| `verification/raw_inputs.json` | Checksums of the fixed inputs |
| `assets/fonts/` | Fandol plotting font, source information, and upstream license documents |

Generated files include:

- `data/results/`: Si/GaN posterior samples and parameter summaries.
- `data/window_results/`: window-specific posterior samples and summaries.
- `data/coverage_results/`: point membership and credible/predictive intervals.
- `data/secant_bridge_results/`: secant interval comparisons.
- `data/validation/`: finite-element and synthetic-experiment results.
- `data/insb_*.csv`: InSb fits, predictions, exclusions, and summaries.
- `figure/en/figure1.png` through `figure5.png`, with corresponding Chinese files
  under `figure/zh/`.
- `tables/insb_prediction_summary_en.tex` and `_zh.tex`: Table V fragments.
- `verification/`: current run logs and numerical check summaries.

`.gitignore` excludes these generated files and local environments. The
experimental input files remain included. `code/OUTPUT_MAP.txt` maps paper
figures and tables to their calculation scripts.

## Data and statistical scope

Si/GaN inputs are digitizations of published plots, not instrument records.
Three partial-depletion GaN curves are re-extracted from the supplied raster;
Si and the four area-calibration curves start from the archived digitized CSVs.
Their complete original extraction procedures are not included.

The InSb analysis reads the original measurement ZIP directly. It includes
159 devices; the reported reference-window comparison uses 127 devices and
the paired low-loss comparison uses 31 devices. Window selection is exploratory.
See `data/insb_protocol.txt` and `data/numerical_protocol.txt` for the full
analysis definitions and limitations.

The main GaN fits use 200,000 posterior draws per film. The three synthetic
noise scenarios each use 1,000 repetitions and 10,000 draws per repetition.
Seeds are fixed in the calculation scripts. Floating-point bound checks do
not constitute directed-rounding interval certificates. Reference CSVs are
comparison targets and are not fitting inputs.

Historical numbers in filenames such as `table3_summary.csv` and
`insb_table8_values.csv` identify calculation stages. The paper's tables are
numbered I–V.

## Sources

- H. Chandrasekar et al., *Estimation of background carrier concentration in
  fully depleted GaN films*, Semiconductor Science and Technology 30, 115018
  (2015). [DOI](https://doi.org/10.1088/0268-1242/30/11/115018).
- O. Vavasour et al., *Data for Effect of HCl cleaning on InSb–Al2O3 MOS
  capacitors*, University of Warwick (2019).
  [Dataset record](https://wrap.warwick.ac.uk/id/eprint/114894/).
- [Fandol fonts](https://ctan.org/pkg/fandol); original notices are retained in
  `assets/fonts/`.

Third-party source materials retain their original terms. This repository does
not assign a new license to those materials.
