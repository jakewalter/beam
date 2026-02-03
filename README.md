# BEAM — Array-Based Seismic Event Detector

A compact explanation and quick usage guide for the BEAM detector implemented in this repository.

## Overview

BEAM provides two detection modes for array-based seismic detection:
- Correlation mode (Gibbons & Ringdal-style): template-matching using normalized cross-correlation to detect repeating events.
- Traditional (STA/LTA beamforming): delay-and-sum beamforming across a grid of apparent velocities (slowness) and backazimuth angles, with STA/LTA detection on beam traces.

The code is designed to: process daily miniseed volumes, write per-subarray per-day results, produce per-day aggregated JSONs, triangulate pairwise intersections across subarrays, optionally run LSQ multi-array localization, and cluster/aggregate to a CSV summary.

## Key Methods & Equations

### 1) Correlation Mode
- Normalized cross-correlation:
  - $$\mathrm{corr}(t) = \frac{x \cdot y_t}{\lVert x \rVert \, \lVert y_t \rVert}$$
    - `y_t` is the sliding window of data aligned to the template time `t`.
  - Gibbons & Ringdal scaled correlation:
  - $$\mathrm{scaled}(t) = \frac{\mathrm{corr}(t)}{\mathrm{RMS}(\mathrm{corr})}$$

This yields robust picks based on shape similarity to the master template.

### 2) Traditional Beamforming (STA/LTA)
-- Delay-and-sum beamforming for a candidate slowness $s$ and backazimuth $\theta$:
  - Convert to slowness vector components:
    - $$s_x = s\sin(\theta)$$
    - $$s_y = s\cos(\theta)$$
  - Per-station delays $\tau_i$ in seconds:
    - $$\tau_i = s_x x_i + s_y y_i\quad \text{(where $x_i$, $y_i$ in km relative to array center)}$$
  - Beamformed signal:
    - $$B(t) = \frac{1}{N}\sum_{i=1}^{N} x_i\bigl(t - \tau_i\bigr)$$

-- Characteristic function (e.g., envelope):
  - $$\mathrm{envelope}(B) = \bigl|\mathcal{H}\{B\}\bigr|$$

-- STA/LTA:
  - $$\mathrm{STA}(t) = \frac{1}{T_{\mathrm{STA}}}\int_{t - T_{\mathrm{STA}}/2}^{t + T_{\mathrm{STA}}/2} |B(\tau)|\,\mathrm{d}\tau$$
  - $$\mathrm{LTA}(t) = \frac{1}{T_{\mathrm{LTA}}}\int_{t - T_{\mathrm{LTA}}/2}^{t + T_{\mathrm{LTA}}/2} |B(\tau)|\,\mathrm{d}\tau$$
  - $$R(t) = \frac{\mathrm{STA}(t)}{\mathrm{LTA}(t) + 1\times 10^{-12}}$$

### 3) FK Analysis (optional refinement)
- Frequency–Wavenumber (FK) analysis in a short window to refine apparent slowness/azimuth that maximize beam power.

### 4) Pairwise Triangulation & LSQ Localization
- Pairwise triangulation uses two arrays' centers and backazimuths to compute intersection of bearing lines.
- For multiple arrays, least-squares optimization finds (lat, lon, origin_time) minimizing time residuals of picks, optionally with azimuth/slowness constraints.

More specifically, the implementation used in this repository follows these steps and models:

- Local projection
  - Convert geographic coordinates (lat, lon) to local Cartesian coordinates (x, y) in kilometers using a simple equirectangular approximation about a chosen origin (centroid of centers):
    $$x = (\lambda - \lambda_0) \cdot 111.32\cos(\phi_0), \qquad y = (\phi - \phi_0)\cdot 110.54$$
    where $(\phi,\lambda)$ are latitude/longitude in degrees and $(\phi_0,\lambda_0)$ is the projection origin.

- Pairwise bearing intersection (two-array triangulation)
  - For two arrays at $(x_1,y_1)$ and $(x_2,y_2)$ with forward bearings (array→source) $\varphi_1$ and $\varphi_2$ (radians, measured from North clockwise), define direction unit vectors
    $$\mathbf{v}_i = (\sin\varphi_i,\;\cos\varphi_i),\quad i=1,2.$$ 
  - Parametric lines are $\mathbf{p}_i(t)=\mathbf{p}_i + t\mathbf{v}_i$. Solve for the scalar parameter $t$ using the 2×2 linear system; explicitly, with
    $$\Delta x = x_2 - x_1,\quad \Delta y = y_2 - y_1,\quad \text{denom} = \sin(\varphi_1-\varphi_2),$$
    $$t_1 = \frac{\Delta x\cos\varphi_2 - \Delta y\sin\varphi_2}{\sin(\varphi_1-\varphi_2)}$$
    and the intersection point is
    $$x^* = x_1 + t_1\sin\varphi_1,\qquad y^* = y_1 + t_1\cos\varphi_1.$$
    If the denominator is (nearly) zero the lines are parallel and no unique intersection exists.

- Least-squares multi-array localization (nonlinear LSQ)
  - Each array $i$ provides an arrival time $t_i$, backazimuth $\beta_i$ and slowness $s_i$ (s/km); let the unknown source be at position $\mathbf{r}=(x,y)$ with origin time $t_0$.
  - Predicted arrival time at array $i$ (approximate plane-wave / far-field model using per-array slowness) is
    $$\hat t_i = t_0 + s_i\,\lVert\mathbf{r}-\mathbf{r}_i\rVert,$$
    where $\mathbf{r}_i$ is the array center in local $(x,y)$ km.
  - Backazimuth provides a directional (cross-track) constraint: with unit bearing $\mathbf{u}_i$ pointing array→source, the perpendicular (cross-track) distance is
    $$\Delta_{\perp,i} = \bigl| (\mathbf{r}-\mathbf{r}_i) \times \mathbf{u}_i \bigr| = \bigl| (x-x_i)u_{i,y} - (y-y_i)u_{i,x}\bigr|.$$
  - The LSQ fitter minimizes a combined objective (sum of squared, weighted residuals):
    $$J(x,y,t_0)=\sum_i\left[\left(\frac{t_i-\hat t_i}{\sigma_t}\right)^2 + \left(\frac{\Delta_{\perp,i}}{\sigma_{\perp}}\right)^2\right],$$
    where $\sigma_t$ and $\sigma_{\perp}$ are scaling/weight parameters (TOA and directional weights).
  - This nonlinear problem is solved iteratively (Gauss–Newton / Levenberg–Marquardt via SciPy's least_squares).  The Jacobian entries used in the linearization are, for the time residuals,
    $$\frac{\partial\hat t_i}{\partial x} = s_i\frac{x-x_i}{\lVert\mathbf{r}-\mathbf{r}_i\rVert},\qquad \frac{\partial\hat t_i}{\partial y} = s_i\frac{y-y_i}{\lVert\mathbf{r}-\mathbf{r}_i\rVert},\qquad \frac{\partial\hat t_i}{\partial t_0}=1,$$
    and for directional residuals the derivatives are the signed components of the perpendicular operator (the derivative of $(x-x_i)u_{i,y}-(y-y_i)u_{i,x}$ with respect to $x,y$).
  - On convergence the optimizer returns an estimate $(x^*,y^*,t_0^*)$ which is converted back to geographic coordinates using the inverse local projection and accompanied by an approximate covariance from $(J^T J)^{-1}$.

These formulas are implemented in `beam/core/triangulation.py` (pairwise intersection) and `beam/core/locator.py` (least-squares multi-array fitter). The code uses a local equirectangular projection (suitable for regional scales) and combines TOA and directional constraints to produce robust location estimates.

### Triangulation methods: intersection vs LSQ (practical details) ✅

- **Two principal estimators are used:**
  - **Geometric intersection (pairwise)** — uses the two arrays' bearings (derived from `backazimuth`) and computes an intersection of the two lines. When bearings have uncertainty we compute a Monte Carlo distribution by perturbing bearings and reporting the median location and a spread (`std_km`). This is exposed via the CLI flags `--mc-az-sigma` (degrees) and `--mc-samples` (number of MC samples). The result is tagged `method='intersection_mc'`.
  - **Least-squares time-residual inversion (LSQ)** — uses arrival times and slowness/velocity (or a forced velocity) and minimizes time residuals (optionally with cross-track directional constraints). When successful this method returns `method='lsq_2array'` (for 2-array fallback) or `lsq_multi` for >2 arrays, and includes `residual_norm` and optional `covariance` for uncertainty estimates.

- **How they interact in the code:** the pipeline computes the geometric intersection first and then—*if* slowness/velocity info is available (or `--lsq-force-vel` is passed) and gating checks pass (velocity range, velocity tolerance, minimum average SNR)—it attempts an LSQ fit for the same pair.
  - If LSQ **succeeds**, the LSQ location **overrides** the geometric intersection result and the output is labeled as LSQ.
  - If LSQ **is not attempted or fails**, the Monte Carlo intersection result (median + `std_km`) is used and labeled `intersection_mc`.

- **CLI & config knobs (most relevant):**
  - `--mc-az-sigma <deg>` (default 3.0) — Monte Carlo azimuth sigma for `intersection_mc`
  - `--mc-samples <N>` (default 200) — number of MC draws
  - `--lsq-force-vel <km/s>` — force LSQ to use a specified velocity even if detections lack slowness
  - `--min-snr <float>` — minimum average SNR for allowing LSQ fallback
  - `--lsq-vel-min`, `--lsq-vel-max`, `--lsq-vel-tol` — gate LSQ by mean pair velocity & pairwise velocity difference
  - `--no-lsq` — disable LSQ fallback entirely (use intersection only)

- **Outputs & what to inspect:**
  - `method` field: `intersection_mc` or `lsq_2array`/`lsq_multi`
  - `error_km`: MC spread (for intersections) or covariance-derived error (for LSQ when available)
  - LSQ diagnostics: `residual_norm`, `origin_time`, `covariance`

> Note: Monte Carlo azimuth perturbation is a quick and effective way to quantify localization robustness; increasing `--mc-az-sigma` (e.g., 3° → 5° → 10°) will show how stable intersections are to bearing uncertainty.

### Recommended experiments & best practices 🧪

- If you see ring-like clusters near arrays, try these steps:
  1. Increase LSQ attempts (use `--lsq-force-vel` or relax gating: lower `--min-snr`, increase `--lsq-vel-tol`) to encourage LSQ to resolve ambiguous intersections.
  2. Use `--mc-az-sigma 5.0` (or larger) to check whether intersections are robust — large `std_km` indicates poor geometric constraints.
  3. Use `scripts/post_process_locations.py --filter-intersection-km <km>` to drop `intersection_mc` points near centers if you prefer to avoid ring artifacts.

- Suggested sensitivity tests:
  - **Velocity grid sweep:** run `--lsq-force-vel` across values (2.5, 3.0, 3.5, 4.0) and compare method fractions, nearest-center histograms, and per-event moves (use `scripts/vel_grid_sample_diff.py`).
  - **SNR/vel gating sweep:** try `--min-snr` at 5/6/8 and `--lsq-vel-tol` at 0.5/1.0/2.0 to find a robust balance of LSQ acceptance vs spurious LSQ locations.

### Future / optional improvements (ideas)

- Per-detection azimuth uncertainty: if individual detections include a `backazimuth_std` (or `baz_std`) field we can use those per-measurement uncertainties in the MC sampling instead of a single global `--mc-az-sigma`.
- SNR-based MC scaling: scale MC azimuth sigma inversely with SNR (lower SNR → larger sigma) to reflect measurement reliability. This is a proposed enhancement and not enabled by default yet.

These additions make it easy to explore uncertainty and choose parameters that minimize ring-like artifacts while retaining real events.

---

## Installation

### Prerequisites
- Python 3.9 or later
- CUDA toolkit (for GPU acceleration; optional but recommended for large-scale processing)
- Git

### Clone the repository
```bash
git clone https://github.com/<username>/beam.git
cd beam
```

### Option 1: Install with Conda (recommended)
```bash
# Create and activate the conda environment
conda env create -f environment_gpu.yml
conda activate beam-gpu

# Note: The environment_gpu.yml includes cupy-cuda13x by default.
# If you have a different CUDA version, edit environment_gpu.yml
# and replace cupy-cuda13x with the appropriate version:
#   - cupy-cuda11x for CUDA 11.x
#   - cupy-cuda12x for CUDA 12.x
```

### Option 2: Install with pip
```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
# Note: Replace cupy-cuda13x with your CUDA version if different
pip install cupy-cuda13x obspy numpy scipy matplotlib pandas pyproj requests protobuf

# Alternatively, install from requirements file
pip install -r requirements_gpu.txt
```

### CPU-only installation
If you don't have a GPU or want to run in CPU-only mode, you can skip CuPy:
```bash
pip install obspy numpy scipy matplotlib pandas pyproj requests protobuf
```

### Verify installation
```bash
# Run tests to verify everything is working
PYTHONPATH=. python -m pytest -q

# Or check if the module can be imported
python -c "import beam; print('BEAM installation successful')"
```

---

## Quick start (new)

If you just cloned the repo and want to run the pipeline end-to-end for a date range defined in `configs/template_config.json` (or another config), try:

```bash
# from repository root
PYTHONPATH=. python -m beam --config configs/template_config.json
```

This is equivalent to invoking `scripts/run_pipeline.py` directly and will run beamforming → triangulation → clustering by default. Use `--skip-beam`, `--skip-triangulate`, or `--skip-cluster` to run only parts of the workflow.

To regenerate the intersection vs LSQ comparison figure (example created earlier) for velocity 3.0:

```bash
PYTHONPATH=. python3 scripts/plot_intersection_vs_lsq.py --base plots/vel_grid_full_relaxed --vel 3.0 --outdir plots/vel_grid_full_relaxed/plots --max-match-km 50 --min-line-offset-km 2
```

To produce all per-velocity map-pairs and thumbnails you can run (example):

```bash
for v in 2.5 3.0 3.5 4.0; do
  PYTHONPATH=. python3 scripts/plot_intersection_vs_lsq.py --base plots/vel_grid_full_relaxed --vel $v --outdir plots/vel_grid_full_relaxed/plots --max-match-km 50 --min-line-offset-km 2
done
# then regenerate the HTML index if desired
python3 scripts/make_intersection_index.py --base plots/vel_grid_full_relaxed/plots
```

Notes:
- `python -m beam` is a convenience wrapper that delegates to `scripts/run_pipeline.py` and accepts the same CLI flags.
- The `Makefile` still provides shorthand for common steps (`make detect`, `make triage`, `make cluster`, `make test`).

## Workflows & Output Files

- Typical pipeline runs per-day (or a date range) and creates:
  - `plot_dir/subarray_{i}/detections_YYYYMMDD.json` (per-subarray per-day)
  - `plot_dir/detections_YYYYMMDD.json` (per-day aggregated JSON)
  - `plot_dir/locations_YYYYMMDD.json` (pairwise triangulation results)
  - `plot_dir/locations_lsq_YYYYMMDD.json` (LSQ multi-array location results, when at least 3 arrays) 
  - `plot_dir/locations_summary_YYYYMMDD.csv` (deduplicated summary for mapping & analysis)
  - `plot_dir/detections.txt` (legacy summary output)

- The pipeline orchestrator `run_pipeline.py` wraps the daily runs and the follow-up triangular & cluster steps. It prefers `detections_YYYYMMDD.json` when present for triangulation.

## Configuration & Command-Line Options

- Configuration via `configs/template_config.json` (or YAML).
- Some important options:
  - `data_dir`: base directory with daily miniseed files.
  - `inventory`: station XML file or folder (use `inventory_pattern` / `inventory_name` / `inventory_tag` to select files).
  - `mode`: `traditional` or `correlation`.
  - Traditional parameters:
    - `vel_min`, `vel_max`, `vel_step` — velocity grid (km/s)
    - `az_step` — azimuth spacing (degrees)
    - `sta_len`, `lta_len` — STA/LTA windows (seconds)
    - `cf_method` — `envelope`, `energy`, or `kurtosis`.
  - FK & postprocess:
    - `fk_max_per_subarray`, `fk_min_snr` — FK refinement controls.
  - GPU options:
    - `gpu` (true/false), and `gpu_safety_factor` (int) — smaller values use more memory (faster but risk OOM).
    - `BEAM_GPU_SAFETY_FACTOR` env var is used; the code also dynamically reads this value during runtime if needed.
  - `time_tol` — seconds tolerance for pairing detections across subarrays when triangulating.
  - `cluster_radius_km`, `cluster_min_members` — clustering/deduping options.

- Typical CLI invocation:
  ```bash
  PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/template_config.json
  ```
- CLI overrides: any config key can be overridden via CLI flags (e.g., `--gpu-safety-factor 2`, `--time-tol 20`).

## Post-processing (triangulation only)
- You can run triangulation only (without re-running detections) by using `--skip-beam` on `scripts/run_pipeline.py` or calling `scripts/triangulate_from_detections_json.py` directly.

Example (triangulate a single day using per-day file and known centers):
```bash
PYTHONPATH=. python3 scripts/triangulate_from_detections_json.py \
  --detections /path/to/plot_dir/detections_YYYYMMDD.json \
  --centers /path/to/plot_dir/centers.json \
  --date YYYYMMDD --time-tol 20.0 --outdir /path/to/plot_dir
```

## Running the pipeline: full run vs step-by-step 🔁

You can either run the full end-to-end pipeline (beamforming → triangulation → clustering) or run each stage independently. The orchestrator `scripts/run_pipeline.py` wraps the three stages and exposes flags to skip or tune each step.

- Full pipeline (recommended for end-to-end processing):

```bash
PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/template_config.json
```

- Run beamforming only (useful to generate or re-run detections):

```bash
# run pipeline but skip triangulation & clustering
PYTHONPATH=. python3 scripts/run_pipeline.py --config configs/template_config.json --skip-triangulate --skip-cluster

# or run the beam driver directly for a single day/range (more control):
PYTHONPATH=. python3 beam_driver.py --mode traditional --data-dir /path/to/data --start 20200601 --end 20200601 --plot --plot-dir /path/to/plot_dir
```

- Run triangulation only (pairwise intersections from `detections_YYYYMMDD.json`):

```bash
PYTHONPATH=. python3 scripts/triangulate_from_detections_json.py \
  --detections /path/to/plot_dir/detections_YYYYMMDD.json \
  --centers /path/to/plot_dir/centers.json \
  --date YYYYMMDD --time-tol 30.0 --outdir /path/to/plot_dir
```

- Run clustering only (create CSV summaries from `locations_YYYYMMDD.json`):

```bash
PYTHONPATH=. python3 scripts/cluster_locations.py \
  --locations /path/to/plot_dir/locations_YYYYMMDD.json \
  --out /path/to/plot_dir/locations_summary_YYYYMMDD.csv \
  --cluster-km 20.0 --min-members 1
```

### Flags & options you should know about
- `--skip-beam`, `--skip-triangulate`, `--skip-cluster` — use these with `run_pipeline.py` to run or skip stages.
- `--time-tol` (triangulation) — time tolerance (seconds) for pairing detections across arrays; increase to 30–120 s for long-period / surface-wave detections.
- Clustering new flags: `--triangulated-min-arrays N` and `--strict-triangulated` (see `scripts/cluster_locations.py`) — control how a cluster is considered "triangulated":
  - default: a cluster is triangulated if any member has detections from ≥N arrays (N=2 by default)
  - `--strict-triangulated`: require the union of arrays across cluster members to be ≥N (stronger requirement)

### Practical tips
- If your plotted map shows rings/azimuth arcs instead of pointlike events, you are probably plotting single-array detections (backazimuth-only rays). Filter to the triangulated-only CSV (`*_triangulated.csv` or use the `triangulated` column) to plot only multi-array-supported events.
- If you have only two arrays (pairwise triangulation only), you will get pairwise intersections; LSQ localization requires ≥3 arrays and is produced by the pipeline when available as `locations_lsq_YYYYMMDD.json`.

If you'd like, I can add a short example `Makefile` or convenience wrapper to run common combinations of these commands (full pipeline, triangulate-only, cluster-only) to make day-to-day processing simpler.

## Troubleshooting & Tips

  1. Confirm per-subarray outputs exist: `plot_dir/subarray_i/detections_YYYYMMDD.json`.
  2. Check `time_tol`: a small default (5 s) may miss pairings. Increase for long-period/surface waves.
  3. Check inventory discovery (pattern / name / tag) — `--inventory-symbol` flags can help.
  4. GPU memory OOMs: reduce `--gpu-safety-factor` or use CPU-only mode by clearing `--gpu`.

  - Use `--verbose` for detailed logs (including beam counts and debug messages). Be aware: per-beam trace-level debug can be very noisy.
  - The GPU module logs safety & batch sizing info at INFO/DEBUG.

  - Set `BEAM_GPU_SAFETY_FACTOR` environment variable or use `--gpu-safety-factor` on the CLI.
  - Lower safety factor increases usable memory and thus batch size; higher factor reduces usable memory and is safer (but maybe more CPU fallback).

## Tests

## License & Acknowledgments

## Contact

## Contributing & GitHub prep

This repository is ready for publishing to GitHub. Below are notes and the files added to help with sharing, CI, and contribution workflows.

- Added: `.gitignore` — removes generated files and local environments from your Git history.
- Added: `LICENSE` — default MIT License; please update the YEAR and AUTHOR in the file before publishing.
- Added: `CONTRIBUTING.md` — contributor quick start and testing guidelines.
- Added: CI workflow (`.github/workflows/python-app.yml`) to run tests via GitHub Actions on pushes/PRs.
- Added: `Makefile` — convenient short-hand for running beamforming/triangulation/clustering for quick tests.

Before publishing to GitHub:
1. Inspect `LICENSE` and replace the placeholder `YEAR` and `YOUR NAME` with your information.
2. If you prefer another license please swap accordingly (e.g., BSD, Apache). Remove if you wish to manage licensing differently.
3. If you don't want a public CI workflow (the added GitHub Actions file), remove or adjust it.

How to push to GitHub quickly:

```bash
# initialize a new repo and push to GitHub (replace <remote> and branch name as desired)
git init
git add .
git commit -m "Initial import: BEAM project"
git remote add origin git@github.com:<username>/<repo>.git
git push -u origin main
```

Once pushed, enable repository settings (branches, required checks, etc.) and consider setting up repository secrets for any CI needs (e.g., private data or keys).
