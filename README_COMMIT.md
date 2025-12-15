Initial plotting tools for forced-LSQ velocity grid

This repo snapshot includes plotting utilities added during exploratory analysis of the velocity grid run. The aim is to make a small, focused public commit containing the plotting tools and small helper scripts — not the generated plots or intermediate outputs.

Files included in this commit (examples):
- `scripts/plot_vel_grid_zoomed.py` — zoomed per-velocity plotting with color-by / size-by options, Cartopy basemap fallback, and LSQ-only filtering.
- `scripts/plot_lsq_comparison_panel.py` — creates a 2×2 comparison PDF of LSQ-only solutions across velocities.
- `scripts/run_lsq_only_suite.sh`, `scripts/run_lsq_only_combined.sh` — convenience runners to regenerate plots.
- `scripts/README.md` — short usage notes.
- `.gitignore` — excludes `plots/` and generated files.

Notes:
- Generated plots live under `plots/` and are intentionally excluded (see `.gitignore`).
- Tests in this environment may include GPU-dependent packages; CI should run a subset suitable for public builds.

Next steps suggested:
- Tidy a few small functions and add unit tests for `compute_field` and `_compute_sizes`.
- Add a lightweight example dataset or a small sample top-level `plots/` (if desired) that is safe to publish.
