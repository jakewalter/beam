# Plotting scripts

This folder contains plotting utilities used for velocity-grid analysis.

Key scripts:
- `plot_vel_grid_zoomed.py` — generate zoomed maps and color-by / size-by diagnostics for per-velocity post-processed location JSONs.
- `plot_lsq_comparison_panel.py` — creates a 2×2 PDF comparing LSQ-only solutions across velocities (color + size mapping supported).

Convenience runners:
- `run_lsq_only_suite.sh` — run per-velocity LSQ-only baseline maps.
- `run_lsq_only_combined.sh` — run combined size/color LSQ-only maps.

Notes:
- Generated plot files and other outputs live under `plots/` and are intentionally ignored by `.gitignore`.
- Use `--only-method lsq` to make LSQ-only plots, and `--size-by`/`--color-bys` to control mapping.
