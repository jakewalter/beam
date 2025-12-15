#!/usr/bin/env bash
# Run combined LSQ-only plotting: color=SNR, size=azvar (or vice versa if desired)
set -euo pipefail
BASE=plots/vel_grid_full_relaxed
VELS="2.5 3.0 3.5 4.0"
ZOOM=100
COLOR_BYS=azimuth_variance
SIZE_BY=mean_snr
SIZE_RANGE=6,40
for v in $VELS; do
  outdir="$BASE/suite/lsq_only_vel_${v}_combined"
  mkdir -p "$outdir"
  echo "Running combined lsq-only plot for vel=$v -> $outdir"
  python3 scripts/plot_vel_grid_zoomed.py --base "$BASE" --outdir "$outdir" --vels "$v" --zooms "$ZOOM" --color-bys "$COLOR_BYS" --size-by "$SIZE_BY" --size-range "$SIZE_RANGE" --only-method lsq
  echo "Finished $outdir"
done
