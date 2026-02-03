"""
Visualization utilities for BEAM detections.

Provides functions to create daily detection plots including:
- station map with detection markers
- beam time-series around detection (radial/beamformed trace)
- radial beam/polar plot: beam power as a function of backazimuth

These helpers are designed to work with TraditionalBeamformer output.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from obspy import UTCDateTime


def plot_daily_detections(date_str, stream, detections, beamformer, outdir='.', window=120.0):
    """
    Save a figure summarizing detections for a single date.

    Parameters
    ----------
    date_str : str
        YYYYMMDD
    stream : obspy.Stream
        Preprocessed stream used for beamforming
    detections : list
        List of detection dicts (as returned by process_single_day)
    beamformer : TraditionalBeamformer
        Instance used to create beams (used to compute beam traces)
    outdir : str
        Output directory for saved figures
    window : float
        Seconds either side of detection center to plot
    """
    # If there are no detections, still produce a diagnostic figure
    # so users can inspect array geometry and a representative beam
    # (useful when tuning parameters). We'll pick a representative
    # window from the middle of the stream.

    os.makedirs(outdir, exist_ok=True)

    # If there are detections, create one subplot per detection (max 6).
    # Otherwise create a single diagnostic panel for the day.
    if detections:
        n = len(detections)
        nmax = min(n, 6)
        multi_mode = True
    else:
        n = 1
        nmax = 1
        multi_mode = False

    fig = plt.figure(figsize=(14, 4 * nmax))

    # When there are no detections we create a single diagnostic 'fake' det
    dets_to_plot = detections[:nmax] if multi_mode else [None]

    # Pre-compute day-level azimuth stats for the center polar plot
    # Build arrays of backazimuth and snr for all detections (if present)
    if detections:
        az_list = np.array([d.get('backazimuth', np.nan) for d in detections if d is not None and 'backazimuth' in d])
        snr_list = np.array([d.get('snr', np.nan) for d in detections if d is not None and 'snr' in d])
    else:
        az_list = np.array([])
        snr_list = np.array([])

    # Setup histogram bins (narrow bins gives finer angular resolution)
    n_bins = 36
    if az_list.size > 0:
        theta_all = np.radians(az_list % 360.0)
        counts_all, edges_all = np.histogram(theta_all, bins=n_bins, range=(0, 2 * np.pi))
        snr_sums_all, _ = np.histogram(theta_all, bins=edges_all, weights=snr_list)
        mean_snr_all = snr_sums_all / np.where(counts_all == 0, 1.0, counts_all)
    else:
        counts_all = np.zeros(n_bins, dtype=int)
        edges_all = np.linspace(0, 2 * np.pi, n_bins + 1)
        mean_snr_all = np.zeros_like(counts_all, dtype=float)

    for i, det in enumerate(dets_to_plot):
        # Determine central time / parameters for this plot
        if det is not None:
            t_center = det['time']
            vel = det.get('velocity', None)
            slowness = det.get('slowness', None)
            cf_method = det.get('cf_method', 'envelope')
            az = det.get('backazimuth', 0.0)
        else:
            try:
                start = stream[0].stats.starttime
                end = stream[0].stats.endtime
                for tr in stream:
                    if tr.stats.starttime < start:
                        start = tr.stats.starttime
                    if tr.stats.endtime > end:
                        end = tr.stats.endtime
                t_center = (start + (end - start) / 2)
            except Exception:
                t_center = UTCDateTime()
            vel = 3.0
            slowness = 1.0 / vel
            cf_method = 'envelope'
            az = 180.0

        if vel is None and slowness is not None:
            vel = 1.0 / slowness if slowness > 0 else None
        if slowness is None and vel is not None:
            slowness = 1.0 / vel

        # Trim and prepare stream for the beam window
        t0 = UTCDateTime(t_center) - window / 2
        t1 = UTCDateTime(t_center) + window / 2

        st = stream.copy()
        try:
            st.trim(t0, t1, pad=True, fill_value=0.0)
        except Exception:
            try:
                st.trim(t0, t1)
            except Exception:
                st = stream.copy()

        # compute primary beam (for plotting) if possible
        beam = None
        if slowness is not None and az is not None:
            beam = beamformer.beamform(st, slowness, az, normalize=True, use_envelope=True, cf_method=cf_method)

        # Compute beam power across azimuth grid
        azs = np.arange(0, 360, 5)
        powers = []
        for a in azs:
            b = beamformer.beamform(st, slowness, a, normalize=True, use_envelope=True, cf_method=cf_method)
            if b is None:
                powers.append(0.0)
            else:
                data = np.abs(hilbert(b.data))
                n = len(data)
                center = n // 2
                halfw = max(1, int(0.05 * n))
                window_slice = data[max(0, center-halfw):min(n, center+halfw)]
                if window_slice.size == 0:
                    powers.append(np.max(data))
                else:
                    powers.append(np.mean(window_slice))

        powers = np.array(powers)
        if powers.size == 0:
            powers = np.zeros_like(azs, dtype=float)

        # normalize and smooth powers for plotting
        if powers.max() > 0:
            powers = powers / powers.max()
        kernel = np.array([0.25, 0.5, 0.25])
        powers = np.convolve(powers, kernel, mode='same')

        # Plotting
        ax1 = fig.add_subplot(nmax, 3, 3*i+1)
        # station map (x,y) from geometry
        if beamformer.array_geometry:
            xs = [v[0] for v in beamformer.array_geometry.values()]
            ys = [v[1] for v in beamformer.array_geometry.values()]
            ax1.scatter(xs, ys, c='k', s=30)
            for j,sta in enumerate(beamformer.array_geometry.keys()):
                ax1.text(xs[j], ys[j], sta)
            ax1.set_title(f"Stations (date {date_str})")
            ax1.set_xlabel('x (km)')
            ax1.set_ylabel('y (km)')
        else:
            ax1.text(0.5,0.5,'No station geometry available',ha='center')

        ax2 = fig.add_subplot(nmax, 3, 3*i+2)
        if beam is not None:
            times = np.linspace(0, len(beam.data)/beam.stats.sampling_rate, len(beam.data))
            ax2.plot(times, beam.data, color='k', label='beam')
            ax2.set_title(f"Beam time series: vel={vel:.2f} km/s, az={az:.1f}°")
            ax2.axvline((window/2), color='r', linestyle='--')
            ax2.set_xlabel('seconds (relative window start)')
            ax2.grid(True)
        else:
            ax2.text(0.5,0.5,'Beam could not be computed',ha='center')

        ax3 = fig.add_subplot(nmax, 3, 3*i+3, projection='polar')
        # Configure polar axes: 0 at North and clockwise direction
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)

        theta = np.radians(azs)

        # Draw day-level azimuthal histogram bars (counts colored by mean-SNR)
        width = (2 * np.pi) / n_bins
        bin_centers = edges_all[:-1] + width / 2.0
        radii = counts_all.astype(float)

        cmap = plt.get_cmap('viridis')
        colors = mean_snr_all.copy()
        # mark empty bins as NaN so they can be rendered differently
        colors[counts_all == 0] = np.nan
        if np.isfinite(colors).any():
            cmin = np.nanmin(colors)
            cmax = np.nanmax(colors)
        else:
            cmin, cmax = 0.0, 1.0
        norm = plt.Normalize(vmin=cmin, vmax=cmax)
        mapped = cmap(norm(colors))

        # faintly show empty bins in a light grey
        empty_color = (0.9, 0.9, 0.9, 0.6)
        for i, r in enumerate(radii):
            col = mapped[i] if counts_all[i] > 0 else empty_color
            ax3.bar(bin_centers[i], r, width=width * 0.9, bottom=0.0, color=col, edgecolor='k', alpha=0.9)

        # Overlay per-detection beam power curve (scaled to counts for readability)
        overlay_scale = max(1.0, radii.max())
        scaled_powers = powers * overlay_scale * 0.9
        ax3.plot(theta, scaled_powers, '-o', linewidth=1.5, color='C3', markersize=4)
        # mark maximum power azimuth for clarity (on scaled overlay)
        peak_idx = int(np.nanargmax(powers)) if powers.size > 0 else 0
        ax3.plot([theta[peak_idx]], [scaled_powers[peak_idx]], marker='D', color='red')
        ax3.annotate(f"{azs[peak_idx]:.0f}°", xy=(theta[peak_idx], scaled_powers[peak_idx]), xytext=(5, 5), textcoords='offset points')

        # scatter overlay for individual detections (small radius, color by SNR)
        if az_list.size > 0:
            theta_det = np.radians(az_list % 360.0)
            scatter_r = np.ones_like(theta_det) * (0.05 * max(radii.max(), 1.0))
            snr_norm = (snr_list - np.nanmin(snr_list)) / max(1e-6, np.nanmax(snr_list) - np.nanmin(snr_list))
            ax3.scatter(theta_det, scatter_r, c=snr_list, cmap='inferno', s=30 + 50 * snr_norm, alpha=0.9, edgecolor='k')

        # add colorbar for mean-SNR coloring
        try:
            mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
            mappable.set_array(mean_snr_all)
            cbar = fig.colorbar(mappable, ax=ax3, pad=0.1, orientation='vertical')
            cbar.set_label('Mean SNR (per azimuth bin)')
        except Exception:
            # colorbar may fail in some backends / tests; ignore
            pass
        ax3.set_title('Normalized radial beam power')

    fig.tight_layout()
    outfile = os.path.join(outdir, f'detections_{date_str}.png')
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    return outfile


def plot_benchmark_comparison(results, outdir='.', prefix='benchmark'):
    """Create comparison figures for CPU vs GPU benchmarking results.

    Parameters
    ----------
    results : list of dict
        Each dict should contain at least: 'label', 'wall_time', 'unique_detections'
        Optionally: 'per_subarray' (dict mapping id -> count).
    outdir : str
        Directory where figures are written.
    prefix : str
        Filename prefix.

    Returns
    -------
    dict of generated filenames
    """
    os.makedirs(outdir, exist_ok=True)

    labels = [r.get('label', f'r{i}') for i, r in enumerate(results)]
    times = [r.get('wall_time', 0.0) for r in results]
    counts = [r.get('unique_detections', 0) for r in results]

    out_files = {}

    # bar chart: runtime and detections side-by-side
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = range(len(results))
    bar1 = ax1.bar([i - 0.15 for i in x], times, width=0.3, color='C0', label='wall_time (s)')
    ax1.set_ylabel('Wall time (s)')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    bar2 = ax2.bar([i + 0.15 for i in x], counts, width=0.3, color='C1', label='unique detections')
    ax2.set_ylabel('Unique detections')

    # add legend
    ax1.legend(handles=[bar1, bar2], loc='upper left')
    fig.tight_layout()
    f1 = os.path.join(outdir, f"{prefix}_summary.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    out_files['summary'] = f1

    # scatter plot: detections vs runtime
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(times, counts, c='C2')
    for i, lab in enumerate(labels):
        ax.annotate(lab, (times[i], counts[i]))
    ax.set_xlabel('Wall time (s)')
    ax.set_ylabel('Unique detections')
    ax.grid(True)
    f2 = os.path.join(outdir, f"{prefix}_scatter.png")
    fig2.tight_layout()
    fig2.savefig(f2, dpi=150)
    plt.close(fig2)
    out_files['scatter'] = f2

    # If per-subarray counts exist for all results, create grouped bar plot
    all_per = [r.get('per_subarray', None) for r in results]
    if all(v is not None for v in all_per):
        # build list of subarray ids (union)
        all_ids = sorted({k for d in all_per for k in (d.keys())})
        n = len(all_ids)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        width = 0.8 / max(1, len(results))
        x = range(n)
        for i, d in enumerate(all_per):
            vals = [d.get(k, 0) for k in all_ids]
            ax3.bar([xi + (i - len(results)/2.0) * width for xi in x], vals, width=width, label=results[i].get('label'))
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(k) for k in all_ids])
        ax3.set_xlabel('Subarray ID')
        ax3.set_ylabel('Detections')
        ax3.legend()
        fig3.tight_layout()
        f3 = os.path.join(outdir, f"{prefix}_per_subarray.png")
        fig3.savefig(f3, dpi=150)
        plt.close(fig3)
        out_files['per_subarray'] = f3

    return out_files
