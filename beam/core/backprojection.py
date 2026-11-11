"""
Back-projection (source-scanning) locator.

Rather than locating from per-array picks of time, backazimuth and slowness (as
``triangulation`` and ``locator`` do), back projection stacks waveform energy
directly. For every trial source node ``j`` and origin time ``t0`` it sums each
station's characteristic function (CF) at the predicted arrival time:

    B(j, t0) = sum_p sum_i w_i * CF_i^p( t0 + T_p(j, i) )

where ``T_p(j, i)`` is the travel time of phase ``p`` from node ``j`` to station
``i``. The node and origin time of the maximum is the location. Nothing is
picked first, so arrays and lone stations contribute on equal terms, a
detection that one sub-array missed still adds its energy, and the shape of the
brightness map shows how well the location is constrained.

Pipeline
--------
1. :func:`make_source_grid` / :func:`grid_around_stations` -- trial nodes
   (lat, lon, depth).
2. :func:`travel_time_table` -- ``(n_nodes, n_stations)`` times per phase, from
   a constant velocity (straight rays, any phase including surface waves) or a
   1-D ObsPy TauP model (body waves). Build once and reuse for every event.
3. :func:`characteristic_functions` -- per-station CFs, normalised so that
   noise contributes about the same at every station regardless of gain.
4. :func:`backproject` -- the stack, its maximum, and an uncertainty estimate.

Weights
-------
A dense sub-array of ten stations would otherwise outvote a lone station ten to
one and pull the location toward itself. :func:`subarray_weights` gives each
sub-array a total weight of 1, shared among its stations.

Limits worth knowing
--------------------
**Stack every phase that is in the data.** A P-only stack on records that also
carry S can align P at one sub-array with S at another and produce a location
that is wrong by tens of km yet coherent and sharply peaked -- nothing in the
result flags it. In synthetic tests (three 5-station sub-arrays, S twice P)
P-only missed by 15 km at coherence 0.89; P+S located within 1 km.

**The grid must enclose the sources.** A source outside the grid is drawn to
an interior node, often with a tight error estimate; ``on_edge`` catches only
some of these. Size ``margin_km`` generously and treat a drop in coherence as a
warning.

Range resolution needs aperture: a single compact array constrains direction
well and distance poorly, so expect an elongated brightness ridge, not a point,
when only one array sees the event. Depth trades off against origin time
unless stations at different distances (or both P and S) constrain it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

KM_PER_DEG = 111.19


# --------------------------------------------------------------------- grid ---
@dataclass
class SourceGrid:
    """Trial source nodes, flattened in (depth, y, x) order.

    ``lat``, ``lon``, ``depth_km``, ``x_km`` and ``y_km`` each have length
    ``n_nodes``; ``shape`` is ``(n_depth, n_y, n_x)`` for reshaping a per-node
    quantity into a volume. ``x_km``/``y_km`` are east/north of ``origin``.
    """
    lat: np.ndarray
    lon: np.ndarray
    depth_km: np.ndarray
    x_km: np.ndarray
    y_km: np.ndarray
    shape: Tuple[int, int, int]
    origin: Tuple[float, float]
    step_km: float
    depths: np.ndarray = field(default_factory=lambda: np.zeros(1))

    @property
    def n_nodes(self) -> int:
        return int(self.lat.size)


def make_source_grid(center_lat: float, center_lon: float, half_width_km: float,
                     step_km: float, depths_km: Sequence[float] = (0.0,),
                     half_height_km: Optional[float] = None) -> SourceGrid:
    """Regular grid of trial sources centred on (``center_lat``, ``center_lon``).

    Parameters
    ----------
    half_width_km, half_height_km : float
        Half-extent east-west and north-south (``half_height_km`` defaults to
        ``half_width_km``).
    step_km : float
        Horizontal node spacing.
    depths_km : sequence of float
        Trial depths (km, positive down, relative to sea level).
    """
    if step_km <= 0 or half_width_km <= 0:
        raise ValueError("step_km and half_width_km must be positive")
    hh = half_width_km if half_height_km is None else half_height_km
    nx = int(np.ceil(half_width_km / step_km))
    ny = int(np.ceil(hh / step_km))
    xs = np.arange(-nx, nx + 1) * step_km
    ys = np.arange(-ny, ny + 1) * step_km
    depths = np.atleast_1d(np.asarray(depths_km, dtype=np.float64))
    D, Y, X = np.meshgrid(depths, ys, xs, indexing="ij")
    coslat = np.cos(np.radians(center_lat))
    lat = center_lat + Y / KM_PER_DEG
    lon = center_lon + X / (KM_PER_DEG * coslat)
    return SourceGrid(lat=lat.ravel(), lon=lon.ravel(), depth_km=D.ravel(),
                      x_km=X.ravel(), y_km=Y.ravel(), shape=D.shape,
                      origin=(float(center_lat), float(center_lon)),
                      step_km=float(step_km), depths=depths)


def grid_around_stations(station_coords: Mapping[str, Sequence[float]],
                         margin_km: float, step_km: float,
                         depths_km: Sequence[float] = (0.0,)) -> SourceGrid:
    """Grid spanning the stations' bounding box plus ``margin_km`` on every side."""
    lats = np.array([v[0] for v in station_coords.values()], dtype=np.float64)
    lons = np.array([v[1] for v in station_coords.values()], dtype=np.float64)
    clat, clon = 0.5 * (lats.min() + lats.max()), 0.5 * (lons.min() + lons.max())
    half_w = 0.5 * (lons.max() - lons.min()) * KM_PER_DEG * np.cos(np.radians(clat)) + margin_km
    half_h = 0.5 * (lats.max() - lats.min()) * KM_PER_DEG + margin_km
    return make_source_grid(clat, clon, half_w, step_km, depths_km, half_height_km=half_h)


# ------------------------------------------------------------- travel times ---
def _epicentral_km(grid: SourceGrid, lat: float, lon: float) -> np.ndarray:
    from obspy.geodetics import locations2degrees
    return np.asarray(locations2degrees(grid.lat, grid.lon, lat, lon)) * KM_PER_DEG


def travel_time_table(grid: SourceGrid, station_coords: Mapping[str, Sequence[float]],
                      stations: Sequence[str], *, velocity: Optional[float] = None,
                      model: Optional[str] = None, phase: str = "P",
                      n_distances: int = 200) -> np.ndarray:
    """Travel times (s) from every node to every station, shape (n_nodes, n_sta).

    Give exactly one of:

    ``velocity`` (km/s)
        Straight rays at constant speed through the 3-D node-station distance
        (station elevation in metres, as ObsPy stores it, is included). Works
        for any phase -- including surface waves, where it is the group
        velocity and depth is meaningless (use a single depth of 0).
    ``model``
        A 1-D ObsPy TauP model name (``'iasp91'``, ``'ak135'``, ...). The first
        arrival of ``phase`` (``'P'`` tries p/P/Pn/Pg, ``'S'`` likewise) is
        tabulated against distance for each grid depth and interpolated.
        Station elevation is ignored.
    """
    if (velocity is None) == (model is None):
        raise ValueError("give exactly one of velocity= or model=")
    missing = [s for s in stations if s not in station_coords]
    if missing:
        raise KeyError(f"no coordinates for stations {missing}")
    tt = np.empty((grid.n_nodes, len(stations)), dtype=np.float64)

    if velocity is not None:
        if velocity <= 0:
            raise ValueError("velocity must be positive")
        for k, sta in enumerate(stations):
            lat, lon = station_coords[sta][0], station_coords[sta][1]
            elev_km = (station_coords[sta][2] if len(station_coords[sta]) > 2 else 0.0) / 1000.0
            h = _epicentral_km(grid, lat, lon)
            dz = grid.depth_km + elev_km
            tt[:, k] = np.hypot(h, dz) / velocity
        return tt

    from obspy.taup import TauPyModel
    taup = TauPyModel(model=model)
    base = phase.upper()
    phase_list = [base.lower(), base, base + "n", base + "g"]
    epi = np.column_stack([_epicentral_km(grid, station_coords[s][0], station_coords[s][1])
                           for s in stations])
    dmax = float(epi.max()) * 1.01 + 1e-3
    dist_km = np.linspace(0.0, dmax, n_distances)
    for depth in np.unique(grid.depth_km):
        times = np.full(dist_km.size, np.nan)
        for m, d in enumerate(dist_km):
            arr = taup.get_travel_times(source_depth_in_km=float(max(depth, 0.0)),
                                        distance_in_degree=float(d / KM_PER_DEG),
                                        phase_list=phase_list)
            if arr:
                times[m] = min(a.time for a in arr)
        ok = np.isfinite(times)
        if ok.sum() < 2:
            raise ValueError(f"TauP returned no {phase} arrivals at depth {depth} km")
        sel = grid.depth_km == depth
        tt[sel] = np.interp(epi[sel], dist_km[ok], times[ok])
    return tt


def subarray_weights(stations: Sequence[str],
                     subarrays: Optional[Sequence[Sequence[str]]]) -> np.ndarray:
    """Per-station weights giving each sub-array (or lone station) a total of 1."""
    w = np.ones(len(stations), dtype=np.float64)
    if not subarrays:
        return w
    for group in subarrays:
        members = [k for k, s in enumerate(stations) if s in set(group)]
        for k in members:
            w[k] = 1.0 / len(members)
    return w


# -------------------------------------------------- characteristic functions ---
def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    kernel = np.ones(width) / width
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), -1, x)


def characteristic_functions(traces: np.ndarray, sampling_rate: float, *,
                             method: str = "sta_lta", sta_s: float = 0.5,
                             lta_s: float = 10.0, smooth_s: float = 0.5,
                             target_rate: Optional[float] = None
                             ) -> Tuple[np.ndarray, float]:
    """Normalised, smoothed characteristic functions for back projection.

    Parameters
    ----------
    traces : ndarray, shape (n_sta, n_samples)
        Filtered waveforms on a common time base.
    method : {'sta_lta', 'envelope'}
        ``'sta_lta'``: classic STA/LTA minus 1, clipped at 0 -- gain-free, ~0
        on noise and peaked at onsets; the default. ``'envelope'``: Hilbert
        envelope divided by each station's maximum -- keeps the whole wave
        train, suits emergent or surface-wave arrivals.
    smooth_s : float
        Moving-average width. It should be comparable to the travel-time error
        you expect from the velocity model, or a correct node will fail to
        stack.
    target_rate : float, optional
        Resample the (smooth) CFs to about this rate by block averaging; the
        stack cost scales with it. ``None`` keeps ``sampling_rate``.

    Returns
    -------
    cf : ndarray, shape (n_sta, n_out)
    rate : float
        Sampling rate of ``cf``.
    """
    x = np.asarray(traces, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("traces must be (n_stations, n_samples)")
    x = x - x.mean(axis=1, keepdims=True)
    fs = float(sampling_rate)
    if method == "sta_lta":
        from beam.core.gpu_beam import classic_sta_lta
        nsta = max(1, int(round(sta_s * fs)))
        nlta = max(nsta + 1, int(round(lta_s * fs)))
        cf = np.asarray(classic_sta_lta(x, nsta, nlta, use_gpu=False))
        cf = np.clip(np.nan_to_num(cf) - 1.0, 0.0, None)
    elif method == "envelope":
        from scipy.signal import hilbert
        cf = np.abs(hilbert(x, axis=1))
    else:
        raise ValueError(f"unknown CF method {method!r}")
    cf = _moving_average(cf, int(round(smooth_s * fs)))
    if method == "envelope":
        peak = cf.max(axis=1, keepdims=True)
        cf = cf / np.where(peak > 0, peak, 1.0)

    if target_rate and target_rate < fs:
        q = int(fs // target_rate)
        if q > 1:
            n = (cf.shape[1] // q) * q
            cf = cf[:, :n].reshape(cf.shape[0], -1, q).mean(axis=2)
            fs = fs / q
    return cf, fs


# ------------------------------------------------------------------- stack ---
@dataclass
class BackprojectionResult:
    lat: float
    lon: float
    depth_km: float
    origin_time: float          # same time base as ``start_time`` (e.g. epoch s)
    peak: float                 # stack value at the maximum
    coherence: float            # peak / attainable maximum (every CF at its own max), in [0, 1]
    contrast: float             # peak / median node brightness
    node: int
    on_edge: bool               # maximum on the grid boundary: likely outside it
                                # (the converse does not hold -- see module notes)
    err_x_km: float             # spread of nodes above ``err_level``, with the grid's
                                # own quantisation error (step/sqrt(12)) as a floor
    err_y_km: float
    err_z_km: float
    n_stations: int
    brightness: np.ndarray      # (n_nodes,) max over origin time
    best_t0: np.ndarray         # (n_nodes,) origin time of that max

    def to_dict(self, include_maps: bool = False) -> Dict:
        out = {k: v for k, v in self.__dict__.items()
               if k not in ("brightness", "best_t0")}
        out = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                   int(v) if isinstance(v, (int, np.integer)) else float(v))
               for k, v in out.items()}
        if include_maps:
            out["brightness"] = self.brightness.tolist()
        return out


def backproject(cfs: Union[np.ndarray, Sequence[np.ndarray]],
                tts: Union[np.ndarray, Sequence[np.ndarray]],
                rate: float, start_time: float, grid: SourceGrid, *,
                weights: Optional[np.ndarray] = None,
                t0_range: Optional[Tuple[float, float]] = None,
                err_level: float = 0.9,
                chunk_nodes: int = 4096) -> BackprojectionResult:
    """Stack characteristic functions along predicted travel times.

    Parameters
    ----------
    cfs, tts : array or list of arrays
        One CF matrix ``(n_sta, n_t)`` and one travel-time table
        ``(n_nodes, n_sta)`` per phase, in matching station order. Different
        phases may share a CF.
    rate : float
        CF sampling rate (Hz).
    start_time : float
        Time of CF sample 0 (epoch seconds, or any consistent base).
    weights : ndarray (n_sta,), optional
        Station weights, e.g. from :func:`subarray_weights`.
    t0_range : (float, float), optional
        Origin times to scan, in the ``start_time`` base. Defaults to every
        origin time whose earliest predicted arrival lies inside the CF.
    err_level : float
        Nodes whose background-removed brightness is at least this fraction of
        the peak's define the reported spread (``err_*_km``).
    """
    if isinstance(cfs, np.ndarray) and cfs.ndim == 2:
        cfs = [cfs]
    if isinstance(tts, np.ndarray) and tts.ndim == 2:
        tts = [tts]
    if len(cfs) != len(tts):
        raise ValueError("need one travel-time table per CF")
    n_sta, n_t = cfs[0].shape
    for cf, tt in zip(cfs, tts):
        if cf.shape != (n_sta, n_t) or tt.shape != (grid.n_nodes, n_sta):
            raise ValueError("CF/travel-time shapes disagree with each other or the grid")
    w = np.ones(n_sta) if weights is None else np.asarray(weights, dtype=np.float64)

    shifts = [np.rint(tt * rate).astype(np.int64) for tt in tts]
    if t0_range is None:
        # every origin time for which at least one predicted arrival can fall
        # inside the CF; arrivals outside it contribute zero, so edge origin
        # times are biased low, never high
        k_lo = -max(int(s.max()) for s in shifts)
        k_hi = n_t - 1 - min(int(s.min()) for s in shifts)
    else:
        k_lo = int(np.floor((t0_range[0] - start_time) * rate))
        k_hi = int(np.ceil((t0_range[1] - start_time) * rate))
    if k_hi < k_lo:
        raise ValueError("empty origin-time range")
    ks = np.arange(k_lo, k_hi + 1)

    brightness = np.empty(grid.n_nodes)
    best_k = np.empty(grid.n_nodes, dtype=np.int64)
    padded = [np.concatenate([np.zeros((n_sta, 1)), cf], axis=1) for cf in cfs]  # col 0 = outside
    for a in range(0, grid.n_nodes, chunk_nodes):
        b = min(a + chunk_nodes, grid.n_nodes)
        stack = np.zeros((b - a, ks.size))
        for pad, sh in zip(padded, shifts):
            for i in range(n_sta):
                if w[i] == 0:
                    continue
                idx = sh[a:b, i][:, None] + ks[None, :]
                valid = (idx >= 0) & (idx < n_t)
                stack += w[i] * pad[i][np.where(valid, idx + 1, 0)]
        j = np.argmax(stack, axis=1)
        brightness[a:b] = stack[np.arange(b - a), j]
        best_k[a:b] = ks[j]

    node = int(np.argmax(brightness))
    peak = float(brightness[node])
    med = float(np.median(brightness))
    attainable = float(sum(np.sum(w * cf.max(axis=1)) for cf in cfs))

    rel = (brightness - med) / (peak - med) if peak > med else np.zeros_like(brightness)
    sel = rel >= err_level
    wt = np.clip(rel[sel], 0, None)
    def spread(v, step):
        var = float(np.sum(wt * (v[sel] - v[node]) ** 2) / np.sum(wt)) if wt.sum() > 0 else 0.0
        return float(np.sqrt(var + step ** 2 / 12.0))
    dz = float(np.min(np.diff(grid.depths))) if grid.depths.size > 1 else 0.0

    nd, ny, nx = grid.shape
    d_i, y_i, x_i = np.unravel_index(node, grid.shape)
    on_edge = (y_i in (0, ny - 1)) or (x_i in (0, nx - 1)) or (nd > 1 and d_i in (0, nd - 1))

    return BackprojectionResult(
        lat=float(grid.lat[node]), lon=float(grid.lon[node]),
        depth_km=float(grid.depth_km[node]),
        origin_time=float(start_time + best_k[node] / rate),
        peak=peak, coherence=peak / attainable if attainable > 0 else float("nan"),
        contrast=peak / med if med > 0 else float("inf"),
        node=node, on_edge=bool(on_edge),
        err_x_km=spread(grid.x_km, grid.step_km), err_y_km=spread(grid.y_km, grid.step_km),
        err_z_km=spread(grid.depth_km, dz),
        n_stations=int(np.count_nonzero(w)),
        brightness=brightness, best_t0=start_time + best_k / rate)


# --------------------------------------------------------------- figure ---
def plot_brightness(result: BackprojectionResult, grid: SourceGrid,
                    station_coords: Mapping[str, Sequence[float]], outfile: str,
                    title: str = "") -> str:
    """Map of brightness (max over depth) with stations and the located source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = result.brightness.reshape(grid.shape).max(axis=0)
    lats = grid.lat.reshape(grid.shape)[0]
    lons = grid.lon.reshape(grid.shape)[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    pc = ax.pcolormesh(lons, lats, vol, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax, label="stack (max over origin time and depth)")
    ax.plot([v[1] for v in station_coords.values()], [v[0] for v in station_coords.values()],
            "^", color="white", mec="black", ms=7, label="stations")
    ax.plot(result.lon, result.lat, "*", color="red", mec="black", ms=16, label="back projection")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title or f"depth {result.depth_km:.1f} km, coherence {result.coherence:.2f}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    return outfile
