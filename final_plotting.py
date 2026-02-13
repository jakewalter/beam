import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse
import warnings
import csv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, message="The handle 'handle' has a label of '_nolegend_' which cannot be automatically added to the legend.")

def get_day_folders(project_folder):
    """Returns a sorted list of valid day folder paths."""
    day_folders = []
    for item in os.listdir(project_folder):
        path = os.path.join(project_folder, item)
        if os.path.isdir(path) and item.isdigit() and len(item) == 8:
            day_folders.append(path)
    print(f"[DEBUG] Found {len(day_folders)} day folders: {day_folders}")
    return sorted(day_folders)

def read_station_locations(day_folders, seismicflag):
    """Reads unique station locations from all StationLocs files."""
    stations = set()
    for day_folder in day_folders:
        search_path = os.path.join(day_folder, f"StationLocs_*_flag{seismicflag}.txt")
        station_files = glob.glob(search_path)
        print(f"[DEBUG] Searching for station files in {day_folder}: {station_files}")
        for station_file in station_files:
            try:
                # StationLocs file format has varied across drivers and versions.
                # Possible layouts:
                #   1) StationName Lat Lon  (string, float, float)
                #   2) EpochID Lat Lon      (int, float, float)
                #   3) Lon Lat              (float, float)
                # We'll parse line-by-line and be permissive: detect token types and
                # extract lat/lon correctly, then add them to a unique set.
                with open(station_file, 'r') as sf:
                    lines = [l.strip() for l in sf.readlines() if l.strip()]
                if len(lines) <= 1:
                    print(f"[DEBUG] No station rows found in {station_file}")
                    continue
                # parse each non-header row
                count = 0
                for line in lines[1:]:
                    toks = line.split()
                    # Skip rows that are too short
                    if len(toks) < 2:
                        continue
                    lon_val = None
                    lat_val = None
                    # Case A: first token is float-like and second token is float-like -> assume lon lat
                    try:
                        f0 = float(toks[0])
                        f1 = float(toks[1])
                        # Heuristic: if first token is a very large number (> 1e5) it's likely epoch or id,
                        # so treat format as (id lat lon) instead of (lon lat)
                        if abs(f0) > 1e5 and len(toks) >= 3:
                            # likely epoch/id; tokens: id lat lon
                            try:
                                lat_val = float(toks[1]); lon_val = float(toks[2])
                            except Exception:
                                continue
                        else:
                            # assume lon lat
                            lon_val = f0; lat_val = f1
                    except Exception:
                        # first token isn't float -> could be station name; try tokens 1 and 2
                        if len(toks) >= 3:
                            try:
                                lat_val = float(toks[1]); lon_val = float(toks[2])
                            except Exception:
                                # fall back to skip
                                continue
                        else:
                            continue

                    if lon_val is None or lat_val is None:
                        continue
                    # Add to set as tuple (lon, lat) rounded to 6 decimals to merge near-identical points
                    stations.add((round(float(lon_val), 6), round(float(lat_val), 6)))
                    count += 1
                print(f"[DEBUG] Parsed {count} station rows from {station_file}")
            except (IOError, StopIteration, ValueError) as e:
                print(f"Warning: Could not read or parse station file {station_file}: {e}")
    print(f"[DEBUG] Total unique stations found: {len(stations)}")
    # Convert set to sorted numpy array for consistent ordering
    if not stations:
        return np.empty((0, 2))
    stations_list = sorted(list(stations), key=lambda x: (x[0], x[1]))
    return np.array(stations_list)

def read_source_locations(day_folders, seismicflag):
    """Reads all source locations and ellipse data from SourceLocs files.

    Tries to read `nTri` and `gridFit` columns if present. Returns an array with columns
    [lat, lon, emin_ax, emaj_ax, el_ang, vslow, nTri, gridFit]. If columns are not
    available, they will be filled with -1 (nTri) or 0 (gridFit).
    """
    sources = []
    for day_folder in day_folders:
        search_path = os.path.join(day_folder, f"SourceLocs_*_flag{seismicflag}.txt")
        source_files = glob.glob(search_path)
        print(f"[DEBUG] Searching for source files in {day_folder}: {source_files}")
        for source_file in source_files:
            try:
                # Read Lat, Lon, EMinAx, EMajAx, ElAngl, Vslow, nTri, gridFit
                # Column indices: 5=Lat, 6=Lon, 7=EMinAx, 8=EMajAx, 9=ElAngl, 11=Vslow, 16=nTri, 20=gridFit
                # Try to read all columns including gridFit
                candidate_indices = [16, 13, 14, 15, 12, 17, 18, 19]
                data = None
                for idx in candidate_indices:
                    try:
                        # Try to read with gridFit column (20)
                        tmp = np.loadtxt(source_file, skiprows=1, usecols=(5, 6, 7, 8, 9, 11, idx, 20), ndmin=2)
                        if tmp.ndim == 1:
                            tmp = tmp.reshape(1, -1)
                        if tmp.shape[1] < 7:
                            continue
                        # sanity-check: nTri column should be integer-like for most rows
                        ntri_col = tmp[:, 6]
                        finite = np.isfinite(ntri_col)
                        if finite.sum() == 0:
                            continue
                        near_int = np.abs(ntri_col[finite] - np.round(ntri_col[finite])) < 1e-3
                        if np.count_nonzero(near_int) >= max(1, int(0.5 * finite.sum())):
                            med = float(np.median(np.abs(ntri_col[finite])))
                            if med > 200:
                                continue
                            data = tmp
                            print(f"[DEBUG] Read {data.shape[0]} sources (nTri at col {idx}, gridFit at col 20) from {source_file}")
                            break
                    except (ValueError, IndexError):
                        # gridFit column might not exist, try without it
                        try:
                            tmp = np.loadtxt(source_file, skiprows=1, usecols=(5, 6, 7, 8, 9, 11, idx), ndmin=2)
                            if tmp.ndim == 1:
                                tmp = tmp.reshape(1, -1)
                            if tmp.shape[1] < 7:
                                continue
                            ntri_col = tmp[:, 6]
                            finite = np.isfinite(ntri_col)
                            if finite.sum() == 0:
                                continue
                            near_int = np.abs(ntri_col[finite] - np.round(ntri_col[finite])) < 1e-3
                            if np.count_nonzero(near_int) >= max(1, int(0.5 * finite.sum())):
                                med = float(np.median(np.abs(ntri_col[finite])))
                                if med > 200:
                                    continue
                                # Add gridFit=0 (unknown/OK) column
                                data = np.hstack((tmp, np.zeros((tmp.shape[0], 1))))
                                print(f"[DEBUG] Read {data.shape[0]} sources (nTri at col {idx}, no gridFit) from {source_file}")
                                break
                        except (ValueError, IndexError):
                            continue

                if data is None:
                    # Fallback: older/simple layout without nTri or gridFit
                    try:
                        tmp = np.loadtxt(source_file, skiprows=1, usecols=(5, 6, 7, 8, 9, 11), ndmin=2)
                        if tmp.ndim == 1:
                            tmp = tmp.reshape(1, -1)
                        # Add nTri=-1 and gridFit=0 columns
                        data = np.hstack((tmp, np.full((tmp.shape[0], 1), -1.0), np.zeros((tmp.shape[0], 1))))
                        print(f"[DEBUG] Read {data.shape[0]} sources (no nTri, no gridFit) from {source_file}")
                    except ValueError:
                        raise
                print(f"[DEBUG] Read {data.shape[0]} sources from {source_file}")
                if data.size > 0:
                    sources.extend(data)
            except (IOError, StopIteration, ValueError) as e:
                print(f"Warning: Could not read or parse source file {source_file}: {e}")
    print(f"[DEBUG] Total sources found: {len(sources)}")
    return np.array(sources) if sources else np.empty((0, 8))

def read_coarse_locations(day_folders, seismicflag):
    """Reads all coarse locations from CoarseLocs files (lat, lon, min_ax, maj_ax, ang, vslow)."""
    coarse_sources = []
    for day_folder in day_folders:
        # Accept multiple historical filename variants (CoarseLocs_*, CoarseLocations_*)
        patterns = [f"CoarseLocs_*_flag{seismicflag}.txt", f"CoarseLocations_*_flag{seismicflag}.txt", f"Coarse_*_flag{seismicflag}.txt"]
        coarse_files = []
        for p in patterns:
            coarse_files.extend(glob.glob(os.path.join(day_folder, p)))
        # ensure unique
        coarse_files = sorted(set(coarse_files))
        print(f"[DEBUG] Searching for coarse files in {day_folder}: {coarse_files}")
        for coarse_file in coarse_files:
            try:
                # We'll try several fallbacks to be permissive about historical file layouts:
                # 1) files that contain lat, lon, min_ax, maj_ax, ang, vslow (preferred)
                # 2) files that contain lat, lon and nTri only (fallback)
                data = None
                # try preferred layout: lat, lon, min_ax, maj_ax, ang, vslow, optional nTri
                try:
                    tmp = np.loadtxt(coarse_file, skiprows=1, usecols=(5, 6, 7, 8, 9, 11), ndmin=2)
                    if tmp.ndim == 1:
                        tmp = tmp.reshape(1, -1)
                    # Append nTri column if missing
                    if tmp.shape[1] == 6:
                        tmp = np.hstack((tmp, np.full((tmp.shape[0], 1), -1.0)))
                    data = tmp
                    print(f"[DEBUG] Read {data.shape[0]} coarse sources (preferred layout) from {coarse_file}")
                except Exception:
                    # fallback: parse minimal layout like "Date HH MM SS Lat Lon nTri ..."
                    try:
                        # Expect at least 6 tokens: Date, HH, MM, SS, Lat, Lon, ... -> read cols 4 and 5
                        tmp2 = np.loadtxt(coarse_file, skiprows=1, usecols=(4, 5, 6), ndmin=2)
                        if tmp2.ndim == 1:
                            tmp2 = tmp2.reshape(1, -1)
                        # tmp2 columns: lat, lon, nTri (if present)
                        lat = tmp2[:, 0]
                        lon = tmp2[:, 1]
                        # nTri may or may not be present in column 2; if not, set -1
                        ntri_col = tmp2[:, 2] if tmp2.shape[1] > 2 else np.full(tmp2.shape[0], -1.0)
                        # construct array with default ellipse and vslow placeholders
                        min_ax = np.full(lat.shape, 0.1)
                        maj_ax = np.full(lat.shape, 0.1)
                        ang = np.full(lat.shape, 0.0)
                        vslow = np.full(lat.shape, -1.0)
                        data = np.vstack((lat, lon, min_ax, maj_ax, ang, vslow, ntri_col)).T
                        print(f"[DEBUG] Read {data.shape[0]} coarse sources (minimal layout) from {coarse_file}")
                    except Exception:
                        # Could not parse file - skip with a warning
                        print(f"Warning: Could not parse coarse file {coarse_file} with either preferred or minimal layout.")
                        continue
                if data is not None:
                    coarse_sources.extend(data)
            except (IOError, StopIteration, ValueError) as e:
                print(f"Warning: Could not read or parse coarse file {coarse_file}: {e}")
    print(f"[DEBUG] Total coarse sources found: {len(coarse_sources)}")
    return np.array(coarse_sources) if coarse_sources else np.empty((0, 7))


def compute_event_quality(event_row, min_ntri=3, exclude_zero_vslow=True, max_emaj_km=200.0, max_emaj_ratio=6.0, exclude_grid_edge=False, max_distance_km=None, array_center=None):
    """Return a tuple (pass_bool, flags_list) describing quality checks for a single event.

    event_row must be array-like with columns: lat, lon, emin_ax, emaj_ax, el_ang, vslow, nTri, gridFit
    (nTri may be -1 if unavailable, gridFit=1 means solution hit grid edge).
    Flags include: 'ZERO_VSLOW', 'LOW_NTRI', 'LARGE_EMAJ', 'EMAJ_RATIO', 'BAD_COORDS', 'GRID_EDGE', 'TOO_FAR'
    
    max_distance_km: If set, reject events farther than this from array_center.
    array_center: (lat, lon) tuple for the array center (required if max_distance_km is set).
    """
    flags = []
    try:
        lat = float(event_row[0]); lon = float(event_row[1])
        emin = float(event_row[2]); emaj = float(event_row[3])
        # el_ang = event_row[4]
        vslow = float(event_row[5])
        ntri = int(event_row[6]) if len(event_row) > 6 else -1
        gridfit = int(event_row[7]) if len(event_row) > 7 else 0
    except Exception:
        return False, ['BAD_ROW']

    # Basic coordinate sanity
    if np.isnan(lat) or np.isnan(lon) or not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        flags.append('BAD_COORDS')

    # Zero or invalid slowness
    if exclude_zero_vslow and (np.isnan(vslow) or vslow <= 0):
        flags.append('ZERO_VSLOW')

    # nTri based checks, only if available (>0)
    if ntri >= 0 and ntri < min_ntri:
        flags.append('LOW_NTRI')

    # EMaj size
    if not np.isnan(emaj) and emaj > max_emaj_km:
        flags.append('LARGE_EMAJ')

    # major/minor axis ratio
    if emin > 0 and not np.isnan(emaj) and (emaj / emin) > max_emaj_ratio:
        flags.append('EMAJ_RATIO')

    # Grid edge flag - solution hit edge of search grid (likely unreliable)
    if exclude_grid_edge and gridfit == 1:
        flags.append('GRID_EDGE')

    # Maximum distance from array center
    if max_distance_km is not None and array_center is not None:
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        try:
            _, _, dist_m = geod.inv(array_center[1], array_center[0], lon, lat)
            dist_km = abs(dist_m) / 1000.0
            if dist_km > max_distance_km:
                flags.append('TOO_FAR')
        except Exception:
            pass  # Can't compute distance, don't filter

    passed = len(flags) == 0
    return passed, flags


def evaluate_vslow_acceptance(vslow_raw, vslow_phase='NONE', vslow_tol=0.4, vslow_min_override=None, vslow_max_override=None, auto_detect_units=False):
    """Return (accepted_bool, scaled_vslow (s/km or np.nan), units_str, reason_str).

    This is a module-level helper (suitable for unit tests) implementing the same
    behavior used by main() when doing phase-based vslow acceptance:
    - If vslow_raw <= 0 -> reject.
    - If auto-detect and value < 0.01, treat as s/m and scale *1000 to s/km.
    - If vslow_min_override/vslow_max_override present use those explicit bounds.
    - If vslow_phase is NONE accept any positive value (unless outside explicit bounds).
    - For P/S/SURF phases use canonical slowness with fractional tolerance.
    """
    reason = ''
    try:
        v = float(vslow_raw)
    except Exception:
        return False, np.nan, 'unknown', 'VSLOW_NONNUMERIC'
    if np.isnan(v) or v <= 0:
        return False, v, 's/km', 'VSLOW_NONPOSITIVE'

    units = 's/km'
    scaled = v
    # Heuristic conversion: values < 0.01 are likely s/m -> convert to s/km
    if v < 0.01 and auto_detect_units:
        scaled = v * 1000.0
        units = 's/m->s/km'
        reason = 'CONVERTED_S/M_TO_S/KM'

    # Explicit bounds override
    if vslow_min_override is not None or vslow_max_override is not None:
        low = vslow_min_override if vslow_min_override is not None else -np.inf
        high = vslow_max_override if vslow_max_override is not None else np.inf
        if scaled < low or scaled > high:
            return False, scaled, units, f'VSLOW_OUTSIDE_EXPLICIT_BOUNDS ({low},{high})'
        return True, scaled, units, reason or 'VSLOW_WITHIN_EXPLICIT_BOUNDS'

    if vslow_phase is None:
        vslow_phase = 'NONE'
    vslow_phase = vslow_phase.upper()

    if vslow_phase == 'NONE':
        return True, scaled, units, reason or 'VSLOW_ACCEPTED_NO_PHASE_CHECK'

    phase_canon = {'P': 1.0 / 6.0, 'S': 1.0 / 3.5, 'SURF': 1.0 / 3.0}
    if vslow_phase not in phase_canon:
        return True, scaled, units, 'VSLOW_ACCEPTED_UNKNOWN_PHASE'

    nominal = phase_canon[vslow_phase]
    low = nominal * (1.0 - float(vslow_tol))
    high = nominal * (1.0 + float(vslow_tol))
    if scaled < low or scaled > high:
        return False, scaled, units, f'VSLOW_OUT_OF_RANGE_PHASE({vslow_phase} {low:.4f}-{high:.4f})'
    return True, scaled, units, reason or f'VSLOW_WITHIN_PHASE_RANGE({vslow_phase})'

def create_projection(avg_lat):
    """Creates a suitable map projection based on average latitude."""
    if abs(avg_lat) > 60:
        return ccrs.SouthPolarStereo() if avg_lat < 0 else ccrs.NorthPolarStereo()
    return ccrs.PlateCarree()




def plot_map(stations, sources, seismicflag, project_folder, no_stations=False, source_dot_size=20, include_zero_vslow=False, color_by_ntri=False, output_prefix=None, usgs_matches=None):
    """Creates and saves a map of all stations, sources, and error ellipses.
    
    If color_by_ntri is True, source points will be colored by their nTri value.
    If output_prefix is set, uses that instead of default 'antarctic_aggregated_map_flag{seismicflag}'.
    """
    if sources.size == 0:
        print(f"No source locations found for seismicflag {seismicflag}. Cannot generate plot.")
        return

    # Determine map bounds from event (source) locations only
    if sources.size > 0:
        event_lons = sources[:, 1]
        event_lats = sources[:, 0]
    else:
        print("No event locations to plot.")
        return

    # Calculate tight bounds with minimal padding
    min_lon, max_lon = np.min(event_lons), np.max(event_lons)
    min_lat, max_lat = np.min(event_lats), np.max(event_lats)
    avg_lat = np.mean(event_lats)
    # Padding: 2% of range, but at least 0.05 deg
    lon_pad = max((max_lon - min_lon) * 0.02, 0.05)
    lat_pad = max((max_lat - min_lat) * 0.02, 0.05)

    if avg_lat < -60:  # Antarctic region
        projection = ccrs.SouthPolarStereo()
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([
            min_lon - lon_pad, max_lon + lon_pad,
            max(min_lat - lat_pad, -90), min(max_lat + lat_pad, -60)
        ], crs=ccrs.PlateCarree())
    else:
        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([
            min_lon - lon_pad, max_lon + lon_pad,
            min_lat - lat_pad, max_lat + lat_pad
        ], crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    
    # Add ice feature for Antarctic regions
    if avg_lat < -60:
        try:
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m',
                                                       facecolor='white', alpha=0.8))
        except:
            pass  # Feature might not be available
    
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)

    # Plot unique station locations (unless suppressed)
    if not no_stations and stations.size > 0:
        ax.scatter(stations[:, 0], stations[:, 1], 
                  c='black', marker='^', s=40, 
                  transform=ccrs.PlateCarree(), 
                  label=f'Stations ({len(stations)})', 
                  zorder=5, edgecolors='white', linewidth=1)

    # Plot all valid sources as black circles, no velocity color-coding
    print(f"[DEBUG] sources.shape: {sources.shape}, sources.size: {sources.size}")
    print(f"[DEBUG] stations.shape: {stations.shape}, stations.size: {stations.size}")
    source_count = 0
    ellipse_count = 0
    valid_sources = []
    for idx, src in enumerate(sources):
        if len(src) not in (6, 7, 8):
            print(f"[DEBUG] Skipping malformed source at index {idx}: {src}")
            continue
        # Support 6 columns [lat,lon,emin,emaj,elang,vslow], 
        # 7 columns with nTri, or 8 columns with nTri and gridFit.
        # For plotting we only care about the first six fields.
        lat, lon, min_ax, maj_ax, ang, vslow = src[:6]
        # Skip malformed rows
        if np.isnan([lat, lon, min_ax, maj_ax, ang, vslow]).any():
            print(f"[DEBUG] Skipping invalid source at index {idx}: {src}")
            continue
        # Treat vslow <= 0 as invalid by default; optionally include in plotting
        if vslow <= 0 and not include_zero_vslow:
            print(f"[DEBUG] Skipping invalid (zero) vslow at index {idx}: {src}")
            continue
        print(f"[DEBUG] Valid source {idx}: lat={lat}, lon={lon}, min_ax={min_ax}, maj_ax={maj_ax}, ang={ang}, vslow={vslow}")
        valid_sources.append(src)
        source_count += 1
    print(f"[DEBUG] Valid sources for plotting: {len(valid_sources)}")
    valid_sources = np.array(valid_sources)
    cbar = None  # Will be set if color_by_ntri is enabled
    if valid_sources.size > 0:
        # Check if we should color by nTri
        if color_by_ntri and valid_sources.shape[1] >= 7:
            ntri_values = valid_sources[:, 6]
            # Filter out invalid nTri values (e.g., -1 placeholders)
            valid_ntri_mask = ntri_values >= 0
            if np.any(valid_ntri_mask):
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                cmap = cm.get_cmap('viridis')
                # Use percentile-based range to handle outliers
                valid_ntri = ntri_values[valid_ntri_mask]
                median_ntri = np.median(valid_ntri)
                p5 = np.percentile(valid_ntri, 5)
                p95 = np.percentile(valid_ntri, 95)
                # Extend slightly beyond percentiles for visual range
                vmin = max(1, p5)  # At least 1 triad
                vmax = p95
                # If range is too narrow, use min/max instead
                if vmax <= vmin:
                    vmin = np.min(valid_ntri)
                    vmax = np.max(valid_ntri)
                normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
                print(f"[DEBUG] Plotting {valid_sources.shape[0]} sources colored by nTri (median: {median_ntri:.0f}, colorbar range: {vmin:.0f}-{vmax:.0f}, actual range: {np.min(valid_ntri):.0f}-{np.max(valid_ntri):.0f}).")
                scatter = ax.scatter(valid_sources[:, 1], valid_sources[:, 0], 
                               c=ntri_values, cmap=cmap, norm=normalize,
                               marker='o', s=source_dot_size, 
                               transform=ccrs.PlateCarree(), 
                               zorder=6, edgecolors='white', linewidth=0.5)
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
                cbar.set_label('nTri (number of triads)', fontsize=10)
            else:
                print(f"[DEBUG] No valid nTri values found; falling back to black dots.")
                scatter = ax.scatter(valid_sources[:, 1], valid_sources[:, 0], 
                               c='black', marker='o', s=source_dot_size, 
                               transform=ccrs.PlateCarree(), 
                               zorder=6, edgecolors='white', linewidth=0.5)
        else:
            if color_by_ntri:
                print(f"[DEBUG] color_by_ntri requested but nTri column not available (shape: {valid_sources.shape}); using black dots.")
            print(f"[DEBUG] Plotting {valid_sources.shape[0]} sources as black circles.")
            scatter = ax.scatter(valid_sources[:, 1], valid_sources[:, 0], 
                               c='black', marker='o', s=source_dot_size, 
                               transform=ccrs.PlateCarree(), 
                               zorder=6, edgecolors='white', linewidth=0.5)
        # Plot error ellipses
        for src in valid_sources:
            lat, lon, min_ax, maj_ax, ang, vslow = src[:6]
            km_to_deg = 1.0 / 111.0  # Rough conversion
            ellipse_width = maj_ax * km_to_deg * 2  # Full width
            ellipse_height = min_ax * km_to_deg * 2  # Full height
            ellipse_width = max(ellipse_width, 0.1)
            ellipse_height = max(ellipse_height, 0.1)
            try:
                ellipse = Ellipse(xy=(lon, lat),
                                width=ellipse_width,
                                height=ellipse_height,
                                angle=ang,
                                transform=ccrs.PlateCarree(),
                                facecolor='none',
                                alpha=1.0,
                                edgecolor='darkorange',
                                linewidth=0.8,
                                zorder=4)
                ax.add_patch(ellipse)
                ellipse_count += 1
            except Exception as e:
                print(f"Warning: Could not add ellipse at ({lat}, {lon}): {e}")

    # Create legend
    legend_elements = []
    if not no_stations and stations.size > 0:
        legend_elements.append(plt.scatter([], [], c='black', marker='^', s=40, 
                                         label=f'Stations ({len(stations)})'))
    if source_count > 0:
        # For color_by_ntri mode, note the colorbar explains the colors
        source_label = f'Sources ({source_count})'
        if color_by_ntri and valid_sources is not None and valid_sources.shape[1] >= 7:
            source_label += ' - colored by nTri'
        legend_elements.append(plt.scatter([], [], c='gray', marker='o', s=source_dot_size, 
                                         label=source_label))
    if ellipse_count > 0:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.2, 
                                           edgecolor='darkorange', 
                                           label=f'Error Ellipses ({ellipse_count})'))

    # We'll create the legend including overlays after we've added optional overlays (USGS, invalid vslow)

    # Set title
    signal_types = {0: 'Infrasound', 1: 'Seismic', 3: 'Surface Wave', 4: 'Surface Wave (Alt)'}
    signal_type = signal_types.get(seismicflag, f'Flag {seismicflag}')
    
    title = f'Antarctic {signal_type} Detections\nSources: {source_count}, Stations: {len(stations) if stations.size > 0 else 0}'
    
    ax.set_title(title, fontsize=14, pad=20)
    
    if output_prefix:
        output_filename = os.path.join(project_folder, f'{output_prefix}_map.png')
    else:
        output_filename = os.path.join(project_folder, f'antarctic_aggregated_map_flag{seismicflag}.png')
    # If we're including zero/invalid vslow events for inspection, plot them as faint red x
    if include_zero_vslow:
        invalid_vslow = []
        for src in sources:
            if len(src) not in (6, 7, 8):
                continue
            lat, lon, min_ax, maj_ax, ang, vslow = src[:6]
            if np.isnan([lat, lon]).any():
                continue
            if vslow <= 0:
                invalid_vslow.append((lon, lat))
        if invalid_vslow:
            invalid = np.array(invalid_vslow)
            ax.scatter(invalid[:, 0], invalid[:, 1], c='red', marker='x', s=20, transform=ccrs.PlateCarree(), zorder=7, label='vslow<=0 (shown)')
            print(f"Also plotted {invalid.shape[0]} events with vslow<=0 as red X markers for inspection")

    # Overlay USGS match locations (if provided) - show as squares
    if usgs_matches and len(usgs_matches) > 0:
        try:
            umatch_lats = []
            umatch_lons = []
            for r in usgs_matches:
                try:
                    if isinstance(r, dict):
                        matched = r.get('usgs_match') if 'usgs_match' in r else True
                        lat = float(r.get('lat')) if 'lat' in r else None
                        lon = float(r.get('lon')) if 'lon' in r else None
                    else:
                        matched = True
                        lat = float(r[3]) if len(r) > 3 else None
                        lon = float(r[4]) if len(r) > 4 else None
                    if matched and lat is not None and lon is not None:
                        umatch_lats.append(lat)
                        umatch_lons.append(lon)
                except Exception:
                    continue
            if umatch_lats:
                # Optionally annotate events by color/marker but for now use magenta squares
                ax.scatter(umatch_lons, umatch_lats, c='magenta', marker='s', s=100, transform=ccrs.PlateCarree(), zorder=8, edgecolors='black', linewidth=1, label='USGS match')
                print(f"Overlayed {len(umatch_lats)} USGS matches on the map")
                # Print a short summary by matched phase if available
                try:
                    phase_counts = {}
                    for r in usgs_matches:
                        mp = r.get('matched_phase') if isinstance(r, dict) else ''
                        if mp is None or mp == '':
                            mp = 'unknown'
                        phase_counts[mp] = phase_counts.get(mp, 0) + 1
                    print(f"USGS matched event phase counts: {phase_counts}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Could not overlay USGS matches: {e}")

    # Add any optional overlay legend entries
    try:
        if 'invalid' in locals() and invalid.size > 0:
            legend_elements.append(plt.scatter([], [], c='red', marker='x', s=20, label='vslow<=0 (shown)'))
        if 'umatch_lats' in locals() and len(umatch_lats) > 0:
            legend_elements.append(plt.scatter([], [], c='magenta', marker='s', s=100, label='USGS match'))
    except Exception:
        pass

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Antarctic-focused map saved to {output_filename}")
    print(f"Plotted {source_count} sources, {len(stations) if stations.size > 0 else 0} stations, {ellipse_count} ellipses")

    # Done - image saved & closed above

def plot_stationxml(filename, output_png):
    """Plot stations from a StationXML file as black symbols with unfilled ellipses."""
    try:
        from obspy import read_inventory
    except ImportError:
        print("obspy is required to read StationXML files. Please install it with 'pip install obspy'.")
        return
    inv = read_inventory(filename)
    lons = []
    lats = []
    for net in inv:
        for sta in net:
            if sta.longitude is not None and sta.latitude is not None:
                lons.append(sta.longitude)
                lats.append(sta.latitude)
    if not lons:
        print(f"No station locations found in {filename}.")
        return
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    avg_lat = np.mean(lats)
    lon_pad = max((max_lon - min_lon) * 0.02, 0.05)
    lat_pad = max((max_lat - min_lat) * 0.02, 0.05)
    if avg_lat < -60:
        projection = ccrs.SouthPolarStereo()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([
            min_lon - lon_pad, max_lon + lon_pad,
            max(min_lat - lat_pad, -90), min(max_lat + lat_pad, -60)
        ], crs=ccrs.PlateCarree())
    else:
        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([
            min_lon - lon_pad, max_lon + lon_pad,
            min_lat - lat_pad, max_lat + lat_pad
        ], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)
    # Plot stations as black circles
    ax.scatter(lons, lats, c='black', marker='o', s=40, transform=ccrs.PlateCarree(), label='Stations', zorder=5, edgecolors='white', linewidth=1)
    # Plot unfilled ellipses at each station (fixed size for demo)
    from matplotlib.patches import Ellipse
    for lon, lat in zip(lons, lats):
        ellipse = Ellipse(xy=(lon, lat), width=0.2, height=0.1, angle=0, transform=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='black', linewidth=1.2, zorder=4)
        ax.add_patch(ellipse)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.set_title(f'Stations from {os.path.basename(filename)}', fontsize=14, pad=20)
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"StationXML map saved to {output_png}")


def plot_coarse_map(coarse_sources, seismicflag, project_folder):
    """Creates and saves a map of all coarse sources (no stations)."""
    if coarse_sources.size == 0:
        print(f"No coarse locations found for seismicflag {seismicflag}. Cannot generate coarse map.")
        return

    event_lons = coarse_sources[:, 1]
    event_lats = coarse_sources[:, 0]
    min_lon, max_lon = np.min(event_lons), np.max(event_lons)
    min_lat, max_lat = np.min(event_lats), np.max(event_lats)
    avg_lat = np.mean(event_lats)
    # Padding: 2% of range, at least 0.05 deg
    lon_pad = max((max_lon - min_lon) * 0.02, 0.05)
    lat_pad = max((max_lat - min_lat) * 0.02, 0.05)
    if avg_lat < -60:
        projection = ccrs.SouthPolarStereo()
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([
            min_lon - lon_pad, max_lon + lon_pad,
            max(min_lat - lat_pad, -90), min(max_lat + lat_pad, -60)
        ], crs=ccrs.PlateCarree())
    else:
        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_extent([
            min_lon - lon_pad, max_lon + lon_pad,
            min_lat - lat_pad, max_lat + lat_pad
        ], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    if avg_lat < -60:
        try:
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m',
                                                       facecolor='white', alpha=0.8))
        except:
            pass
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)

    # Split into sources with valid vslow (for color) and those without (plot as black dots)
    velocities = []
    valid_sources = []
    no_vslow_sources = []
    for idx, src in enumerate(coarse_sources):
        if len(src) < 6:
            print(f"[DEBUG] Skipping malformed coarse source at index {idx}: {src}")
            continue
        lat, lon, min_ax, maj_ax, ang = src[:5]
        vslow = src[5] if len(src) > 5 else -1
        # Always record the source; use vslow to decide coloring
        if np.isnan(lat) or np.isnan(lon):
            print(f"[DEBUG] Skipping invalid coarse source at index {idx}: {src}")
            continue
        if np.isfinite(vslow) and vslow > 0:
            velocity = 1.0 / vslow
            velocities.append(velocity)
            valid_sources.append(src)
        else:
            no_vslow_sources.append(src)
    velocities = np.array(velocities) if velocities else np.array([])
    valid_sources = np.array(valid_sources) if valid_sources else np.array([])
    no_vslow_sources = np.array(no_vslow_sources) if no_vslow_sources else np.array([])
    if velocities.size > 0:
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        cmap = cm.get_cmap('plasma')
        normalize = colors.Normalize(vmin=np.min(velocities), vmax=np.max(velocities))
        scatter = ax.scatter(valid_sources[:, 1], valid_sources[:, 0],
                           c=velocities, cmap=cmap, norm=normalize,
                           marker='o', s=60,
                           transform=ccrs.PlateCarree(),
                           zorder=6, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label('Velocity (km/s)', rotation=270, labelpad=20, fontsize=12)
        # Plot error ellipses
        ellipse_count = 0
        for src in valid_sources:
            lat, lon, min_ax, maj_ax, ang, vslow = src[:6]
            km_to_deg = 1.0 / 111.0
            ellipse_width = maj_ax * km_to_deg * 2
            ellipse_height = min_ax * km_to_deg * 2
            ellipse_width = max(ellipse_width, 0.1)
            ellipse_height = max(ellipse_height, 0.1)
            try:
                ellipse = Ellipse(xy=(lon, lat),
                                width=ellipse_width,
                                height=ellipse_height,
                                angle=ang,
                                transform=ccrs.PlateCarree(),
                                facecolor='orange',
                                alpha=0.2,
                                edgecolor='darkorange',
                                linewidth=0.8,
                                zorder=4)
                ax.add_patch(ellipse)
                ellipse_count += 1
            except Exception as e:
                print(f"Warning: Could not add coarse ellipse at ({lat}, {lon}): {e}")
    else:
        ellipse_count = 0

    # Legend for sources and ellipses
    legend_elements = []
    if velocities.size > 0:
        legend_elements.append(plt.scatter([], [], c='gray', marker='o', s=60,
                                         label=f'Coarse Sources ({len(valid_sources)})'))
    if ellipse_count > 0:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.2,
                                           edgecolor='darkorange',
                                           label=f'Error Ellipses ({ellipse_count})'))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=10)

    signal_types = {0: 'Infrasound', 1: 'Seismic', 3: 'Surface Wave', 4: 'Surface Wave (Alt)'}
    signal_type = signal_types.get(seismicflag, f'Flag {seismicflag}')
    title = f'Antarctic {signal_type} Coarse Detections\nCoarse Sources: {len(valid_sources)}'
    if velocities.size > 0:
        title += f'\nVelocity Range: {np.min(velocities):.2f} - {np.max(velocities):.2f} km/s'
    ax.set_title(title, fontsize=14, pad=20)
    output_filename = os.path.join(project_folder, f'antarctic_coarse_map_flag{seismicflag}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Antarctic coarse map saved to {output_filename}")
    if velocities.size > 0:
        print(f"Coarse velocity statistics: min={np.min(velocities):.2f}, max={np.max(velocities):.2f}, mean={np.mean(velocities):.2f} km/s")
    total_plotted = len(valid_sources) + len(no_vslow_sources)
    print(f"Plotted {total_plotted} coarse sources ({len(valid_sources)} with vslow), {ellipse_count} ellipses (no stations)")


def main():

    parser = argparse.ArgumentParser(description="Generate an aggregated plot of source and station locations for a given seismicflag, or plot stations from a StationXML file.")
    parser.add_argument('--project_folder', type=str, required=True, help='Path to the project data folder.')
    parser.add_argument('--seismicflag', type=int, required=True, choices=[0, 1, 3, 4, 5, 6], help='Seismic flag to process.')
    parser.add_argument('--plot_coarse', action='store_true', help='Also generate a map of coarse locations (no stations).')
    parser.add_argument('--stationxml', type=str, default=None, help='Path to the StationXML file. If provided, plot stations from this file instead of text files.')
    parser.add_argument('--output_png', type=str, default='stationxml_map.png', help='Output PNG filename for StationXML plot.')
    parser.add_argument('--no_filtered', action='store_true', help='Do not write a filtered aggregated CSV (default: write filtered CSV).')
    parser.add_argument('--min_ntri', type=int, default=3, help='Minimum number of triads required (when available) to keep an event in filtered CSV (default: 3).')
    parser.add_argument('--max_emaj_km', type=float, default=200.0, help='Maximum EMaj (km) allowed for filtered CSV (default: 200 km).')
    parser.add_argument('--max_emaj_ratio', type=float, default=6.0, help='Maximum EMaj/EMin ratio allowed for filtered CSV (default: 6.0).')
    parser.add_argument('--allow_zero_vslow', action='store_true', help='Allow vslow <= 0 values to remain in filtered CSV (default: exclude them).')
    parser.add_argument('--vslow_accept_phase', type=str, default='NONE', choices=['NONE', 'P', 'S', 'SURF'], help='Only accept events whose vslow falls plausibly within the chosen phase range. NONE disables this check (default: NONE).')
    parser.add_argument('--vslow_tolerance_frac', type=float, default=0.4, help='Fractional tolerance around canonical phase slowness for vslow acceptance (default: 0.4 = ±40%%).')
    parser.add_argument('--vslow_min', type=float, default=None, help='Explicit minimum vslow (s/km) to accept. Overrides phase/tolerance lower bound when set.')
    parser.add_argument('--vslow_max', type=float, default=None, help='Explicit maximum vslow (s/km) to accept. Overrides phase/tolerance upper bound when set.')
    parser.add_argument('--vslow_auto_detect_units', action='store_true', help='Try to auto-detect and convert vslow units (e.g. s/m → s/km) when values look too small (default: False).')
    parser.add_argument('--no_stations', action='store_true', help='Do not plot stations on the map.')
    parser.add_argument('--plot_include_zero_vslow', action='store_true', help='Include events with vslow <= 0 in the plotted map (they are marked as red X).')
    parser.add_argument('--source_dot_size', type=int, default=20, help='Dot size for source locations (default: 20).')
    parser.add_argument('--color_by_ntri', action='store_true', help='Color source scatter points by nTri value (requires nTri column in data).')
    parser.add_argument('--plot_filtered_map', action='store_true', help='Also generate a map for the filtered aggregated CSV (after quality and vslow filters).')
    parser.add_argument('--usgs_match_csv', type=str, default=None, help='Optional CSV with USGS match info (from scripts/postprocess_usgs_matches.py) to overlay matches as squares.')
    parser.add_argument('--exclude_grid_edge', action='store_true', help='Exclude events where solution hit edge of search grid (gridFit=1). These are often unreliable.')
    parser.add_argument('--max_distance_km', type=float, default=None, help='Maximum distance (km) from array center to accept. Events farther away are filtered out.')
    parser.add_argument('--array_center', type=str, default=None, help='Array center as "lat,lon" (e.g., "-77.5,-103.5"). Required if --max_distance_km is set.')
    parser.add_argument('--output_prefix', type=str, default=None, help='Prefix for output filenames (e.g., "flag6_nofilter" produces flag6_nofilter_sources.csv). If not set, uses default naming.')
    args = parser.parse_args()

    # Use StationXML for stations if provided, else use text files
    if args.stationxml:
        try:
            from obspy import read_inventory
        except ImportError:
            print("obspy is required to read StationXML files. Please install it with 'pip install obspy'.")
            return
        inv = read_inventory(args.stationxml)
        stations = []
        for net in inv:
            for sta in net:
                if sta.longitude is not None and sta.latitude is not None:
                    stations.append((sta.longitude, sta.latitude))
        stations = np.array(stations) if stations else np.empty((0, 2))
    else:
        day_folders = get_day_folders(args.project_folder)
        if not day_folders:
            print("No valid day folders found in the project directory.")
            return
        stations = read_station_locations(day_folders, args.seismicflag)

    # Always get sources for plotting
    if not args.stationxml:
        sources = read_source_locations(day_folders, args.seismicflag)
    else:
        # If using stationxml, still need day_folders for sources
        day_folders = get_day_folders(args.project_folder)
        sources = read_source_locations(day_folders, args.seismicflag)

    # Try to load USGS matches if requested
    usgs_matches = None
    if args.usgs_match_csv:
        if os.path.exists(args.usgs_match_csv):
            try:
                with open(args.usgs_match_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = []
                    for r in reader:
                        # Only include rows where usgs_match is true
                        um = str(r.get('usgs_match', '')).lower()
                        if um in ('true', '1', 't', 'yes', 'y'):
                            # lat/lon are columns named 'lat' and 'lon' per script output
                            try:
                                lat = float(r.get('lat'))
                                lon = float(r.get('lon'))
                            except Exception:
                                # If lat/ lon missing use usgs_lat/usgs_lon if present
                                try:
                                    lat = float(r.get('usgs_lat'))
                                    lon = float(r.get('usgs_lon'))
                                except Exception:
                                    continue
                            rows.append({'lat': lat, 'lon': lon, 'usgs_id': r.get('usgs_id'), 'usgs_mag': r.get('usgs_mag'), 'matched_phase': r.get('matched_phase', ''), 'arrival_diff_sec': r.get('arrival_diff_sec', '')})
                    usgs_matches = rows
                    print(f"Loaded {len(rows)} USGS matches from {args.usgs_match_csv}")
            except Exception as e:
                print(f"Warning: Could not read USGS matches CSV {args.usgs_match_csv}: {e}")
        else:
            print(f"USGS matches CSV not found: {args.usgs_match_csv}")

    plot_map(stations, sources, args.seismicflag, args.project_folder, no_stations=args.no_stations, source_dot_size=args.source_dot_size, include_zero_vslow=args.plot_include_zero_vslow, color_by_ntri=args.color_by_ntri, output_prefix=args.output_prefix, usgs_matches=usgs_matches)

    # --- New behavior: write aggregated CSVs for sources so user can inspect combined locations ---
    try:
        if sources is not None and sources.size > 0:
            # Ensure numpy array shape is correct
            src_array = np.asarray(sources)
            if src_array.ndim == 1:
                src_array = src_array.reshape(1, -1)

            # Determine output prefix for CSV files
            csv_prefix = args.output_prefix if args.output_prefix else f'aggregated_sources_flag{args.seismicflag}'
            
            out_all = os.path.join(args.project_folder, f'{csv_prefix}.csv')
            header = 'lat,lon,emin_ax,emaj_ax,el_ang,vslow,nTri,gridFit'
            # Save the raw aggregated list
            np.savetxt(out_all, src_array, delimiter=',', header=header, comments='', fmt='%.6f')
            print(f"Wrote aggregated sources CSV: {out_all} ({src_array.shape[0]} rows)")

            # Also write a deduplicated CSV (rounded to 6 decimals to merge near-identical entries)
            rounded = np.round(src_array, 6)
            unique_rows = np.unique(rounded, axis=0)
            out_unique = os.path.join(args.project_folder, f'{csv_prefix}_unique.csv')
            np.savetxt(out_unique, unique_rows, delimiter=',', header=header, comments='', fmt='%.6f')
            print(f"Wrote deduplicated aggregated sources CSV: {out_unique} ({unique_rows.shape[0]} unique rows)")
        else:
            print(f"No sources found; skipping CSV output for flag {args.seismicflag}.")
    except Exception as e:
        print(f"Warning: could not write aggregated source CSVs: {e}")

    # Optionally write a filtered CSV for easier downstream analysis. By default this is written
    # but can be disabled via --no_filtered on the command line.
    if not args.no_filtered:
        if sources is not None and sources.size > 0:
            try:
                # Ensure numpy array shape is correct (again, safe to re-create)
                src_array = np.asarray(sources)
                if src_array.ndim == 1:
                    src_array = src_array.reshape(1, -1)

                # Construct array with nTri and gridFit columns in case input was shape (N,6) or (N,7)
                if src_array.shape[1] == 6:
                    # Missing both nTri and gridFit
                    src_aug = np.hstack((src_array, np.full((src_array.shape[0], 1), -1.0), np.zeros((src_array.shape[0], 1))))
                elif src_array.shape[1] == 7:
                    # Missing gridFit only
                    src_aug = np.hstack((src_array, np.zeros((src_array.shape[0], 1))))
                else:
                    src_aug = src_array

                # Use CLI-provided thresholds
                min_ntri = int(args.min_ntri)
                max_emaj_km = float(args.max_emaj_km)
                max_emaj_ratio = float(args.max_emaj_ratio)

                # Compute quality and filter
                passes = []
                from collections import Counter
                flag_counts = Counter()
                # We'll also optionally apply a vslow phase-acceptance check. Build results to write
                vslow_phase = args.vslow_accept_phase.upper() if args.vslow_accept_phase else 'NONE'
                vslow_tol = float(args.vslow_tolerance_frac)
                vslow_min_override = float(args.vslow_min) if args.vslow_min is not None else None
                vslow_max_override = float(args.vslow_max) if args.vslow_max is not None else None
                vslow_auto_units = bool(args.vslow_auto_detect_units)
                exclude_grid_edge = bool(args.exclude_grid_edge)
                
                # Parse max distance and array center for distance filtering
                max_distance_km = float(args.max_distance_km) if args.max_distance_km is not None else None
                array_center = None
                if args.array_center:
                    try:
                        lat_str, lon_str = args.array_center.split(',')
                        array_center = (float(lat_str), float(lon_str))
                        print(f"[INFO] Using array center: {array_center}, max distance: {max_distance_km} km")
                    except ValueError:
                        print(f"[WARN] Could not parse --array_center '{args.array_center}', expected 'lat,lon' format")

                # Use module-level evaluate_vslow_acceptance helper for checks
                need_vslow_check = (vslow_phase and vslow_phase.upper() != 'NONE') or (vslow_min_override is not None) or (vslow_max_override is not None)

                for r in src_aug:
                    ok, flags = compute_event_quality(r, min_ntri=min_ntri, exclude_zero_vslow=not args.allow_zero_vslow,
                                                     max_emaj_km=max_emaj_km, max_emaj_ratio=max_emaj_ratio,
                                                     exclude_grid_edge=exclude_grid_edge,
                                                     max_distance_km=max_distance_km, array_center=array_center)
                    # Additionally check vslow-based phase acceptance if requested
                    vs_val = float(r[5]) if len(r) > 5 else np.nan
                    v_accept, v_scaled, v_units, v_reason = evaluate_vslow_acceptance(vs_val, vslow_phase=vslow_phase, vslow_tol=vslow_tol, vslow_min_override=vslow_min_override, vslow_max_override=vslow_max_override, auto_detect_units=vslow_auto_units)
                    # If there's any vslow acceptance condition (phase or explicit bounds) and the event fails, mark as a failing flag
                    if need_vslow_check:
                        if not v_accept:
                            flags.append('VSLOW_REJECT')
                            flags.append(v_reason)
                    passes.append(ok)
                    for f in flags:
                        flag_counts[f] += 1

                passes = np.array(passes, dtype=bool)
                # Filter out events which did not pass compute_event_quality or vslow checks
                filtered = src_aug[passes]
                # If vslow acceptance conditions are enabled (phase or explicit bounds) we must re-check and exclude those not passing
                if need_vslow_check:
                    keep = []
                    reasons = []
                    v_units_col = []
                    v_scaled_col = []
                    for r in filtered:
                        vs_val = float(r[5]) if len(r) > 5 else np.nan
                        # Use provided phase string if present, otherwise 'NONE' so the function will still apply explicit bounds if any
                        phase_for_check = vslow_phase if (vslow_phase and vslow_phase.upper() != 'NONE') else 'NONE'
                        v_accept, v_scaled, v_units, v_reason = evaluate_vslow_acceptance(vs_val, vslow_phase=phase_for_check, vslow_tol=vslow_tol, vslow_min_override=vslow_min_override, vslow_max_override=vslow_max_override, auto_detect_units=vslow_auto_units)
                        if v_accept:
                            keep.append(r)
                            reasons.append('OK')
                        else:
                            reasons.append(v_reason)
                        v_units_col.append(v_units)
                        v_scaled_col.append(v_scaled)
                    filtered = np.array(keep) if keep else np.empty((0, src_aug.shape[1]))
                out_filtered = os.path.join(args.project_folder, f'{csv_prefix}_filtered.csv')
                # Prepare filtered CSV output. If the user requested vslow phase acceptance
                # we append columns that describe the vslow scaled value, units and QC reason.
                if need_vslow_check:
                    header_filtered = header + ',vslow_scaled_s_per_km,vslow_units,vslow_qc_reason'
                    # Convert filtered rows and the vslow metadata into strings and write via csv
                    if filtered.size > 0:
                        # Build rows as lists of strings
                        rows = []
                        # If we previously computed per-row v_units_col/v_scaled_col/reasons, they were
                        # created only for the subset; but to be safe recompute here
                        for r in filtered:
                            lat, lon, emin_ax, emaj_ax, el_ang, vslow_val = list(map(float, r[:6]))
                            ntri = int(r[6]) if r.shape[0] > 6 else -1
                            v_accept, v_scaled, v_units, v_reason = (False, np.nan, 'unknown', 'NO_CHECK')
                            try:
                                # reuse same helper defined earlier in this scope; use phase string if given
                                phase_for_check = args.vslow_accept_phase.upper() if (args.vslow_accept_phase and args.vslow_accept_phase.upper() != 'NONE') else 'NONE'
                                v_accept, v_scaled, v_units, v_reason = evaluate_vslow_acceptance(vslow_val, vslow_phase=phase_for_check, vslow_tol=vslow_tol, vslow_min_override=vslow_min_override, vslow_max_override=vslow_max_override, auto_detect_units=vslow_auto_units)
                            except Exception:
                                pass
                            rows.append([f"{lat:.6f}", f"{lon:.6f}", f"{emin_ax:.6f}", f"{emaj_ax:.6f}", f"{el_ang:.6f}", f"{vslow_val:.6f}", f"{ntri}", f"{v_scaled:.6f}" if not np.isnan(v_scaled) else '', v_units, v_reason])
                        # Write CSV
                        with open(out_filtered, 'w', newline='') as fh:
                            writer = csv.writer(fh)
                            writer.writerow(header_filtered.split(','))
                            writer.writerows(rows)
                    else:
                        # Create empty file with header
                        open(out_filtered, 'w').write(header_filtered + '\n')
                else:
                    # No vslow-phase acceptance requested; write same as before
                    if filtered.size > 0:
                        np.savetxt(out_filtered, filtered, delimiter=',', header=header, comments='', fmt='%.6f')
                    else:
                        # Create empty file with header
                        open(out_filtered, 'w').write(header + '\n')

                print(f"Wrote filtered aggregated sources CSV: {out_filtered} ({filtered.shape[0]} rows) -- filters: min_ntri={min_ntri}, max_emaj_km={max_emaj_km}, max_emaj_ratio={max_emaj_ratio}, allow_zero_vslow={args.allow_zero_vslow}")
                print('Filter summary: ' + ', '.join([f"{k}={v}" for k, v in flag_counts.items()]))
                # Optionally create a map of the filtered sources (preserve the original map too)
                if args.plot_filtered_map:
                    try:
                        import shutil
                        # Use the filtered prefix for the map if output_prefix is set
                        filtered_prefix = f'{csv_prefix}_filtered' if args.output_prefix else None
                        # Pass full filtered array (or at least 7 columns) to include nTri for color_by_ntri
                        filtered_for_plot = filtered if filtered.shape[1] >= 7 else filtered[:, :6]
                        plot_map(stations, filtered_for_plot, args.seismicflag, args.project_folder, no_stations=args.no_stations, source_dot_size=args.source_dot_size, include_zero_vslow=False, color_by_ntri=args.color_by_ntri, output_prefix=filtered_prefix)
                        print(f"Wrote filtered map")
                    except Exception as e:
                        print(f"Warning: could not plot filtered map: {e}")
            except Exception as e:
                print(f"Warning: could not write filtered aggregated CSV: {e}")
        else:
            print(f"No sources found; skipping filtered CSV for flag {args.seismicflag}.")
    else:
        print("Skipping writing filtered aggregated CSV (--no_filtered specified)")

    if args.plot_coarse:
        coarse_sources = read_coarse_locations(day_folders, args.seismicflag)
        plot_coarse_map(coarse_sources, args.seismicflag, args.project_folder)

# To use StationXML plotting, uncomment and use this instead:
# def main():
#     parser = argparse.ArgumentParser(description="Plot stations from a StationXML file as black symbols with unfilled ellipses.")
#     parser.add_argument('--stationxml', type=str, required=True, help='Path to the StationXML file.')
#     parser.add_argument('--output_png', type=str, default='stationxml_map.png', help='Output PNG filename.')
#     args = parser.parse_args()
#     plot_stationxml(args.stationxml, args.output_png)

if __name__ == "__main__":
    main()
