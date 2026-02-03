"""
Array geometry module for BEAM.

Provides station selection and array geometry functions adapted from
AELUMA patterns, including:
- Distance matrix computation
- Station culling (too close/too far)
- Array geometry in local coordinates
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_distance_matrix(latitudes, longitudes):
    """
    Compute pairwise distance matrix between stations.
    
    Uses great-circle distance approximation.
    Adapted from AELUMA distmatrix function.
    
    Parameters
    ----------
    latitudes : array-like
        Station latitudes in degrees
    longitudes : array-like
        Station longitudes in degrees
        
    Returns
    -------
    distmat : ndarray
        N x N distance matrix in km
    distmin : ndarray
        Distance to nearest neighbor for each station
    distmin2 : ndarray
        Distance to 2nd nearest neighbor for each station
    distmin3 : ndarray
        Distance to 3rd nearest neighbor for each station
    """
    lats = np.asarray(latitudes)
    lons = np.asarray(longitudes)
    n = len(lats)
    
    distmat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(lats[i], lons[i], lats[j], lons[j])
            distmat[i, j] = dist
            distmat[j, i] = dist
    
    # Set diagonal to infinity for finding minimums
    distmat_copy = distmat.copy()
    np.fill_diagonal(distmat_copy, np.inf)
    
    # Find 1st, 2nd, 3rd nearest neighbors
    sorted_dists = np.sort(distmat_copy, axis=1)
    
    distmin = sorted_dists[:, 0]
    distmin2 = sorted_dists[:, 1] if n > 1 else np.full(n, np.inf)
    distmin3 = sorted_dists[:, 2] if n > 2 else np.full(n, np.inf)
    
    return distmat, distmin, distmin2, distmin3


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points.
    
    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in degrees
    lat2, lon2 : float
        Second point coordinates in degrees
        
    Returns
    -------
    distance : float
        Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def cull_stations(latitudes, longitudes, station_codes=None,
                  min_spacing=None, max_spacing=None):
    """
    Cull stations based on spacing criteria.
    
    Adapted from AELUMA station culling logic.
    
    Parameters
    ----------
    latitudes : array-like
        Station latitudes
    longitudes : array-like
        Station longitudes
    station_codes : list, optional
        Station codes (for logging)
    min_spacing : float, optional
        Minimum spacing between stations in km.
        Stations closer than this will be culled.
    max_spacing : float, optional
        Maximum distance to 2nd nearest neighbor in km.
        Isolated stations will be culled.
        
    Returns
    -------
    keep_indices : ndarray
        Indices of stations to keep
    culled_indices : ndarray
        Indices of culled stations
    """
    lats = np.asarray(latitudes)
    lons = np.asarray(longitudes)
    nstat = len(lats)
    
    if station_codes is None:
        station_codes = [f"STA{i}" for i in range(nstat)]
    
    keep_mask = np.ones(nstat, dtype=bool)
    
    # Cull stations too close together
    if min_spacing is not None:
        distmat, distmin, distmin2, distmin3 = compute_distance_matrix(lats, lons)
        
        # Use average of distmin and distmin2 to decide which to remove
        mindist = np.min((distmin + distmin2) / 2)
        
        while mindist < min_spacing and np.sum(keep_mask) > 3:
            current_lats = lats[keep_mask]
            current_lons = lons[keep_mask]
            current_indices = np.where(keep_mask)[0]
            
            distmat, distmin, distmin2, _ = compute_distance_matrix(current_lats, current_lons)
            avg_dist = (distmin + distmin2) / 2
            
            # Remove station with smallest average distance to neighbors
            remove_local = np.argmin(avg_dist)
            remove_global = current_indices[remove_local]
            
            keep_mask[remove_global] = False
            logger.debug(f"Culled station {station_codes[remove_global]} "
                        f"(too close, avg dist = {avg_dist[remove_local]:.2f} km)")
            
            # Recompute
            current_lats = lats[keep_mask]
            current_lons = lons[keep_mask]
            
            if len(current_lats) < 3:
                break
            
            distmat, distmin, distmin2, _ = compute_distance_matrix(current_lats, current_lons)
            mindist = np.min((distmin + distmin2) / 2)
    
    # Cull isolated stations (too far from others)
    if max_spacing is not None and np.sum(keep_mask) > 3:
        nstat_prev = 0
        nstat_curr = np.sum(keep_mask)
        
        while nstat_prev != nstat_curr:
            nstat_prev = nstat_curr
            
            current_lats = lats[keep_mask]
            current_lons = lons[keep_mask]
            current_indices = np.where(keep_mask)[0]
            
            if len(current_lats) <= 3:
                break
            
            _, _, distmin2, _ = compute_distance_matrix(current_lats, current_lons)
            
            # Find stations with 2nd nearest neighbor too far
            isolated = distmin2 >= max_spacing
            
            if np.any(isolated):
                for local_idx in np.where(isolated)[0]:
                    global_idx = current_indices[local_idx]
                    keep_mask[global_idx] = False
                    logger.debug(f"Culled station {station_codes[global_idx]} "
                                f"(isolated, 2nd nearest = {distmin2[local_idx]:.2f} km)")
            
            nstat_curr = np.sum(keep_mask)
    
    keep_indices = np.where(keep_mask)[0]
    culled_indices = np.where(~keep_mask)[0]
    
    logger.info(f"Station culling: {nstat} -> {len(keep_indices)} "
               f"(culled {len(culled_indices)})")
    
    return keep_indices, culled_indices


def compute_array_geometry(latitudes, longitudes, elevations=None):
    """
    Compute array geometry in local Cartesian coordinates.
    
    Coordinates are computed relative to array center.
    
    Parameters
    ----------
    latitudes : array-like
        Station latitudes in degrees
    longitudes : array-like  
        Station longitudes in degrees
    elevations : array-like, optional
        Station elevations in meters
        
    Returns
    -------
    geometry : dict
        Dictionary containing:
        - 'x': East-West positions in km
        - 'y': North-South positions in km
        - 'z': Elevations in km (if provided)
        - 'center_lat': Array center latitude
        - 'center_lon': Array center longitude
    """
    lats = np.asarray(latitudes)
    lons = np.asarray(longitudes)
    
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Convert to local coordinates (km)
    # Approximate conversion factors
    km_per_deg_lat = 110.54  # km per degree latitude
    km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))  # varies with latitude
    
    x = (lons - center_lon) * km_per_deg_lon  # East-West
    y = (lats - center_lat) * km_per_deg_lat  # North-South
    
    geometry = {
        'x': x,
        'y': y,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'latitudes': lats,
        'longitudes': lons
    }
    
    if elevations is not None:
        geometry['z'] = np.asarray(elevations) / 1000.0  # Convert to km
    
    return geometry


def compute_array_response(geometry, slowness_x, slowness_y, frequency):
    """
    Compute array response (beam pattern) for given slowness.
    
    Parameters
    ----------
    geometry : dict
        Array geometry from compute_array_geometry()
    slowness_x : float or ndarray
        East-West slowness in s/km
    slowness_y : float or ndarray  
        North-South slowness in s/km
    frequency : float
        Frequency in Hz
        
    Returns
    -------
    response : float or ndarray
        Array response (normalized power)
    """
    x = geometry['x']
    y = geometry['y']
    n = len(x)
    
    # Compute phase delays
    omega = 2 * np.pi * frequency
    
    if np.isscalar(slowness_x):
        phase = omega * (slowness_x * x + slowness_y * y)
        beam = np.sum(np.exp(1j * phase))
        response = np.abs(beam)**2 / n**2
    else:
        # Grid of slownesses
        sx = np.asarray(slowness_x)
        sy = np.asarray(slowness_y)
        
        response = np.zeros((len(sy), len(sx)))
        for i, s_y in enumerate(sy):
            for j, s_x in enumerate(sx):
                phase = omega * (s_x * x + s_y * y)
                beam = np.sum(np.exp(1j * phase))
                response[i, j] = np.abs(beam)**2 / n**2
    
    return response


def slowness_to_velocity_azimuth(slowness_x, slowness_y):
    """
    Convert slowness components to velocity and back-azimuth.
    
    Parameters
    ----------
    slowness_x : float
        East-West slowness in s/km
    slowness_y : float
        North-South slowness in s/km
        
    Returns
    -------
    velocity : float
        Apparent velocity in km/s
    backazimuth : float
        Back-azimuth in degrees (0-360, measured clockwise from North)
    """
    slowness = np.sqrt(slowness_x**2 + slowness_y**2)
    
    if slowness == 0:
        return np.inf, 0.0
    
    velocity = 1.0 / slowness
    
    # Back-azimuth: direction wave is coming FROM
    # atan2(x, y) gives angle from North, measured clockwise
    backazimuth = np.degrees(np.arctan2(slowness_x, slowness_y))
    
    # Ensure 0-360 range
    if backazimuth < 0:
        backazimuth += 360
    
    return velocity, backazimuth


def velocity_azimuth_to_slowness(velocity, backazimuth):
    """
    Convert velocity and back-azimuth to slowness components.
    
    Parameters
    ----------
    velocity : float
        Apparent velocity in km/s
    backazimuth : float
        Back-azimuth in degrees
        
    Returns
    -------
    slowness_x : float
        East-West slowness in s/km
    slowness_y : float
        North-South slowness in s/km
    """
    slowness = 1.0 / velocity
    backazimuth_rad = np.radians(backazimuth)
    
    slowness_x = slowness * np.sin(backazimuth_rad)
    slowness_y = slowness * np.cos(backazimuth_rad)
    
    return slowness_x, slowness_y


def partition_into_subarrays(station_codes, latitudes, longitudes, group_size=7):
    """
    Partition stations into subarrays by greedy nearest-neighbour grouping.

    Parameters
    ----------
    station_codes : list
        List of station codes, same order as latitudes/longitudes
    latitudes : array-like
        Latitudes corresponding to station_codes
    longitudes : array-like
        Longitudes corresponding to station_codes
    group_size : int
        Desired number of stations per subarray

    Returns
    -------
    subarrays : list of list
        List of subarray station code lists. Last group may be smaller.
    """
    codes = list(station_codes)
    lats = np.asarray(latitudes)
    lons = np.asarray(longitudes)
    n = len(codes)

    if n == 0:
        return []

    # Compute distance matrix once
    distmat, _, _, _ = compute_distance_matrix(lats, lons)

    remaining = set(range(n))
    subarrays = []

    while remaining:
        # pick the remaining station with smallest average distance to others
        rem_list = np.array(sorted(list(remaining)))
        avg_dist = np.mean(distmat[np.ix_(rem_list, rem_list)], axis=1)
        seed_idx = rem_list[int(np.argmin(avg_dist))]

        # sort remaining by distance to seed
        rem_sorted = sorted(list(remaining), key=lambda idx: distmat[seed_idx, idx])

        # select up to group_size stations
        selected = rem_sorted[:group_size]
        subarrays.append([codes[i] for i in selected])

        # remove selected from remaining
        for i in selected:
            if i in remaining:
                remaining.remove(i)

    return subarrays
