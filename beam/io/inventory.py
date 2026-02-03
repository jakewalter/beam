"""
Station inventory and metadata handling for BEAM.

Provides functionality to:
- Load StationXML inventory files
- Extract station coordinates
- Match coordinates to station codes
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

from obspy import read_inventory, Inventory

logger = logging.getLogger(__name__)


def load_inventory(folder, pattern='*.xml', tag=None, name=None):
    """
    Load station inventory from XML files in a folder.
    
    Parameters
    ----------
    folder : str
        Path to folder containing StationXML files
    pattern : str
        Glob pattern for inventory files
        
    Returns
    -------
    inventory : obspy.Inventory
        Combined inventory from all matching files
    """
    inv = Inventory()
    
    xml_files = glob.glob(os.path.join(folder, pattern))
    # If an exact filename is requested, find only that file
    if name is not None:
        # Accept full path or basename
        candidate = os.path.join(folder, name) if not os.path.isabs(name) else name
        if os.path.exists(candidate):
            xml_files = [candidate]
        else:
            # If candidate not found, try to match basename among available files
            xml_files = [f for f in xml_files if os.path.basename(f) == name]
    # if tag provided, filter candidate xml files by tag in filename
    if tag is not None:
        xml_files = [f for f in xml_files if tag in os.path.basename(f)]
    
    # Also check for dailyinventory.xml
    daily_inv = os.path.join(folder, 'dailyinventory.xml')
    if os.path.exists(daily_inv) and daily_inv not in xml_files:
        xml_files.append(daily_inv)
    
    for xml_file in sorted(xml_files):
        # Quick XML root tag check to avoid attempting to parse non-StationXML files
        try:
            import xml.etree.ElementTree as ET
            it = ET.iterparse(xml_file, events=("start",))
            event, elem = next(it)
            root_tag = elem.tag
            # StationXML root will include "StationXML" substring; skip others
            if 'StationXML' not in root_tag and 'Inventory' not in root_tag:
                logger.debug(f"Skipping non-StationXML file: {xml_file} (root:{root_tag})")
                continue
        except Exception:
            # If we can't parse the file quickly, skip it to avoid noisy logs
            logger.debug(f"Skipping file due to parse check fail: {xml_file}")
            continue
        try:
            inv_part = read_inventory(xml_file)
            inv.networks.extend(inv_part.networks)
            logger.debug(f"Loaded inventory from {xml_file}")
        except Exception as e:
            logger.warning(f"Failed to read inventory {xml_file}: {e}")
    
    logger.info(f"Loaded inventory with {sum(len(net) for net in inv)} stations")
    return inv


def get_station_coordinates(inventory):
    """
    Extract station coordinates from inventory.
    
    Parameters
    ----------
    inventory : obspy.Inventory
        Station inventory
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: code, latitude, longitude, elevation
    """
    stations = []
    
    for net in inventory:
        for sta in net:
            stations.append({
                'network': net.code,
                'code': sta.code,
                'latitude': sta.latitude,
                'longitude': sta.longitude,
                'elevation': sta.elevation
            })
    
    df = pd.DataFrame(stations)
    return df


def get_station_coords_dict(inventory):
    """
    Get station coordinates as a dictionary.
    
    Parameters
    ----------
    inventory : obspy.Inventory
        Station inventory
        
    Returns
    -------
    coords : dict
        Dictionary with station codes as keys and (lat, lon, elev) tuples as values
    """
    coords = {}
    
    for net in inventory:
        for sta in net:
            coords[sta.code] = (sta.latitude, sta.longitude, sta.elevation)
    
    return coords


def match_stations_to_stream(stream, inventory):
    """
    Match stream traces to inventory stations.
    
    Parameters
    ----------
    stream : obspy.Stream
        Stream of waveforms
    inventory : obspy.Inventory
        Station inventory
        
    Returns
    -------
    matched : dict
        Dictionary mapping station codes to (trace, coordinates) tuples
    unmatched : list
        List of station codes in stream but not in inventory
    """
    coords = get_station_coords_dict(inventory)
    
    matched = {}
    unmatched = []
    
    for tr in stream:
        sta = tr.stats.station
        if sta in coords:
            matched[sta] = {
                'trace': tr,
                'latitude': coords[sta][0],
                'longitude': coords[sta][1],
                'elevation': coords[sta][2]
            }
        else:
            unmatched.append(sta)
    
    if unmatched:
        logger.warning(f"Stations not in inventory: {unmatched}")
    
    return matched, unmatched


def build_coord_lookup(folder):
    """
    Build a coordinate lookup from StationXML files.
    
    Creates a dictionary mapping (lat, lon) tuples to station codes,
    useful for reverse lookup when only coordinates are available.
    
    Parameters
    ----------
    folder : str
        Path to folder containing StationXML files
        
    Returns
    -------
    lookup : dict
        Dictionary mapping (round(lat, 5), round(lon, 5)) -> station_code
    """
    lookup = {}
    
    xml_files = glob.glob(os.path.join(folder, '*.xml'))
    
    for xml_file in sorted(xml_files):
        try:
            inv = read_inventory(xml_file)
            for net in inv:
                for sta in net:
                    key = (round(sta.latitude, 5), round(sta.longitude, 5))
                    lookup[key] = sta.code
        except Exception:
            continue
    
    return lookup


def download_inventory(client, network, station='*', location='*', 
                       channel='*', starttime=None, endtime=None,
                       level='station'):
    """
    Download station inventory from FDSN web service.
    
    Parameters
    ----------
    client : obspy.clients.fdsn.Client
        FDSN client instance
    network : str
        Network code(s)
    station : str
        Station code(s)
    location : str
        Location code(s)
    channel : str
        Channel code(s)
    starttime : UTCDateTime, optional
        Start time for station operation
    endtime : UTCDateTime, optional
        End time for station operation
    level : str
        Level of detail ('network', 'station', 'channel', 'response')
        
    Returns
    -------
    inventory : obspy.Inventory
        Downloaded inventory
    """
    try:
        inventory = client.get_stations(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            level=level
        )
        logger.info(f"Downloaded inventory with {sum(len(net) for net in inventory)} stations")
        return inventory
    except Exception as e:
        logger.error(f"Failed to download inventory: {e}")
        return Inventory()


def save_inventory(inventory, filepath):
    """
    Save inventory to StationXML file.
    
    Parameters
    ----------
    inventory : obspy.Inventory
        Inventory to save
    filepath : str
        Output file path
    """
    inventory.write(filepath, format='STATIONXML')
    logger.info(f"Saved inventory to {filepath}")
