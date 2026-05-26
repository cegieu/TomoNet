import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def boundingTOMO(east_limit, west_limit, north_limit, south_limit, Xsta, BLh_ori, NAME):
    """
    Function to cut GNSS stations outside the bounding model.

    Parameters
    ----------
    east_limit : float
        Eastern boundary longitude (degrees)
    west_limit : float
        Western boundary longitude (degrees)
    north_limit : float
        Northern boundary latitude (degrees)
    south_limit : float
        Southern boundary latitude (degrees)
    Xsta : np.ndarray
        (3, N) array with XYZ coordinates of GNSS stations
    BLh_ori : np.ndarray
        (N, ≥6) array with GNSS station info:
        col 2: latitude (deg), col 3: longitude (deg),
        col 4: H, col 5: h
    NAME : list[str] | np.ndarray
        Names of GNSS stations

    Returns
    -------
    X, Y, Z : np.ndarray
        Cartesian coordinates of filtered GNSS stations
    lat, lon : np.ndarray
        Latitude and longitude in radians
    h, H : np.ndarray
        Heights
    NAME : list[str]
        Filtered names of GNSS stations
    """

    # Extract station data
    X = Xsta[0, :].copy()
    Y = Xsta[1, :].copy()
    Z = Xsta[2, :].copy()
    lat = np.radians(BLh_ori[:, 1])
    lon = np.radians(BLh_ori[:, 2])
    H = BLh_ori[:, 3].copy()
    h = BLh_ori[:, 4].copy()

    # Indices outside bounding box
    mask_outside = (
        (np.degrees(lon) > east_limit) |
        (np.degrees(lon) < west_limit) |
        (np.degrees(lat) > north_limit) |
        (np.degrees(lat) < south_limit)
    )

    # Keep only inside stations
    X = X[~mask_outside]
    Y = Y[~mask_outside]
    Z = Z[~mask_outside]
    lat = lat[~mask_outside]
    lon = lon[~mask_outside]
    h = h[~mask_outside]
    H = H[~mask_outside]

    if isinstance(NAME, np.ndarray):
        NAME = NAME[~mask_outside].tolist()
    else:
        NAME = [n for n, keep in zip(NAME, ~mask_outside) if keep]

    return X, Y, Z, lat, lon, h, H, NAME

def removeStat(X, Y, Z, lat, lon, h, H, NAME, stat_range):
    """
    Filter out GNSS stations to ensure a minimum spatial density.
    
    Parameters
    ----------
    X, Y, Z : array-like
        Cartesian coordinates of GNSS stations.
    lat, lon : array-like
        Latitude and longitude of GNSS stations.
    h, H : array-like
        Altitudes of GNSS stations.
    NAME : list or array-like
        Names of GNSS stations.
    stat_range : float
        Minimum distance (in same units as X,Y,Z) to keep stations.
        
    Returns
    -------
    X, Y, Z, lat, lon, h, H, NAME : arrays/lists
        Filtered station data.
    """
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    h = np.asarray(h)
    H = np.asarray(H)
    NAME = np.array(NAME)
    
    n_stations = len(X)
    keep_station = np.ones(n_stations, dtype=bool)
    
    coords = np.column_stack((X, Y, Z))
    distances = squareform(pdist(coords))
    
    for i in range(n_stations):
        if keep_station[i]:
            close_stations = (distances[i, :] < stat_range)
            close_stations[i] = False  # exclude self
            keep_station[close_stations] = False
    
    X = X[keep_station]
    Y = Y[keep_station]
    Z = Z[keep_station]
    lat = lat[keep_station]
    lon = lon[keep_station]
    h = h[keep_station]
    H = H[keep_station]
    NAME = NAME[keep_station]
    
    return X, Y, Z, lat, lon, h, H, NAME


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h: float, a: float, f: float) -> np.ndarray:
    """
    Convert geodetic coordinates to ECEF coordinates.

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude in degrees.
    lon_deg : float
        Geodetic longitude in degrees.
    h : float
        Height above the ellipsoid in meters.
    a : float
        Semi-major axis of the ellipsoid (equatorial radius) in meters.
    f : float
        Flattening of the ellipsoid (f = (a - b) / a).

    Returns
    -------
    np.ndarray
        ECEF coordinates [X, Y, Z] in meters.
    """
    # Convert degrees to radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Compute semi-minor axis
    b = a * (1 - f)

    # First eccentricity squared
    e2 = (a**2 - b**2) / a**2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    # ECEF coordinates
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + h) * np.sin(lat)

    return np.array([X, Y, Z])


def read_stat(filename: str):
    """
    Function to read BLH with formatted format

    Parameters
    ----------
    filename : str
        Path to the .txt file

    Returns
    -------
    NAME : list
        Names of GNSS stations
    BLh : np.ndarray
        Coordinates of GNSS stations (num, lat, lon, h, H)
    """
    # Define the column names and types
    col_names = ['Name', 'num', 'lat', 'lon', 'h', 'H']
    
    # Read the file using pandas
    df = pd.read_csv(filename, 
                     sep='\t', 
                     names=col_names, 
                     usecols=range(6),  # Ignore extra columns if any
                     dtype={'Name': str, 'num': float, 'lat': float, 'lon': float, 'h': float, 'H': float},
                     engine='python')

    # Extract BLh as numpy array and NAME as list
    BLh = df[['num', 'lat', 'lon', 'h', 'H']].to_numpy()
    NAME = df['Name'].tolist()

    return NAME, BLh
