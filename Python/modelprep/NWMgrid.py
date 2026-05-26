import numpy as np
import warnings
from scipy.io import savemat
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta

def struc(model, latT, lonT, levT, switches):
    """
    Generate 2D coordinates of tomography model nodes for ray tracing.
    
    Parameters
    ----------
    model : dict or object
        Model structure (kept for interface consistency, not used here).
    latT : array-like
        Latitudes of tomography model voxel centers.
    lonT : array-like
        Longitudes of tomography model voxel centers.
    levT : array-like
        Altitudes of tomography model voxel centers.
    switches : dict or object
        Switches structure (kept for interface consistency, not used here).
    
    Returns
    -------
    BLHstruc : np.ndarray
        Array of shape (3, n_nodes, n_levels), containing [lat; lon; height].
    """
    B_T = latT
    L_T = lonT
    h_T_o = levT

    L_T_o = np.tile(L_T, (1, latT.shape[0]))
    B_T_o = np.tile(B_T, (lonT.shape[0],1))
    
    # Reshape, transpose, and replicate
    B_T_o = B_T_o.reshape(-1, 1)
    B_T_o = np.sort(B_T_o,axis=0).T
    B_T_o = np.tile(B_T_o[:, :, np.newaxis], (1, 1, levT.shape[0]))
    
    L_T_o = np.tile(L_T_o[:, :, np.newaxis], (1, 1, levT.shape[0]))
    h_T_o = np.tile(h_T_o, (B_T_o.shape[1], 1))
    
    h_T_new = np.zeros((1, h_T_o.shape[0], h_T_o.shape[1]))
    for w in range(levT.shape[0]):
        h_T_new[0, :, w] = h_T_o[:, w]

    # Replace h_T_o with h_T_new
    h_T_o = h_T_new
    
    # Final stacked structure
    BLh_pudel_rad = np.vstack([B_T_o, L_T_o, h_T_o])
    BLHstruc = BLh_pudel_rad

    return BLHstruc


def load_geoid_model(filepath: str):
    lats = np.linspace(-90, 90, 181)   
    lons = np.linspace(-180, 180, 361) 
    geoid_grid = np.zeros((len(lats), len(lons))) 

    return lats, lons, geoid_grid


class GeoidModel:
    def __init__(self, filepath: str):
        lats, lons, geoid_grid = load_geoid_model(filepath)
        # Build interpolator (lat, lon ordering!)
        self.interpolator = RegularGridInterpolator(
            (lats, lons), geoid_grid, bounds_error=False, fill_value=np.nan
        )

    def height(self, lat: float, lon: float) -> float:
        return float(self.interpolator((lat, lon)))


# Instantiate once (example, replace with your real EGM2008 grid file)
egm2008_model = GeoidModel("egm2008_file_here")

def geoidheight2(lat: float, lon: float, model: str = "egm2008") -> float:
    """
    Compute geoid height using interpolation from EGM2008 grid.
    """
    if model.lower() == "egm2008":
        return egm2008_model.height(lat, lon)
    else:
        raise ValueError(f"Unsupported model: {model}")

def undu(lat1, lat2, lon1, lon2, n, path, name):
    """
    Generate undulation grid for 3DRT software.
    
    Parameters
    ----------
    lat1, lat2 : float
        Minimum and maximum latitude of area boundary points [degrees].
    lon1, lon2 : float
        Minimum and maximum longitude of area boundary points [degrees].
    n : float
        Grid resolution [degrees].
    path : str
        Path to save .mat file.
    name : str
        Filename (without extension) for saved .mat file.
    
    Returns
    -------
    unduera : np.ndarray
        Grid of geoid undulations.
    """
    dlat = abs(lat1 - lat2)
    dlon = abs(lon1 - lon2)

    # Determine number of grid points
    n_lat = int(dlat / n) + 1
    n_lon = int(dlon / n) + 1

    unduera = np.zeros((n_lon, n_lat))

    for i in range(n_lat):
        for j in range(n_lon):
            lat = max(lat1, lat2) - i * n
            lon = min(lon1, lon2) + j * n
            unduera[j, i] = geoidheight2(lat, lon, 'egm2008')

    # Save as MATLAB .mat file
    savemat(f"{path}/{name}.mat", {'unduera': unduera})

    return unduera


def fera5Epoch(eGps: np.ndarray):
    """
    Convert orbit epochs from SP3 file to ERA5-compatible NWP epochs.

    Parameters
    ----------
    eGps : np.ndarray
        Shape (n, 6). Each row: [YYYY, MM, DD, hh, mm, ss]

    Returns
    -------
    eNwp : np.ndarray
        Shape (n, 5). Each row: [YYYY, MM, DD, ERA5 epoch (0-23), layer index]
    erafn : list of str
        ERA5 filenames for each epoch
    """
    eGps = np.asarray(eGps)
    n = eGps.shape[0]
    
    # Decimal hours
    hours = eGps[:, 3] + eGps[:, 4] / 60 + eGps[:, 5] / 3600
    hours_round = np.round(hours).astype(int)
    
    # Initialize eNwp
    eNwp = np.zeros((n, 5), dtype=int)
    eNwp[:, :3] = eGps[:, :3]  # YYYY, MM, DD
    eNwp[:, 3] = hours_round   # ERA5 epoch

    mask_24 = hours_round == 24
    if np.any(mask_24):
        for idx in np.where(mask_24)[0]:
            # Wrap hour to 1
            eNwp[idx, 3] = 1
            # Add one day
            dt = datetime(int(eGps[idx, 0]), int(eGps[idx, 1]), int(eGps[idx, 2])) + timedelta(days=1)
            eNwp[idx, :3] = [dt.year, dt.month, dt.day]

    # Layer index = ERA5 epoch + 1
    eNwp[:, 4] = eNwp[:, 3] + 1

    # Generate ERA5 filenames
    erafn = [
        f"ERA5_{int(row[0])}-{int(row[1]):02d}-{int(row[2]):02d}.nc"
        for row in eNwp
    ]

    return eNwp, erafn


def fgeom2geop(lat: np.ndarray, hgeom: np.ndarray) -> np.ndarray:
    """
    Convert geometric heights (km) to geopotential heights (km) using
    Somigliana's equation for normal gravity on the WGS84 ellipsoid.

    Parameters
    ----------
    lat : np.ndarray
        Geocentric latitude in radians (can be scalar or array)
    hgeom : np.ndarray
        Geometric height in km (same shape as lat)

    Returns
    -------
    hgeop : np.ndarray
        Geopotential height in km (same shape as input)
    """

    # WGS84 ellipsoid constants
    a = 6378.137          # semi-major axis [km]
    b = 6356.7523142      # semi-minor axis [km]
    e2 = (a**2 - b**2) / a**2  # eccentricity squared
    g0 = 9.80665          # WMO gravity [m/s^2]
    ga = 9.7803253359     # equatorial gravity [m/s^2]
    gr = 0.003449787      # gravity ratio

    # Normal gravity on the ellipsoid (Somigliana 1929)
    sin_lat2 = np.sin(lat)**2
    g = ga * (1 + 1.9e-3 * sin_lat2) / np.sqrt(1 - e2 * sin_lat2)

    # Effective radius
    Reff = a / (1 + e2/2 + gr - e2 * sin_lat2)

    # Geopotential height
    hgeop = (g / g0) * (Reff * hgeom / (Reff + hgeom))

    return hgeop


def fgeop2geom(lat: np.ndarray, hgeop: np.ndarray) -> np.ndarray:
    """
    Convert geopotential heights (km) to geometric heights (km) using
    Somigliana's equation for normal gravity on the WGS84 ellipsoid.

    Parameters
    ----------
    lat : np.ndarray
        Geocentric latitude in radians (can be scalar or array)
    hgeop : np.ndarray
        Geopotential height in km (same shape as lat)

    Returns
    -------
    hgeom : np.ndarray
        Geometric height in km (same shape as input)
    """

    # WGS84 ellipsoid constants
    a = 6378.137          # semi-major axis [km]
    b = 6356.7523142      # semi-minor axis [km]
    e2 = (a**2 - b**2) / a**2  # eccentricity squared
    g0 = 9.80665          # WMO gravity [m/s^2]
    ga = 9.7803253359     # equatorial gravity [m/s^2]
    gr = 0.003449787      # gravity ratio

    # Normal gravity on the ellipsoid (Somigliana 1929)
    sin_lat2 = np.sin(lat)**2
    g = ga * (1 + 1.9e-3 * sin_lat2) / np.sqrt(1 - e2 * sin_lat2)

    # Effective radius
    Reff = a / (1 + e2/2 + gr - e2 * sin_lat2)

    # Geometric height
    hgeom = Reff * hgeop / (g / g0 * Reff - hgeop)

    return hgeom


def stdatmo(H_in, Toffset=0, Units='SI', GeomFlag=False):
    """
    STDATMO Find gas properties in Earth's atmosphere.
    
    Parameters
    ----------
    H_in : array_like
        Altitude input (scalar, vector, or ND array). SI: meters, US: feet.
    Toffset : array_like, optional
        Temperature offset from standard (K for SI, R for US). Default: 0.
    Units : str or tuple of str, optional
        Units specification: 'SI' (default) or 'US', or ('SI', 'US').
    GeomFlag : bool, optional
        If True, H_in is geometric altitude, else geopotential. Default: False.
    
    Returns
    -------
    rho : ndarray
        Density [kg/m³ SI, slug/ft³ US]
    a : ndarray
        Speed of sound [m/s SI, ft/s US]
    temp : ndarray
        Temperature [K SI, R US]
    press : ndarray
        Pressure [Pa SI, lbf/ft² US]
    kvisc : ndarray
        Kinematic viscosity [m²/s SI, ft²/s US]
    ZorH : ndarray
        Altitude [m SI, ft US], geometric or geopotential
    """

    H_in = np.asarray(H_in, dtype=float)
    Toffset = np.asarray(Toffset, dtype=float)

    # Units handling
    if isinstance(Units, (list, tuple)) and len(Units) == 2:
        Unitsin, Unitsout = Units
    else:
        Unitsin = Unitsout = Units

    Uin = Unitsin.lower() == 'us'
    Uout = Unitsout.lower() == 'us'

    if Uin:
        H_in = H_in * 0.3048      # ft -> m
        Toffset = Toffset * 5/9   # °F/°R -> K

    # Convert geometric altitude to geopotential if necessary
    RE = 6356766.0  # Earth radius [m]
    if GeomFlag:
        Hgeop = (RE * H_in) / (RE + H_in)
    else:
        Hgeop = H_in.copy()

    # Atmospheric layer table: [index, lapse_rate, base_temp, base_H, base_P]
    D = np.array([
        [1,   -0.0065, 288.15,          0,          101325],
        [2,    0.0,    216.65,      11000,       22632.04],
        [3,    0.001,  216.65,      20000,        5474.88],
        [4,    0.0028, 228.65,      32000,         868.02],
        [5,    0.0,    270.65,      47000,         110.91],
        [6,   -0.0028, 270.65,      51000,          66.94],
        [7,   -0.002,  214.65,      71000,           3.96],
        [8,    0.0,    186.945908,  84852.046,      0.3734]
    ])

    K = D[:,1]      # lapse rate [K/m]
    T0 = D[:,2]     # base temp [K]
    H0 = D[:,3]     # base geopotential height [m]
    P0 = D[:,4]     # base pressure [Pa]

    # Constants
    R = 287.05287       # J/kg/K
    gamma = 1.4
    g0 = 9.80665
    Bs = 1.458e-6
    S = 110.4
    hmax = 90000.0

    temp = np.zeros_like(H_in)
    press = np.zeros_like(H_in)

    # Layer indices
    n1 = (Hgeop <= H0[1])
    n2 = (Hgeop <= H0[2]) & (Hgeop > H0[1])
    n3 = (Hgeop <= H0[3]) & (Hgeop > H0[2])
    n4 = (Hgeop <= H0[4]) & (Hgeop > H0[3])
    n5 = (Hgeop <= H0[5]) & (Hgeop > H0[4])
    n6 = (Hgeop <= H0[6]) & (Hgeop > H0[5])
    n7 = (Hgeop <= H0[7]) & (Hgeop > H0[6])
    n8 = (Hgeop <= hmax) & (Hgeop > H0[7])
    n9 = (Hgeop > hmax)

    def layer_calc(mask, i):
        if not np.any(mask):
            return
        if K[i] == 0:
            temp[mask] = T0[i]
            press[mask] = P0[i] * np.exp(-g0*(Hgeop[mask]-H0[i])/(R*T0[i]))
        else:
            TonTi = 1 + K[i]*(Hgeop[mask]-H0[i])/T0[i]
            temp[mask] = TonTi*T0[i]
            press[mask] = P0[i] * TonTi ** (-g0/(K[i]*R))

    for mask, idx in zip([n1,n2,n3,n4,n5,n6,n7,n8], range(8)):
        layer_calc(mask, idx)

    if np.any(n9):
        warnings.warn("One or more altitudes above upper limit.")
        temp[n9] = np.nan
        press[n9] = np.nan

    temp += Toffset
    rho = press / (R * temp)
    a = np.sqrt(gamma * R * temp)
    kvisc = (Bs * temp**1.5 / (temp + S)) / rho

    # Compute geometric altitude if necessary
    if GeomFlag:
        ZorH = Hgeop
    else:
        ZorH = (RE * Hgeop) / (RE - Hgeop)

    # Convert to US units if requested
    if Uout:
        rho /= 515.3788
        a /= 0.3048
        temp *= 1.8
        press /= 47.88026
        kvisc /= 0.09290304
        ZorH /= 0.3048

    return rho, a, temp, press, kvisc, ZorH