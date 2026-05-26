import numpy as np
from scipy.interpolate import RegularGridInterpolator

def pBLh2ZHDRT(model):
    """
    Generate Zenith Hydrostatic Delay (ZHD) values based on a priori model.

    Parameters
    ----------
    model : object
        Object containing tomography and ray tracing model parameters.
        Required attributes:
            - BLh : array of station positions [n x 4] (lat, lon, height, ?)
            - NAME : array of station names
            - lat_TOMO : 1D array of latitude grid
            - lon_TOMO : 1D array of longitude grid
            - levels_TOMO : 1D array of vertical levels
            - pres3D : 3D array of pressure values corresponding to the grid

    Returns
    -------
    ZHD : ndarray
        Zenith Hydrostatic Delay values for each station
    """
    # Extract RT station info
    RTstat = model['BLh']
    RTName = model['NAME']
    common, ia, ib = np.intersect1d(RTName, model['NAME'], return_indices=True)
    
    BLh = RTstat[ia, :]
    RTstat = RTstat[ia, 1:4]  # keep only columns 2-4 for interpolation

    # Setup 3D interpolator
    interpolator = RegularGridInterpolator(
        (model['lat_TOMO'], model['lon_TOMO'], model['levels_TOMO']),
        model['pres3D'],
        method='linear',  
        bounds_error=False,
        fill_value=None
    )
    
    interpolator2 = RegularGridInterpolator(
        (model['lat_TOMO'], model['lon_TOMO'], model['levels_TOMO']),
        model['temp3D'],
        method='linear',  
        bounds_error=False,
        fill_value=None
    )

    # Interpolate pressure at each RT station
    p = np.array([interpolator(llRay) for llRay in RTstat])
    t = np.array([interpolator2(llRay) for llRay in RTstat])
    # Zenith Hydrostatic Delay formula
    lat_rad = np.deg2rad(BLh[:,1])[:, np.newaxis]  # convert to radians
    h_m = BLh[:,3][:, np.newaxis]  # station height

    ZHD = (0.0022767 * p) / (1 - 0.00266 * np.cos(2*lat_rad) - 0.00000028 * h_m)

    return ZHD, p, t