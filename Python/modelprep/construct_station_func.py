import numpy as np
from timerecalc import cal2jd

def BLH2XYZ(B, L, H):
    """
    Convert geodetic coordinates (latitude, longitude, height) to ECEF X,Y,Z.

    Parameters
    ----------
    B : float or array
        Latitude in radians
    L : float or array
        Longitude in radians
    H : float or array
        Height in meters

    Returns
    -------
    X, Y, Z : float or array
        ECEF coordinates in meters
    """
    a = 6378137.0
    b = 6356752.314
    eks = np.sqrt((a**2 - b**2)/a**2)

    B = np.atleast_1d(B)
    L = np.atleast_1d(L)
    H = np.atleast_1d(H)

    X = np.zeros_like(B, dtype=float)
    Y = np.zeros_like(B, dtype=float)
    Z = np.zeros_like(B, dtype=float)

    for i in range(len(B)):
        R = a / np.sqrt(1 - eks**2 * np.sin(B[i])**2)
        X[i] = (R + H[i]) * np.cos(B[i]) * np.cos(L[i])
        Y[i] = (R + H[i]) * np.cos(B[i]) * np.sin(L[i])
        Z[i] = (R * (1 - eks**2) + H[i]) * np.sin(B[i])

    return X, Y, Z

def xyzSP32elaz(XYZ, xyz):
    """
    Compute azimuth and elevation from reference XYZ to target xyz.

    Parameters
    ----------
    XYZ : array-like
        Reference ECEF coordinates [3,]
    xyz : array-like
        Target ECEF coordinates [3,]

    Returns
    -------
    el : float
        Elevation angle in radians
    az : float
        Azimuth angle from north (clockwise) in radians
    """
    from math import sin, cos, atan2, sqrt, pi

    XYZ = np.array(XYZ, dtype=float).reshape(3)
    xyz = np.array(xyz, dtype=float).reshape(3)

    # Convert reference ECEF to geodetic
    lat, lon, _ = ECEF2LLA(XYZ)

    lat = float(np.asarray(lat).squeeze())
    lon = float(np.asarray(lon).squeeze())

    cl, sl = cos(lon), sin(lon)
    cb, sb = cos(lat), sin(lat)

    F = np.array([[-sl, -sb*cl, cb*cl],
                  [cl, -sb*sl, cb*sl],
                  [0, cb, sb]])

    local_vector = F.T @ (xyz - XYZ)

    E, N, U = local_vector
    hor_dis = sqrt(E**2 + N**2)

    if hor_dis < 1e-20:
        az = 0.0
        el = pi/2
    else:
        az = atan2(E, N)
        el = atan2(U, hor_dis)

    if az < 0:
        az += 2*pi

    return el, az


def doy2jd(yr, doy):
    """
    Convert year and day-of-year to Julian date.

    Parameters
    ----------
    yr : int
        Year
    doy : int or float
        Day of year

    Returns
    -------
    jd : float
        Julian date
    """
    return cal2jd(yr, 1, 1) + doy  

def vmf1_ht(ah, aw, dmjd, dlat, ht, zd):
    """
    Compute VMF1 hydrostatic and wet mapping functions with height correction.

    Parameters
    ----------
    ah, aw : float
        Hydrostatic and wet coefficients
    dmjd : float
        Modified Julian Date
    dlat : float
        Ellipsoidal latitude [rad]
    ht : float
        Ellipsoidal height [m]
    zd : float
        Zenith distance [rad]

    Returns
    -------
    vmf1h, vmf1w : float
        Hydrostatic and wet mapping functions
    """
    import numpy as np

    pi = np.pi

    # Reference day offset
    doy = dmjd - 44239 + 1 - 28

    # Hydrostatic parameters
    bh = 0.0029
    c0h = 0.062

    if dlat < 0:  # Southern hemisphere
        phh, c11h, c10h = pi, 0.007, 0.002
    else:         # Northern hemisphere
        phh, c11h, c10h = 0.0, 0.005, 0.001

    ch = c0h + ((np.cos(doy/365.25*2*pi + phh) + 1)*c11h/2 + c10h)*(1 - np.cos(dlat))

    sine = np.sin(pi/2 - zd)

    beta  = bh / (sine + ch)
    gamma = ah / (sine + beta)
    topcon = (1 + ah / (1 + bh / (1 + ch)))
    vmf1h = topcon / (sine + gamma)

    # Height correction
    a_ht, b_ht, c_ht = 2.53e-5, 5.49e-3, 1.14e-3
    hs_km = ht / 1000
    beta  = b_ht / (sine + c_ht)
    gamma = a_ht / (sine + beta)
    topcon = (1 + a_ht / (1 + b_ht / (1 + c_ht)))
    ht_corr_coef = 1/sine - topcon / (sine + gamma)
    ht_corr = ht_corr_coef * hs_km
    vmf1h += ht_corr

    # Wet mapping function
    bw, cw = 0.00146, 0.04391
    beta  = bw / (sine + cw)
    gamma = aw / (sine + beta)
    topcon = (1 + aw / (1 + bw / (1 + cw)))
    vmf1w = topcon / (sine + gamma)

    return vmf1h, vmf1w


def ECEF2LLA(x_ecef, method=0):
    """
    Convert ECEF coordinates to geodetic latitude, longitude, and height.

    Parameters
    ----------
    x_ecef : ndarray
        ECEF coordinates (n x 3) [X, Y, Z] in meters
    method : int, optional
        0 - fast approximation (default)
        1 - iterative, more accurate

    Returns
    -------
    lla : ndarray
        Geodetic coordinates (n x 3) [lat(rad), lon(rad), height(m)]
    """
    x_ecef = np.atleast_2d(x_ecef)
    n_points = x_ecef.shape[0]

    RE = 6378137.0                # semi-major axis (m)
    f = 1/298.257223563           # flattening
    b = 6356752.314               # semi-minor axis (m)

    lla = np.zeros((n_points, 3))

    if method == 0:
        # Fast closed-form approximation
        e2 = (RE**2 - b**2) / b**2  # second numerical eccentricity squared

        p = np.sqrt(x_ecef[:,0]**2 + x_ecef[:,1]**2)
        theta = np.arctan2(RE * x_ecef[:,2], p * b)

        lla[:,0] = np.arctan2(
            x_ecef[:,2] + e2 * b * np.sin(theta)**3,
            p - e2 * RE * np.cos(theta)**3
        )
        lla[:,1] = np.arctan2(x_ecef[:,1], x_ecef[:,0])

        n = RE**2 / np.sqrt(RE**2 * np.cos(lla[:,0])**2 + b**2 * np.sin(lla[:,0])**2)
        lla[:,2] = p / np.cos(lla[:,0]) - n

    elif method == 1:
        # Iterative method
        tol = 1e-14
        max_iter = 15
        m = 1
        dlat = np.ones(n_points)

        e2 = (RE**2 - b**2) / RE**2
        p = np.sqrt(x_ecef[:,0]**2 + x_ecef[:,1]**2)
        lat = np.arctan2(x_ecef[:,2], p * (1 - e2))

        lla[:,1] = np.arctan2(x_ecef[:,1], x_ecef[:,0])

        while np.any(dlat > tol) and m <= max_iter:
            n = RE**2 / np.sqrt(RE**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2)
            lla[:,2] = p / np.cos(lat) - n

            newlat = np.arctan2(x_ecef[:,2], p * (1 - e2 * (n / (n + lla[:,2]))))
            dlat = np.abs(lat - newlat)
            lat = newlat
            lla[:,0] = lat
            m += 1

            if m > max_iter:
                print(f"Warning: Maximum iterations ({max_iter}) exceeded in ECEF2LLA")
                lla[:,0] = lat
                break
    else:
        raise ValueError("method must be 0 (fast) or 1 (iterative)")
        
   
    return lla[:,0], lla[:,1], lla[:,2]  
