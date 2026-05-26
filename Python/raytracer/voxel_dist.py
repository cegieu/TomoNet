import numpy as np
import find_num_v as fv

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

def sind(x):
    return np.sin(np.asarray(x) * deg2rad)

def cosd(x):
    return np.cos(np.asarray(x) * deg2rad)

def tand(x):
    return np.tan(np.asarray(x) * deg2rad)

def asind_deg(x):
    return np.arcsin(np.asarray(x)) * rad2deg

def atan2_deg(y, x):
    return np.arctan2(y, x) * rad2deg

def acotd(x):
    return np.arctan(1 / np.asarray(x)) * rad2deg


def cot(x):
    return 1.0 / np.tan(x)

def voxel_dist(phi_N, lam_N, h_N, phi_st, lam_st, h_st, ele, azi):
    """
    Compute straight ray paths within each voxel.
    Parameters
    ----------
    phi_N : np.ndarray
        Latitude of voxel centers [deg]
    lam_N : np.ndarray
        Longitude of voxel centers [deg]
    h_N : np.ndarray
        Height of voxel centers [m]
    phi_st, lam_st : float
        Station latitude/longitude [deg]
    h_st : float
        Station height [m]
    ele : float
        Elevation angle [deg]
    azi : float
        Azimuth angle [deg]
    
    Returns
    -------
    ray : dict
        ray['d_voxel'] : list of path lengths in each voxel [m]
        ray['n_voxel'] : list of [lat_idx, lon_idx] for each voxel
    """
    deg2rad = np.pi / 180
    rad2deg = 1 / deg2rad

    # unique lat/lon
    phi_list = np.unique(phi_N)
    lam_list = np.unique(lam_N)


    if phi_list.size < 3 or lam_list.size < 3:
        raise ValueError("phi_list and lam_list must contain at least 3 unique values each.")
    
    delta_phi = (phi_list[2] - phi_list[1]) / 2.0
    delta_lam = (lam_list[2] - lam_list[1]) / 2.0
    
    # build phi_v and lam_v (inner model boundaries)
    phi_v_inner = []
    for i in range(1, len(phi_list) - 1):   
        phi_v_inner.append(phi_list[i] - delta_phi)
        phi_v_inner.append(phi_list[i] + delta_phi)
    phi_v_inner = np.asarray(phi_v_inner)
    
    lam_v_inner = []
    for i in range(1, len(lam_list) - 1):   
        lam_v_inner.append(lam_list[i] - delta_lam)
        lam_v_inner.append(lam_list[i] + delta_lam)
    lam_v_inner = np.asarray(lam_v_inner)
    
    phi_v_inner = np.unique(phi_v_inner)
    lam_v_inner = np.unique(lam_v_inner)
    
    # ---------- compute voxel model boundaries - outer model ----------
    phi_v = np.concatenate((
        [2.0 * phi_list[0] - phi_v_inner[0]],
        phi_v_inner,
        [2.0 * phi_list[-1] - phi_v_inner[-1]]
    ))
    lam_v = np.concatenate((
        [2.0 * lam_list[0] - lam_v_inner[0]],
        lam_v_inner,
        [2.0 * lam_list[-1] - lam_v_inner[-1]]
    ))
    
    # ---------- store original center heights ----------
    h_voxc = h_N 
    
    # ---------- compute heights of voxel edges (h_voxel) ----------
    h_voxel = np.zeros(h_voxc.size + 1, dtype=float)
    h_voxel[0] = h_voxc[0] - (h_voxc[1] - h_voxc[0]) / 2.0
    for i in range(1, h_voxc.size + 1):
        h_voxel[i] = h_voxel[i - 1] + 2.0 * (h_voxc[i - 1] - h_voxel[i - 1])
    
    # ---------- Ellipsoid / Gaussian radius (R_g) ----------
    a = 6378137.0
    b = 6356752.3142
    
    e2 = (a**2 - b**2) / (b**2)
    cel = a**2 / b
    
    # phi_st used in degrees in original code
    V = np.sqrt(1.0 + e2 * (cosd(phi_st)**2))
    dNell = cel / V
    dMell = cel / (V**3)
    R_g = np.sqrt(dMell * dNell)   # scalar
    
    # ---------- Determine height levels for given voxel grid ----------
    # Complement angle to azimuth (az), see law of sin
    # sin_az computed for each phi_v element
    sin_az = sind(azi) * cosd(phi_st) / cosd(phi_v)   # vectorized over phi_v
    # now compute az where |sin_az| <= 1
    mask_valid = np.abs(sin_az) <= 1
    az_vals = np.full_like(sin_az, np.nan, dtype=float)
    az_vals[mask_valid] = asind_deg(sin_az[mask_valid])
    
    # Geocentric angle for lon
    eta_lon = acotd( (sind(azi) / tand(lam_v - lam_st)) / cosd(phi_st) + tand(phi_st) * cosd(azi) )
    
    # Geocentric angle(s) for lat (two solutions per valid phi_v)
    phi_v_valid = phi_v[mask_valid]   # subset used in the eta_lat formulas
    az_valid = az_vals[mask_valid]
    
    # avoid division by zero in the denominator: sind((azi-az)/2)
    num = tand((phi_st - phi_v_valid) / 2.0) * sind((azi + az_valid) / 2.0)
    den = sind((azi - az_valid) / 2.0)
    eta_lat1 = np.arctan2(num, den) * rad2deg
    eta_lat1 = eta_lat1*2
    
    az_alt = 180.0 - az_valid
    num2 = tand((phi_st - phi_v_valid) / 2.0) * sind((azi + az_alt) / 2.0)
    den2 = sind((azi - az_alt) / 2.0)
    eta_lat2 = np.arctan2(num2, den2) * rad2deg
    eta_lat2 = eta_lat2*2
    
    # We'll flatten to one array of lat-etas (concatenate)
    eta_lat = np.concatenate((eta_lat1, eta_lat2))
    
    # Compute height for given voxel model boundaries
    # h_lat: using eta_lat; h_lon: using eta_lon
    h_lat = (cosd(ele) / cosd(ele + eta_lat) - 1.0) * R_g + h_st
    h_lon = (cosd(ele) / cosd(ele + eta_lon) - 1.0) * R_g + h_st
    
    # extract all relevant height levels (unique); convert h_voxel to 1D
    h_voxel_arr = np.asarray(h_voxel).ravel()
    h_N_all = np.unique(np.concatenate((np.asarray(h_lat).ravel(), np.asarray(h_lon).ravel(), h_voxel_arr)))
    
    # keep only positive heights <= last voxel edge
    h_N_filtered = h_N_all[(h_N_all > 0.0) & (h_N_all <= h_voxel_arr[-1])]
    
    # define distance from earth center (Earth radius + height level)
    r = R_g + h_N_filtered
    
    # remove all pressure levels below or equal to station height (h_st)
    mask_above_st = h_N_filtered > h_st
    h_N_filtered = h_N_filtered[mask_above_st]
    r = r[mask_above_st]
    
    # create new arrays beginning at the station height
    h_N = np.concatenate(([h_st], h_N_filtered))
    r = np.concatenate(([R_g + h_st], r))
    
    # ---------- ray-tracing initialization ----------
    n_levels = h_N.size
    
    # initialize arrays
    s = np.zeros(n_levels - 1, dtype=float)
    z = np.zeros(n_levels, dtype=float)
    y = np.zeros(n_levels, dtype=float)
    eta = np.zeros(n_levels, dtype=float)
    delta = np.zeros(n_levels, dtype=float)
    theta = np.zeros(n_levels, dtype=float)
    phi_p = np.zeros(n_levels, dtype=float)
    lam_p = np.zeros(n_levels, dtype=float)
    i_pos = np.zeros(n_levels, dtype=int)
    
    # set start elevation and azimuth angle [rad]
    e0 = ele * deg2rad
    a0 = azi * deg2rad
    
    # latitude and longitude of start point (in radians)
    phi_p[0] = phi_st * deg2rad
    lam_p[0] = lam_st * deg2rad
    
    # get entries (i_pos) of closest voxel column
    # NOTE: this assumes find_num_v(lat_deg, lon_deg, phi_v, lam_v) returns an integer index.
    i_pos[0] = fv.find_num_v(phi_p[0] * rad2deg, lam_p[0] * rad2deg, phi_v, lam_v)
    
    # define values for the first and second point on the ray
    theta[0] = e0  # in [rad]
    
    # calculate distance s[0] from the first to the second point
    s[0] = -r[0] * np.sin(theta[0]) + np.sqrt(r[1]**2 - r[0]**2 * (np.cos(theta[0]))**2)
    
    # define z1 and z2
    z[0] = r[0]
    z[1] = z[0] + s[0] * np.sin(e0)
    
    # define y2 (y1 = 0)
    y[0] = 0.0
    y[1] = s[0] * np.cos(e0)
    
    # eta1 and eta2
    eta[0] = 0.0
    eta[1] = np.arctan2(y[1], z[1])
    
    # theta2
    theta[1] = np.arccos(np.cos(theta[0] + eta[1]))
    
    # arc length delta between start and second point
    delta[1] = eta[1]
    
    # latitude and longitude of second point (in radians)
    phi_p[1] = np.arcsin(np.sin(phi_p[0]) * np.cos(eta[1]) + np.cos(phi_p[0]) * np.sin(eta[1]) * np.cos(a0))
    # lam_p formula uses cot; use 1/np.tan for cot
    lam_p[1] = lam_p[0] + np.arctan2(np.sin(a0), (1.0 / np.tan(eta[1]) * np.cos(phi_p[0]) - np.sin(phi_p[0]) * np.cos(a0)))
    
    # get entries (i_pos) of closest voxel column for second point
    i_pos[1] = fv.find_num_v(phi_p[1] * rad2deg, lam_p[1] * rad2deg, phi_v, lam_v)
    
    # ---------- loop over remaining height levels ----------
    for i in range(1, n_levels - 1):
        # s(i) formula
        s[i] = -r[i] * np.sin(theta[i]) + np.sqrt(r[i + 1]**2 - r[i]**2 * (np.cos(theta[i]))**2)
        z[i + 1] = z[i] + s[i] * np.sin(e0)
        y[i + 1] = y[i] + s[i] * np.cos(e0)
        eta[i + 1] = np.arctan2(y[i + 1], z[i + 1])
        delta[i + 1] = eta[i + 1] - eta[i]
        theta[i + 1] = np.arccos(np.cos(theta[i] + delta[i + 1]))
        phi_p[i + 1] = np.arcsin(np.sin(phi_p[0]) * np.cos(eta[i + 1]) + np.cos(phi_p[0]) * np.sin(eta[i + 1]) * np.cos(a0))
        lam_p[i + 1] = lam_p[0] + np.arctan2(np.sin(a0), (1.0 / np.tan(eta[i + 1]) * np.cos(phi_p[0]) - np.sin(phi_p[0]) * np.cos(a0)))
        # get entries (i_pos) of closest voxel column
        i_pos[i + 1] = fv.find_num_v(phi_p[i + 1] * rad2deg, lam_p[i + 1] * rad2deg, phi_v, lam_v)

    
    id_in = (
    (phi_p * rad2deg <= phi_v[-1]) &
    (phi_p * rad2deg >= phi_v[0]) &
    (lam_p * rad2deg <= lam_v[-1]) &
    (lam_p * rad2deg >= lam_v[0])
    )

    # counter variable
    indices = np.where((h_N[id_in][0] - h_voxel) < 0)[0]
    n = indices[0] - 1 if indices.size > 0 else None
    ray = {"d_voxel": [], "n_voxel": []}

    for j in range(len(h_N[id_in]) - 1):
        # check if intersection is not empty
        if np.intersect1d(h_voxel, h_N[j]).size > 0:
            n += 1

        # Ray path in each voxel
        ray["d_voxel"].append(s[j])
        ray["n_voxel"].append([i_pos[j + 1], n])
        
    # Affected voxel: stack i_pos[j+1] with n
    return ray