from typing import Dict, Any
import numpy as np
from scipy import interpolate
import find_num_v as fv

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

def sind(x): return np.sin(np.asarray(x) * deg2rad)
def cosd(x): return np.cos(np.asarray(x) * deg2rad)
def tand(x): return np.tan(np.asarray(x) * deg2rad)
def asind_deg(x): return np.degrees(np.arcsin(np.asarray(x)))
def atan2d(y, x): return np.degrees(np.arctan2(np.asarray(y), np.asarray(x)))

def acot(x):
    x = np.asarray(x, dtype=float)
    return np.arctan(1.0 / x)

def acotd(x):
    return np.arctan(1 / np.asarray(x)) * rad2deg

# Simple wrapper for interp1 and interp2
def interp1d_matlab(x, y, xi, kind='linear', fill_value='extrapolate'):
    """Mimics MATLAB interp1 for 1D linear/extrapolate. x and xi are 1D."""
    f = interpolate.interp1d(x, y, kind=kind, axis=0, fill_value=fill_value, bounds_error=False)
    return f(xi)

def interp2_spline(lat_grid, lon_grid, values2d, phi_q, lam_q):
    """
    Mimic MATLAB interp2(..., 'spline') for a regular grid.
    lat_grid, lon_grid are 2D as in your MATLAB code (shape nlon x nlat).
    values2d must have the same shape. phi_q, lam_q are points in degrees.
    This constructs a RectBivariateSpline assuming lat/lon are separable and sorted.
    """
    # Extract unique sorted coordinates from the grid
    lat_1d = np.unique(lat_grid)
    lon_1d = np.unique(lon_grid)
    vals = np.asarray(values2d)
    try:
        rbs = interpolate.RectBivariateSpline(lat_1d, lon_1d, vals)
    except Exception:
        # try transpose
        rbs = interpolate.RectBivariateSpline(lat_1d, lon_1d, vals.T)
    # query points (handle scalar or array)
    phi_q = np.atleast_1d(phi_q)
    lam_q = np.atleast_1d(lam_q)
    result = np.array([rbs(phi, lam)[0, 0] for phi, lam in zip(phi_q, lam_q)])
    # if originally scalars, return scalar
    if result.size == 1:
        return result[0]
    return result

# ---------- The main function ----------
def voxel_dist_2D(
    ref_h: np.ndarray,
    ref_w: np.ndarray,
    phi_N: np.ndarray,
    lam_N: np.ndarray,
    h_N_center: np.ndarray,
    phi_st: float,
    lam_st: float,
    h_st: float,
    ele: float,
    azi: float
) -> Dict[str, Any]:
    """
    Python translation of MATLAB voxel_dist_2D.
    Inputs:
      - ref_h, ref_w: 2D arrays with shape (n_heights, n_columns)
      - phi_N, lam_N: 2D arrays with shape (1, n_columns) typically (replicated later)
      - h_N_center: 1D array of center heights (column vector in MATLAB)
      - phi_st, lam_st, h_st, ele, azi: station scalar parameters (degrees, m, degrees)
    Returns:
      - ray: dict with keys e0, e_out, geom_bend_voxel (np.array), d_voxel (np.array), n_voxel (np.array Nx2)
    """
    # copy inputs / ensure arrays
    ref_h = np.asarray(ref_h, dtype=float)
    ref_w = np.asarray(ref_w, dtype=float)
    phi_N = np.asarray(phi_N)
    lam_N = np.asarray(lam_N)
    h_voxc = np.asarray(h_N_center).astype(float).ravel()  # center heights

    # ----- heights of voxel edges (h_voxel) -----
    h_voxel = np.zeros(h_voxc.size + 1, dtype=float)
    h_voxel[0] = h_voxc[0] - (h_voxc[1] - h_voxc[0]) / 2.0
    for i in range(1, h_voxc.size + 1):
        h_voxel[i] = h_voxel[i - 1] + 2.0 * (h_voxc[i - 1] - h_voxel[i - 1])

    h_N_new = np.arange(h_voxel[0], h_voxel[-1] + 1e-9, 50.0)  # 50 m steps

    # interpolate ref_h/ref_w to new heights h_N_new for each column
    ncols = ref_h.shape[1]
    Nh = np.zeros((h_N_new.size, ncols), dtype=float)
    Nw = np.zeros((h_N_new.size, ncols), dtype=float)
    for i in range(ncols):
        Nh[:, i] = interp1d_matlab(h_voxc, ref_h[:, i], h_N_new)
        Nw[:, i] = interp1d_matlab(h_voxc, ref_w[:, i], h_N_new)

    # set negative values to zero
    Nh[Nh < 0] = 0.0
    Nw[Nw < 0] = 0.0

    # position vectors for new heights: replicate phi_N and lam_N down the new height axis
    phi_N_rep = np.tile(phi_N, (h_N_new.size, 1))
    lam_N_rep = np.tile(lam_N, (h_N_new.size, 1))

    # restore new values for return/processing
    h_N = np.tile(h_N_new.reshape(-1, 1), (1, ncols))
    ref_w = Nw.copy()
    ref_h = Nh.copy()

    # ----- ellipsoid parameters and Gaussian radius -----
    a = 6378137.0
    b = 6356752.3142
    e2 = (a**2 - b**2) / (b**2)
    cel = a**2 / b
    V = np.sqrt(1.0 + e2 * (cosd(phi_st)**2))
    dNell = cel / V
    dMell = cel / (V**3)
    R_g = np.sqrt(dMell * dNell)

    # total refractivities
    ref = ref_h + ref_w  

    # distance from earth center
    r = R_g + h_N[:, 0]

    # refractivity at station height (interpolated)
    rref = R_g + h_st
    if h_st < h_N[0, 0]:
        nref = ref[0, :].copy()
    else:
        nref = np.zeros(ncols, dtype=float)
        for i in range(ncols):
            nref[i] = np.interp(h_st, h_N[:, i], ref[:, i])

    # discard levels <= station height
    mask_above = h_N[:, 0] > h_st
    h_N = h_N[mask_above, :]
    r = r[mask_above]
    ref = ref[mask_above, :]

    # create new arrays starting at station height
    h_N = np.concatenate((np.tile(h_st, (1, ncols)), h_N), axis=0)
    r = np.concatenate((np.array([rref]), r), axis=0)
    ref = np.vstack((nref, ref))

    # calculate mean refractivities between levels (mref)
    mref = np.zeros_like(ref, dtype=float)
    for i in range(ref.shape[0] - 1):
        mref[i, :] = 0.5 * (ref[i + 1, :] + ref[i, :])
    mref[-1, :] = ref[-1, :]

    # refractive index (1 + N*1e-6)
    mref = mref * 1e-6 + 1.0

    # grid coords for interp2
    nlon = len(np.unique(lam_N))
    lat_grid = phi_N_rep[0, :].reshape(nlon, -1)   # may need to adapt shape based on your inputs
    lon_grid = lam_N_rep[0, :].reshape(nlon, -1)

    # --- initialize ray tracing variables ---
    n_levels = h_N.shape[0]
    s = np.zeros(n_levels - 1, dtype=float)
    z = np.zeros(n_levels, dtype=float)
    y = np.zeros(n_levels, dtype=float)
    eta = np.zeros(n_levels, dtype=float)
    delta = np.zeros(n_levels, dtype=float)
    theta = np.zeros(n_levels, dtype=float)
    eps = np.zeros(n_levels, dtype=float)
    phi_p = np.zeros(n_levels, dtype=float)
    lam_p = np.zeros(n_levels, dtype=float)
    i_pos = np.zeros(n_levels, dtype=int)

    # bending correction at station
    g_bend_st = 0.02 * np.exp(-h_N[0, 0] / 6000.0) / tand(ele)

    # starting angles (radians)
    e0 = ele * deg2rad + g_bend_st * deg2rad
    a0 = azi * deg2rad

    eps_final = -90.0 * deg2rad

    # start lat/lon in radians
    phi_p[0] = phi_st * deg2rad
    lam_p[0] = lam_st * deg2rad

    phi_list = np.unique(phi_N_rep[0, :])
    lam_list = np.unique(lam_N_rep[0, :])

    # build inner boundaries
    if phi_list.size < 3 or lam_list.size < 3:
        raise ValueError("phi/lam list must have at least 3 unique values")
    phi_v_inner = []
    for i in range(1, phi_list.size - 1):
        phi_v_inner.append(phi_list[i] - (phi_list[2] - phi_list[1]) / 2.0)
        phi_v_inner.append(phi_list[i] + (phi_list[2] - phi_list[1]) / 2.0)
    phi_v_inner = np.unique(np.asarray(phi_v_inner))

    lam_v_inner = []
    for i in range(1, lam_list.size - 1):
        lam_v_inner.append(lam_list[i] - (lam_list[2] - lam_list[1]) / 2.0)
        lam_v_inner.append(lam_list[i] + (lam_list[2] - lam_list[1]) / 2.0)
    lam_v_inner = np.unique(np.asarray(lam_v_inner))

    phi_v = np.concatenate(([2.0 * phi_list[0] - phi_v_inner[0]], phi_v_inner, [2.0 * phi_list[-1] - phi_v_inner[-1]]))
    lam_v = np.concatenate(([2.0 * lam_list[0] - lam_v_inner[0]], lam_v_inner, [2.0 * lam_list[-1] - lam_v_inner[-1]]))

    # initial i_pos (0-based)
    i_pos[0] = fv.find_num_v(phi_p[0] * rad2deg, lam_p[0] * rad2deg, phi_v, lam_v)

    n_iter = 0
    max_iter = 10

    # iterative loop until outgoing elevation matches target
    while abs(ele * deg2rad - eps_final) > 1e-5 and n_iter < max_iter:
        theta[0] = e0
        eps[0] = theta[0]
        # compute first step s[0]
        s[0] = -r[0] * np.sin(theta[0]) + np.sqrt(max(0.0, r[1]**2 - r[0]**2 * (np.cos(theta[0])**2)))
        z[0] = r[0]
        y[0] = 0.0
        z[1] = z[0] + s[0] * np.sin(eps[0])
        y[1] = y[0] + s[0] * np.cos(eps[0])
        eta[0] = 0.0
        eta[1] = np.arctan2(y[1], z[1])

        # interpolate mref to current ray position (first two levels)
        mref_new = np.ones(2, dtype=float)
        try:
            # use interp2-like spline
            mref_new[0] = interp2_spline(lat_grid, lon_grid, mref[0, :].reshape(nlon, -1), phi_p[0] * rad2deg, lam_p[0] * rad2deg)
            mref_new[1] = interp2_spline(lat_grid, lon_grid, mref[1, :].reshape(nlon, -1), phi_p[0] * rad2deg, lam_p[0] * rad2deg)
        except Exception:
            # fallback to 1D interpolation across columns (nearest)
            col_idx = int(i_pos[0])
            mref_new[0] = mref[0, col_idx]
            mref_new[1] = mref[1, col_idx]

        # replace NaNs with 1
        mref_new = np.where(np.isnan(mref_new), 1.0, mref_new)

        # theta[1]
        arg = (mref_new[0] / mref_new[1]) * np.cos(theta[0] + eta[1])
        theta[1] = np.arccos(arg)
        eps[1] = theta[1] - eta[1]
        delta[1] = eta[1]
        phi_p[1] = np.arcsin(np.sin(phi_p[0]) * np.cos(eta[1]) + np.cos(phi_p[0]) * np.sin(eta[1]) * np.cos(a0))
        lam_p[1] = lam_p[0] + np.arctan2(np.sin(a0), (1.0 / np.tan(eta[1]) * np.cos(phi_p[0]) - np.sin(phi_p[0]) * np.cos(a0)))
        i_pos[1] = fv.find_num_v(phi_p[1] * rad2deg, lam_p[1] * rad2deg, phi_v, lam_v)

        # loop over remaining height levels
        inside_break = False
        for i in range(1, n_levels - 1):
            s[i] = -r[i] * np.sin(theta[i]) + np.sqrt(max(0.0, r[i + 1]**2 - r[i]**2 * (np.cos(theta[i])**2)))
            z[i + 1] = z[i] + s[i] * np.sin(eps[i])
            y[i + 1] = y[i] + s[i] * np.cos(eps[i])
            eta[i + 1] = np.arctan2(y[i + 1], z[i + 1])
            delta[i + 1] = eta[i + 1] - eta[i]

            # interpolate mref at levels i and i+1 to ray position
            try:
                mref_new_1 = interp2_spline(lat_grid, lon_grid, mref[i, :].reshape(nlon, -1), phi_p[i] * rad2deg, lam_p[i] * rad2deg)
                mref_new_2 = interp2_spline(lat_grid, lon_grid, mref[i + 1, :].reshape(nlon, -1), phi_p[i] * rad2deg, lam_p[i] * rad2deg)
            except Exception:
                col_idx = int(i_pos[i])
                mref_new_1 = mref[i, col_idx]
                mref_new_2 = mref[i + 1, col_idx]

            mref_new_1 = 1.0 if np.isnan(mref_new_1) else mref_new_1
            mref_new_2 = 1.0 if np.isnan(mref_new_2) else mref_new_2

            arg = (mref_new_1 / mref_new_2) * np.cos(theta[i] + delta[i + 1])
            theta[i + 1] = np.real(np.arccos(arg))
            eps[i + 1] = theta[i + 1] - eta[i + 1]
            phi_p[i + 1] = np.arcsin(np.sin(phi_p[0]) * np.cos(eta[i + 1]) + np.cos(phi_p[0]) * np.sin(eta[i + 1]) * np.cos(a0))
            lam_p[i + 1] = lam_p[0] + np.arctan2(np.sin(a0), (1.0 / np.tan(eta[i + 1]) * np.cos(phi_p[0]) - np.sin(phi_p[0]) * np.cos(a0)))
            i_pos[i + 1] = fv.find_num_v(phi_p[i + 1] * rad2deg, lam_p[i + 1] * rad2deg, phi_v, lam_v)

            # check if ray left voxel model
            if (phi_p[i + 1] * rad2deg >= phi_v[-1] or
                phi_p[i + 1] * rad2deg <= phi_v[0] or
                lam_p[i + 1] * rad2deg >= lam_v[-1] or
                lam_p[i + 1] * rad2deg < lam_v[0]):
                n_iter = max_iter
                inside_break = True
                break

        # bending correction at top and compute eps_final
        g_bend_end = 0.02 * np.exp(-h_N[-1, 0] / 6000.0) / tand(ele)
        eps_final = eps[i + 1] - g_bend_end * deg2rad

        # update e0 for next iteration
        if n_iter < 5:
            e0 = e0 + (ele * deg2rad - eps_final)
            n_iter += 1
        else:
            break


    # compute geometric bending per level
    dgeo = np.zeros(n_levels - 1, dtype=float)
    for n_idx in range(n_levels - 1):
        dgeo[n_idx] = s[n_idx] - np.cos(eps[n_idx] - eps_final) * s[n_idx]

    # delete lowest entries (station)
    h_N = np.delete(h_N, 0, axis=0)
    i_pos = np.delete(i_pos, 0)

    # get unique voxel columns traversed inside ray
    # determine indices inside voxel model:
    id_in_mask = ((phi_p * rad2deg) <= phi_v[-1]) & ((phi_p * rad2deg) >= phi_v[0]) & \
                 ((lam_p * rad2deg) <= lam_v[-1]) & ((lam_p * rad2deg) >= lam_v[0])
    id_in = np.where(id_in_mask)[0].tolist()
    if len(id_in) > 0:
        id_in = id_in[:-1]

    i_vox = np.unique(i_pos[id_in]).astype(int)

    ray = {
        "e0": e0 * rad2deg,
        "e_out": eps_final * rad2deg,
        "geom_bend_voxel": [],
        "d_voxel": [],
        "n_voxel": []
    }

    # Path lengths per voxel (nested loop)
    n_count = 0
    for j in range(len(h_voxel) - 1):
        # vertical selection (indexes of h_N within this vertical voxel)
        ind2 = np.where((h_N[id_in, 0] > h_voxel[j]) & (h_N[id_in, 0] <= h_voxel[j + 1]))[0]
        # ind2 are indices *into* id_in; we need actual id indices
        real_ind2 = [id_in[idx] for idx in ind2]
        for i_col in range(len(i_vox)):
            # horizontal selection (i_pos in those id_in equal to this column)
            ind1 = np.where(i_pos[id_in] == i_vox[i_col])[0]  # indices into id_in
            # intersection:
            inter = np.intersect1d(ind1, ind2)
            if inter.size > 0:
                n_count += 1
                # convert inter (indices into id_in) to actual ids
                actual_ids = [id_in[int(k)] for k in inter]
                geom_val = np.sum(dgeo[actual_ids])
                dval = np.sum(s[actual_ids]) + geom_val
                ray["geom_bend_voxel"].append(geom_val)
                ray["d_voxel"].append(dval)
                # store as (column_index, vertical_index)
                ray["n_voxel"].append([int(i_vox[i_col]), int(j)])
    # convert lists to numpy arrays for convenience
    if ray["geom_bend_voxel"]:
        ray["geom_bend_voxel"] = np.array(ray["geom_bend_voxel"], dtype=float)
        ray["d_voxel"] = np.array(ray["d_voxel"], dtype=float)
        ray["n_voxel"] = np.array(ray["n_voxel"], dtype=int)
    else:
        ray["geom_bend_voxel"] = np.array([])
        ray["d_voxel"] = np.array([])
        ray["n_voxel"] = np.array([[]], dtype=int)

    return ray