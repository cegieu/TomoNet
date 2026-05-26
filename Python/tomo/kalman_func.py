import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse.linalg import svds
from scipy.linalg import svd


def PzeroApr(Nw_apr, A, num_lat, num_lon, levels, BLh_pudel):
    """
    Function to get first estimate of dynamic disturbance in the wet refractivity
    based on provided Nw data (best working NWP forecasts).
    Translated from MATLAB to Python 3.10.
    """

    # Extract height profile (assuming BLh_pudel is 3D: [3,1,:])
    h = np.squeeze(BLh_pudel[2, 0, :])  # MATLAB 3->2 (0-based)

    # Build NWP extension (mean over lat*lon per level)
    NWPext = []
    for i in range(len(Nw_apr)):
        NWPone = np.reshape(Nw_apr[i], (levels,num_lat * num_lon)).T
        NWPext.append(np.mean(NWPone, axis=0))
    NWPext = np.array(NWPext)

    modQ = np.cov(NWPext, rowvar=False)
    modQ = np.sqrt(np.diag(modQ))

    if np.var(modQ) > 0.1:
        # Case: real NWP data
        modQ = np.tile(modQ, (num_lat * num_lon,1)).T
        modQ = np.reshape(modQ, (num_lat * num_lon * levels, 1))
    else:
        # Case: only models available → empirical variance function (Trzcina et al., 2016)
        p1 = [-0.0552, 0.6214]
        p2 = [-0.7693, 7.7347]
        h_km = h / 1000.0

        modQ = np.zeros_like(h_km)

        lin_idx = np.where(h_km <= 3.875)[0]
        modQ[lin_idx] = p1[0] * h_km[lin_idx] + p1[1]

        exp_idx = np.where(h_km > 3.875)[0]
        modQ[exp_idx] = p2[1] * np.exp(p2[0] * h_km[exp_idx])

        modQ = np.tile(modQ, (num_lat * num_lon,1)).T
        modQ = np.reshape(modQ, (num_lat * num_lon * levels, 1))
        modQ = modQ * 9

    # Handle over-constrained case
    if modQ.shape[0] < A.shape[1]:
        extra = np.full((A.shape[1] - modQ.shape[0], 1), 0.00018)
        modQ = np.vstack((modQ, extra))
        
    modQ = modQ**2
    Q = np.diag(np.array(modQ).ravel())

    return Q

def getGain(Pminus, A, R_SWD):
    """
    Estimate the Kalman gain matrix.

    Inputs:
        Pminus : predicted covariance matrix from Kalman filter
        A      : observation matrix with signal derivatives
        R_SWD  : errors of the slant delays

    Outputs:
        K      : Kalman gain matrix
        con    : condition number
        Ainv   : pseudo-inverse of Anew
        Anew   : weighted observation matrix
        thin   : dictionary containing Kalman filtering parameters
    """

    # Weighted observation matrix
    Anew = A @ Pminus @ A.T + np.diag(R_SWD**2)
    ile_w, ile_kol = Anew.shape

    # Compute SVD
    try:
        density = np.count_nonzero(Anew) / Anew.size
        if density < 0.1:
            # Sparse matrix, use svds
            U, S, Vt = svds(Anew, k=min(ile_w, ile_kol))
            V = Vt.T
        else:
            U, S, Vt = svd(Anew, full_matrices=False)
            V = Vt.T
    except Exception:
        # Fallback economic SVD
        U, S, Vt = svd(Anew, full_matrices=False)
        V = Vt.T
        print("Warning: getGain: SVD inversion failed. Using fallback SVD.")

    # Process singular values
    proc1 = np.log(np.diag(S) if S.ndim > 1 else S)
    bzd = np.where(proc1 > -30)[0]
    proc2 = np.diff(proc1[bzd], n=2)
    theta = np.arctan(proc2)

    # Sort by descending absolute theta
    theta_abs = np.abs(theta)
    theta_ind = np.argsort(-theta_abs)

    # Ensure small initial singular values are not selected
    k = 0
    num = theta_ind[k]
    while theta_ind[k] < 0.1 * A.shape[0] and k < len(theta_ind) - 1:
        k += 1
        num = theta_ind[k]

    # Compute condition number
    SD = S
    if num > 0:
        con = SD[0] / SD[num - 1]
    else:
        con = np.nan

    # Limit condition number if too large
    if con > 300:
        con_test = SD[0] / SD[:num-1]
        indices = np.where(con_test < 300)[0]
        if len(indices) > 0:
            num = indices[-1]
            con = con_test[num]

    # Prepare truncated singular value matrix
    SD = SD[:num]
    SD = 1.0 / SD
    SD = np.diag(SD)

    # Construct Splus
    Splus = np.zeros((ile_w, ile_kol))
    Splus[:num, :num] = SD

    # Compute pseudo-inverse and Kalman gain
    Ainv = V @ Splus.T @ U.T
    K = Pminus @ A.T @ Ainv

    # Save Kalman filtering parameters
    thin = {
        "thind": np.sort(theta_ind[:num]),
        "theta": theta
    }

    return K, con, Ainv, Anew, thin

def getGainR(Pminus, A, R_SWD, SWD, xPminus):
    """
    Robust estimation of Kalman gain matrix (Python 3.10 translation).
    
    Inputs:
        Pminus  : predicted covariance matrix
        A       : observational matrix
        R_SWD   : observation errors (slant delays)
        SWD     : observation vector
        xPminus : predicted state vector
        
    Outputs:
        K   : Kalman gain matrix
        Wd  : modified Kalman weights
        thin: structure with Kalman filtering parameters
    """

    sig = 1.0  # apriori sigma for observations
    c = 2.0
    R = np.diag(R_SWD)
    # Initial weight matrix
    W = 1.0 / (R**2)

    # Compute residuals
    try:
        e = SWD - A @ xPminus
    except Exception:
        print("Warning: getGainR: Kalman processing failed")
        e = np.zeros_like(SWD)

    # Adjust weights for robust estimation
    Wd_array = W.copy()
    for i in range(len(W)):
        if np.sqrt(W[i]) * abs(e[i]) > sig * c:
            Wd_array[i] = c * sig * np.sqrt(W[i]) / abs(e[i])

    Wd = 1.0 / np.sqrt(Wd_array)

    # Modified observation matrix for gain computation
    Anew = A @ Pminus @ A.T + sig**2 * (1.0 / W)

    ile_w, ile_kol = Anew.shape

    # Compute SVD
    try:
        if np.count_nonzero(Anew) / Anew.size < 0.1:
            # Sparse matrix, use svds
            U, S, Vt = svds(Anew, k=min(ile_w, ile_kol))
        else:
            # Dense SVD
            U, S, Vt = svd(Anew, full_matrices=False)
    except Exception:
        # Fallback economic SVD
        U, S, Vt = svd(Anew, full_matrices=False)
        print("Warning: getGainR: SVD inversion failed. Using fallback SVD.")

    # Process singular values for robust selection
    proc1 = np.log(S)
    bzd = np.where(proc1 > -30)[0]
    proc2 = np.diff(proc1[bzd], n=2)
    theta = np.arctan(proc2)

    # Sort by absolute theta descending
    theta_abs = np.abs(theta)
    theta_ind = np.argsort(-theta_abs)

    # Select singular values avoiding small initial indices
    k = 0
    num = theta_ind[k]
    while theta_ind[k] < 0.1 * Anew.shape[0] and k < len(theta_ind) - 1:
        k += 1
        num = theta_ind[k]

    # Compute condition number
    SD = S.copy()
    if num > 0:
        con = SD[0] / SD[num - 1]
    else:
        con = np.nan

    if con > 300:
        con_test = SD[0] / SD[:num-1]
        indices = np.where(con_test < 300)[0]
        if len(indices) > 0:
            num = indices[-1]
            con = con_test[num]

    SD = SD[:num]
    SD = 1.0 / SD
    SD = np.diag(SD)

    Splus = np.zeros((ile_w, ile_kol))
    Splus[:num, :num] = SD

    Ainv = Vt.T @ Splus.T @ U.T

    # Kalman gain
    K = Pminus @ A.T @ Ainv

    thin = {
        "thind": np.sort(theta_ind[:num]),
        "theta": theta,
        "Anew": np.diag(Anew)
    }

    return K, Wd, thin


def QApr(Nw_apr, A, num_lat, num_lon, levels, BLh_pudel, switches):
    """
    Function to get first estimate of dynamic disturbance in the wet refractivity
    based on provided Nw data (best working NWP forecasts).
    Translated from MATLAB to Python 3.10.
    """

    # Extract height profile (assuming BLh_pudel is 3D: [3,1,:] in MATLAB → [2,0,:] in Python)
    h = np.squeeze(BLh_pudel[2, 0, :])

    # Build NWP extension (mean over lat*lon per level)
    NWPext = []
    for i in range(len(Nw_apr)):
        NWPone = np.reshape(Nw_apr[i], (levels,num_lat * num_lon)).T
        NWPext.append(np.mean(NWPone, axis=0))
    NWPext = np.array(NWPext)

    modQ = np.cov(NWPext, rowvar=False)
    modQ = np.sqrt(np.diag(modQ))

    if np.var(modQ) > 0.1:
        # Case: real NWP data
        modQ = np.tile(modQ, (num_lat * num_lon,1)).T
        modQ = np.reshape(modQ, (num_lat * num_lon * levels, 1))
    else:
        if switches["solution"] == ['SYNTHETIC']:
            # Synthetic case with reference profile
            href = np.array([150, 275, 400, 575, 750, 1000, 1250, 1500,
                             1750, 2000, 2250, 2500, 2750, 3250, 3750,
                             4500, 5250, 6000, 6750, 7500, 8250, 10000, 11750])
            modQref = np.array([13.29, 12.35, 12.07, 11.63, 11.55, 11.03,
                                10.20, 8.98, 7.91, 6.92, 5.98, 5.17,
                                4.52, 3.43, 2.79, 2.18, 1.75, 1.47,
                                1.19, 0.98, 0.77, 0.44, 0.22])

            # Interpolate with cubic spline
            f = interp1d(href, modQref, kind="cubic", fill_value="extrapolate")
            modQ = f(h)

            modQ = np.tile(modQ.reshape(1, -1), (num_lat * num_lon, 1)).T
            modQ = np.reshape(modQ, (num_lat * num_lon * levels, 1))
        else:
            # Empirical variance model (Trzcina et al., 2016)
            p1 = [-0.0552, 0.6214]
            p2 = [-0.7693, 7.7347]
            h_km = h / 1000.0

            modQ = np.zeros_like(h_km)

            lin_idx = np.where(h_km <= 3.875)[0]
            modQ[lin_idx] = p1[0] * h_km[lin_idx] + p1[1]

            exp_idx = np.where(h_km > 3.875)[0]
            modQ[exp_idx] = p2[1] * np.exp(p2[0] * h_km[exp_idx])

            modQ = np.tile(modQ, (num_lat * num_lon,1)).T
            modQ = np.reshape(modQ, (num_lat * num_lon * levels, 1))
            modQ = modQ * 2

    # Handle over-constrained case
    if modQ.shape[0] < A.shape[1]:
        extra = np.full((A.shape[1] - modQ.shape[0], 1), 0.00018)
        modQ = np.vstack((modQ, extra))

    Q = modQ

    return Q

