import numpy as np
from typing import List, Dict, Any, Tuple

def polyfit_scaled(x: np.ndarray, y: np.ndarray, deg: int):
    """
    Polynomial fit with scaling (like numpy.polynomial.polynomial.polyfit)
    but also returns covariance matrix.
    """
    mu = x.mean()
    scale = (x.max() - x.min()) / 2.0
    if scale == 0:
        scale = 1.0  # avoid division by zero if x is constant

    # rescale x
    x_scaled = (x - mu) / scale

    # fit polynomial in scaled x
    coeffs, cov = np.polyfit(x_scaled, y, deg, cov=True)

    return coeffs, cov, scale, mu


def orb2poly(data: np.ndarray, deg: int, fitlen: float, arclen: float
             ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    orb2poly fits polynomials into SP3 data (separately for X, Y, Z, dTs)

    Parameters
    ----------
    data : ndarray, shape (n,6)
        SP3 data [epoch, prn, X(m), Y(m), Z(m), dTs(s*10^6)]
        (epoch is GPSweek*7*24*3600+GPSsec)
    deg : int
        Polynomial degree. Warning: interpolation may not work properly if deg < 16.
    fitlen : float
        Length of data used for polynomial fit [s]
    arclen : float
        Maximum length of arc [s] (only this part will be used for interpolation)

    Returns
    -------
    poly : list of dict
        List indexed by PRN (poly[prn]) containing dictionaries:
            'from', 'to' - arc start and end epochs
            'X', 'Y', 'Z', 'dT' - polynomial coefficients (highest degree first)
            'X_s', 'X_mu', etc. - scaling factors from polyfit
            'X_cov', etc. - covariance matrices of coefficients
    fr : ndarray
        [n_arc,] start times of arcs
    to : ndarray
        [n_arc,] end times of arcs
    """

    if arclen > fitlen:
        fitlen = arclen

    epoch0 = np.min(data[:, 0])
    epoch1 = np.max(data[:, 0])
    prns = np.unique(data[:, 1]).astype(int)

    n_arc = int(np.ceil((epoch1 - epoch0) / arclen))

    # Initialize outputs
    fr = np.full(n_arc, np.nan)
    to = np.full(n_arc, np.nan)

    # Preallocate poly structure: dict of dicts
    max_prn = int((50 + np.max(prns)) // 50 * 50)
    poly = [[{
        'from': np.nan,
        'to': np.nan,
        'X': np.full(deg + 1, np.nan),
        'X_s': np.nan, 'X_mu': np.nan, 'X_cov': None,
        'Y': np.full(deg + 1, np.nan),
        'Y_s': np.nan, 'Y_mu': np.nan, 'Y_cov': None,
        'Z': np.full(deg + 1, np.nan),
        'Z_s': np.nan, 'Z_mu': np.nan, 'Z_cov': None,
        'dT': np.full(deg + 1, np.nan),
        'dT_s': np.nan, 'dT_mu': np.nan, 'dT_cov': None
    } for _ in range(n_arc)] for _ in range(max_prn + 1)]

    arc_to = epoch1

    for a in range(n_arc - 1, -1, -1):  # reverse loop
        arc_fr = arc_to - arclen + 0.5
        fit_to = arc_to + (fitlen - arclen) / 2
        fit_fr = arc_fr - (fitlen - arclen) / 2

        if fit_to > epoch1:
            fit_to = epoch1
        if fit_fr < epoch0:
            fit_fr = epoch0

        fdata = data[(data[:, 0] >= fit_fr) & (data[:, 0] <= fit_to), :]

        for i, p in enumerate(prns):
            if i == 0:
                fr[a] = arc_fr
                to[a] = arc_to

            pfdata = fdata[fdata[:, 1] == p, :]

            poly[p][a]['from'] = arc_fr
            poly[p][a]['to'] = arc_to

            if pfdata.shape[0] >= deg:
                # X
                c, cov, s, mu = polyfit_scaled(pfdata[:, 0], pfdata[:, 2], deg)
                poly[p][a]['X'] = c
                poly[p][a]['X_cov'] = cov
                poly[p][a]['X_s'], poly[p][a]['X_mu'] = s, mu

                # Y
                c, cov, s, mu = polyfit_scaled(pfdata[:, 0], pfdata[:, 3], deg)
                poly[p][a]['Y'] = c
                poly[p][a]['Y_cov'] = cov
                poly[p][a]['Y_s'], poly[p][a]['Y_mu'] = s, mu

                # Z
                c, cov, s, mu = polyfit_scaled(pfdata[:, 0], pfdata[:, 4], deg)
                poly[p][a]['Z'] = c
                poly[p][a]['Z_cov'] = cov
                poly[p][a]['Z_s'], poly[p][a]['Z_mu'] = s, mu

                # dT
                c, cov, s, mu = polyfit_scaled(pfdata[:, 0], pfdata[:, 5], deg)
                poly[p][a]['dT'] = c
                poly[p][a]['dT_cov'] = cov
                poly[p][a]['dT_s'], poly[p][a]['dT_mu'] = s, mu

        arc_to = arc_fr - 0.5

    return poly, fr, to

def polyval(p: np.ndarray, x: np.ndarray, s: float = 1.0, mu: float = 0.0
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equivalent of MATLAB polyval(p, x, S, mu) using numpy.
    Evaluates a polynomial with scaling and centering.

    Parameters
    ----------
    p : ndarray
        Polynomial coefficients in descending order (highest power first).
    x : ndarray or float
        Points at which to evaluate.
    s : float, optional
        Scaling factor used during polyfit (default 1).
    mu : float, optional
        Mean value used for centering during polyfit (default 0).

    Returns
    -------
    val : ndarray
        Polynomial evaluated at x.
    delta : ndarray
        Placeholder (zeros), since MATLAB returns error estimates here.
    """
    x = np.asarray(x)
    p = np.asarray(p)

    # Apply scaling/centering
    x_adj = (x - mu) / s

    # Evaluate polynomial
    val = np.polyval(p, x_adj)

    delta = np.zeros_like(val)
    return val, delta