from __future__ import annotations
import math
from typing import Tuple
import numpy as np
from typing import Tuple
from timerecalc import jd2doy
from UNB3 import UNB3MM
from pathlib import Path


def gpt2(
    dmjd: float,
    dlat: np.ndarray,
    dlon: np.ndarray,
    hell: np.ndarray,
    nstat: int,
    it: int,
    epoch,
    pathgrd,
    grdmodel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    
    Compute atmospheric and mapping-function related quantities from GPT2 grid.

    Parameters
    ----------
    dmjd : float
        modified Julian date (scalar)
    dlat : (nstat,) array_like
        ellipsoidal latitude in radians [-pi/2 : +pi/2]
    dlon : (nstat,) array_like
        longitude in radians (either -pi..pi or 0..2*pi)
    hell : (nstat,) array_like
        ellipsoidal height in meters
    nstat : int
        number of stations (length of dlat/dlon/hell)
    it : int
        if it == 1, use constant fields (no time variation). if 0 include
        annual and semi-annual time terms.
    grid_path : str
        path to gpt2_5.grd (default 'gpt2_5.grd')

    Returns
    -------
    p, T, dT, e, ah, aw, undu : tuple of numpy arrays
        p   : pressure in hPa, shape (nstat,)
        T   : temperature in deg C, shape (nstat,)
        dT  : temperature lapse rate in deg / km, shape (nstat,)
        e   : water vapour pressure in hPa, shape (nstat,)
        ah  : hydrostatic mapping-function coefficient (at zero height), shape (nstat,)
        aw  : wet mapping-function coefficient, shape (nstat,)
        undu: geoid undulation in m, shape (nstat,)
    """

    # constants
    dmjd1 = dmjd - 51544.5  # change epoch to Jan 1, 2000
    gm = 9.80665  # m/s^2
    dMtr = 28.965e-3  # kg/mol
    Rg = 8.3143  # J/(K*mol)
    grid_path = Path(pathgrd)/ "gpt2_5.grd"
    # time factors (annual & semiannual)
    if it == 1:
        cosfy = coshy = sinfy = sinhy = 0.0
    else:
        ang1 = dmjd1 / 365.25 * 2.0 * math.pi
        cosfy = math.cos(ang1)
        sinfy = math.sin(ang1)
        ang2 = dmjd1 / 365.25 * 4.0 * math.pi
        coshy = math.cos(ang2)
        sinhy = math.sin(ang2)

    # read grid file
    # file expected to have a header line, then 2592 data lines with at least 34 floats per line
    ngrid = 2592
    # allocate arrays: each grid point has 5 coefficients for mean, cos yearly, sin yearly, cos semi, sin semi
    pgrid = np.zeros((ngrid, 5), dtype=float)
    Tgrid = np.zeros((ngrid, 5), dtype=float)
    Qgrid = np.zeros((ngrid, 5), dtype=float)
    dTgrid = np.zeros((ngrid, 5), dtype=float)
    u = np.zeros(ngrid, dtype=float)      # geoid undulation (m)
    Hs = np.zeros(ngrid, dtype=float)     # orthometric grid height (m)
    ahgrid = np.zeros((ngrid, 5), dtype=float)
    awgrid = np.zeros((ngrid, 5), dtype=float)
    if epoch == 0:
        with open(grid_path, "r") as fh:
            # skip first comment line
            _ = fh.readline()
            for n in range(ngrid):
                line = fh.readline()
                if not line:
                    raise EOFError(f"Unexpected end of grid file at line {n+2}")
                vec = [float(x) for x in line.strip().split()]
                # Expecting at least 34 floats per line (like the MATLAB version)
                if len(vec) < 34:
                    raise ValueError(f"Grid line {n+2} has too few entries ({len(vec)})")
    
                pgrid[n, :] = vec[2:7]
                Tgrid[n, :] = vec[7:12]
                Qgrid[n, :] = np.array(vec[12:17]) / 1000.0
                dTgrid[n, :] = np.array(vec[17:22]) / 1000.0
                u[n] = vec[22]
                Hs[n] = vec[23]
                ahgrid[n, :] = np.array(vec[24:29]) / 1000.0
                awgrid[n, :] = np.array(vec[29:34]) / 1000.0
                
        grdmodel = {
            "pgrid": pgrid,
            "Tgrid": Tgrid,
            "dTgrid": dTgrid,
            "Qgrid": Qgrid,
            "u": u,
            "Hs": Hs,
            "ahgrid": ahgrid,
            "awgrid": awgrid
            }   
    else:
        pgrid = grdmodel["pgrid"]
        Tgrid = grdmodel["Tgrid"]
        dTgrid = grdmodel["dTgrid"]
        Qgrid = grdmodel["Qgrid"]
        u = grdmodel["u"]
        Hs = grdmodel["Hs"]
        ahgrid = grdmodel["ahgrid"]
        awgrid = grdmodel["awgrid"]

    # prepare outputs
    p = np.zeros(nstat, dtype=float)
    T = np.zeros(nstat, dtype=float)
    dT = np.zeros(nstat, dtype=float)
    e = np.zeros(nstat, dtype=float)
    ah = np.zeros(nstat, dtype=float)
    aw = np.zeros(nstat, dtype=float)
    undu = np.zeros(nstat, dtype=float)

    # convenience
    two_pi = 2.0 * math.pi

    # process each station
    for k in range(nstat):
        if np.isscalar(dlat):
            lat = float(dlat)
            lon = float(dlon)
            height_ell = float(hell)
        else:
            lat = float(dlat[k])
            lon = float(dlon[k])
            height_ell = float(hell[k])

        # ensure positive longitude in degrees 0..360
        if lon < 0.0:
            plon = (lon + two_pi) * 180.0 / math.pi
        else:
            plon = lon * 180.0 / math.pi

        ppod = (-lat + 0.5 * math.pi) * 180.0 / math.pi

        ipod = math.floor((ppod + 5.0) / 5.0)    
        ilon = math.floor((plon + 5.0) / 5.0)    

        diffpod = (ppod - (ipod * 5.0 - 2.5)) / 5.0
        difflon = (plon - (ilon * 5.0 - 2.5)) / 5.0

        if ipod == 37:
            ipod = 36

        # convert to zero-based for Python array indexing:
        indx1 = int((ipod - 1) * 72 + ilon - 1)

        # check bilinear applicability (nearest neighbor near poles)
        bilinear = (ppod > 2.5) and (ppod < 177.5)

        if not bilinear:
            ix = indx1
            undu_k = u[ix]
            undu[k] = undu_k
            hgt = height_ell - undu_k
            # evaluate grid coefficients with time terms
            T0 = (Tgrid[ix, 0] + Tgrid[ix, 1] * cosfy + Tgrid[ix, 2] * sinfy +
                  Tgrid[ix, 3] * coshy + Tgrid[ix, 4] * sinhy)
            p0 = (pgrid[ix, 0] + pgrid[ix, 1] * cosfy + pgrid[ix, 2] * sinfy +
                  pgrid[ix, 3] * coshy + pgrid[ix, 4] * sinhy)
            Q = (Qgrid[ix, 0] + Qgrid[ix, 1] * cosfy + Qgrid[ix, 2] * sinfy +
                 Qgrid[ix, 3] * coshy + Qgrid[ix, 4] * sinhy)
            dT_k = (dTgrid[ix, 0] + dTgrid[ix, 1] * cosfy + dTgrid[ix, 2] * sinfy +
                    dTgrid[ix, 3] * coshy + dTgrid[ix, 4] * sinhy)
            # station height - grid height
            redh = hgt - Hs[ix]
            # temperature at station height in Celsius
            T_k = T0 + dT_k * redh - 273.15
            # lapse rate in deg / km
            dT[k] = dT_k * 1000.0
            T[k] = T_k
            # virtual temperature in Kelvin
            Tv = T0 * (1.0 + 0.6077 * Q)
            c = gm * dMtr / (Rg * Tv)
            # pressure in hPa
            p[k] = (p0 * math.exp(-c * redh)) / 100.0
            # water vapour pressure in hPa
            e[k] = (Q * p[k]) / (0.622 + 0.378 * Q)
            # mapping-function coefficients
            ah[k] = (ahgrid[ix, 0] + ahgrid[ix, 1] * cosfy + ahgrid[ix, 2] * sinfy +
                     ahgrid[ix, 3] * coshy + ahgrid[ix, 4] * sinhy)
            aw[k] = (awgrid[ix, 0] + awgrid[ix, 1] * cosfy + awgrid[ix, 2] * sinfy +
                     awgrid[ix, 3] * coshy + awgrid[ix, 4] * sinhy)
        else:
            # bilinear interpolation: indices of four surrounding grid nodes
            # ipod1 = ipod + sign(diffpod); ilon1 = ilon + sign(difflon)
            def sgn(x: float) -> int:
                if x > 0:
                    return 1
                if x < 0:
                    return -1
                return 0

            ipod1 = int(ipod + sgn(diffpod))
            ilon1 = int(ilon + sgn(difflon))

            # wrap longitude index if necessary (1..72)
            if ilon1 == 73:
                ilon1 = 1
            if ilon1 == 0:
                ilon1 = 72

            # construct 1-based indices and convert to zero-based python indices
            indx = np.zeros(4, dtype=int)
            indx[0] = int((ipod - 1) * 72 + ilon - 1)      # same as indx1
            indx[1] = int((ipod1 - 1) * 72 + ilon - 1)     # along same longitude (pod+sign, lon)
            indx[2] = int((ipod - 1) * 72 + ilon1 - 1)     # along same polar dist (pod, lon+sign)
            indx[3] = int((ipod1 - 1) * 72 + ilon1 - 1)    # diagonal

            pl = np.zeros(4, dtype=float)
            Tl = np.zeros(4, dtype=float)
            dTl = np.zeros(4, dtype=float)
            Ql = np.zeros(4, dtype=float)
            ahl = np.zeros(4, dtype=float)
            awl = np.zeros(4, dtype=float)
            undul = np.zeros(4, dtype=float)

            for l in range(4):
                ix = int(indx[l])
                undul[l] = u[ix]
                hgt = height_ell - undul[l]
                T0 = (Tgrid[ix, 0] + Tgrid[ix, 1] * cosfy + Tgrid[ix, 2] * sinfy +
                      Tgrid[ix, 3] * coshy + Tgrid[ix, 4] * sinhy)
                p0 = (pgrid[ix, 0] + pgrid[ix, 1] * cosfy + pgrid[ix, 2] * sinfy +
                      pgrid[ix, 3] * coshy + pgrid[ix, 4] * sinhy)
                Ql[l] = (Qgrid[ix, 0] + Qgrid[ix, 1] * cosfy + Qgrid[ix, 2] * sinfy +
                         Qgrid[ix, 3] * coshy + Qgrid[ix, 4] * sinhy)
                Hs1 = Hs[ix]
                redh = hgt - Hs1
                dTl[l] = (dTgrid[ix, 0] + dTgrid[ix, 1] * cosfy + dTgrid[ix, 2] * sinfy +
                          dTgrid[ix, 3] * coshy + dTgrid[ix, 4] * sinhy)
                Tl[l] = T0 + dTl[l] * redh - 273.15
                Tv = T0 * (1.0 + 0.6077 * Ql[l])
                c = gm * dMtr / (Rg * Tv)
                pl[l] = (p0 * math.exp(-c * redh)) / 100.0
                ahl[l] = (ahgrid[ix, 0] + ahgrid[ix, 1] * cosfy + ahgrid[ix, 2] * sinfy +
                          ahgrid[ix, 3] * coshy + ahgrid[ix, 4] * sinhy)
                awl[l] = (awgrid[ix, 0] + awgrid[ix, 1] * cosfy + awgrid[ix, 2] * sinfy +
                          awgrid[ix, 3] * coshy + awgrid[ix, 4] * sinhy)

            dnpod1 = abs(diffpod)
            dnpod2 = 1.0 - dnpod1
            dnlon1 = abs(difflon)
            dnlon2 = 1.0 - dnlon1

            # pressure bilinear
            R1 = dnpod2 * pl[0] + dnpod1 * pl[1]
            R2 = dnpod2 * pl[2] + dnpod1 * pl[3]
            p[k] = dnlon2 * R1 + dnlon1 * R2

            # temperature bilinear ( C already)
            R1 = dnpod2 * Tl[0] + dnpod1 * Tl[1]
            R2 = dnpod2 * Tl[2] + dnpod1 * Tl[3]
            T[k] = dnlon2 * R1 + dnlon1 * R2

            # lapse rate deg/km
            R1 = dnpod2 * dTl[0] + dnpod1 * dTl[1]
            R2 = dnpod2 * dTl[2] + dnpod1 * dTl[3]
            dT[k] = (dnlon2 * R1 + dnlon1 * R2) * 1000.0

            # humidity -> water vapour pressure
            R1 = dnpod2 * Ql[0] + dnpod1 * Ql[1]
            R2 = dnpod2 * Ql[2] + dnpod1 * Ql[3]
            Q = dnlon2 * R1 + dnlon1 * R2
            e[k] = (Q * p[k]) / (0.622 + 0.378 * Q)

            # hydrostatic & wet coefficients
            R1 = dnpod2 * ahl[0] + dnpod1 * ahl[1]
            R2 = dnpod2 * ahl[2] + dnpod1 * ahl[3]
            ah[k] = dnlon2 * R1 + dnlon1 * R2

            R1 = dnpod2 * awl[0] + dnpod1 * awl[1]
            R2 = dnpod2 * awl[2] + dnpod1 * awl[3]
            aw[k] = dnlon2 * R1 + dnlon1 * R2

            # undulation
            R1 = dnpod2 * undul[0] + dnpod1 * undul[1]
            R2 = dnpod2 * undul[2] + dnpod1 * undul[3]
            undu[k] = dnlon2 * R1 + dnlon1 * R2

    return p, T, dT, e, ah, aw, undu, grdmodel


# small helper to satisfy local name usage earlier
def cos(x: float) -> float:
    return math.cos(x)


def sin(x: float) -> float:
    return math.sin(x)


def distr_e_unb3RT(
    BLh_pudel_num: np.ndarray, undu: np.ndarray, jd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate water vapor pressure distribution (UNB3 model) inside tomography model.

    Parameters
    ----------
    BLh_pudel_num : ndarray
        Voxel centers [3D grid: lat, lon, h]
    undu : ndarray
        Geoid undulation (same length as h)
    jd : ndarray
        Julian day(s)

    Returns
    -------
    E3D : ndarray
        Water vapor partial pressure distribution [mbar], reshaped to 3D grid
    P : ndarray
        Pressure
    """

    wi2, ki2, wa2 = BLh_pudel_num.shape
    BLh_pudel_num_2D = np.zeros((3,wa2* ki2))
    cc = BLh_pudel_num[1,:,:]
    cc = cc.T.reshape(1,ki2*wa2)
    dd = BLh_pudel_num[0,:,0]
    dd = np.tile(dd, wa2)
    ee = BLh_pudel_num[2,0,:]
    ee = np.tile(ee,ki2)
    BLh_pudel_num_2D[0,:] = dd
    BLh_pudel_num_2D[1,:] = cc
    BLh_pudel_num_2D[2,:] = np.sort(ee)


    B = BLh_pudel_num_2D[0, :].T  # lat
    L = BLh_pudel_num_2D[1, :].T  # lon
    h = BLh_pudel_num_2D[2, :].T  # height

    H = h - undu

    # Convert to radians
    B = np.deg2rad(B)
    L = np.deg2rad(L)

    # Convert JD -> DOY
    doy,_ = jd2doy(jd)  
    if np.isscalar(doy):       # True if x is a single value (int, float, etc.)
        c = 1
    else:
        c = len(doy) 
    E = np.zeros((len(B), c))
    P = np.zeros((len(B), c))
    if np.isscalar(doy):  
        for j in range(len(B)):
            _, P[j, 0], E[j, 0], _ = UNB3MM(B[j], H[j], doy)
    else:
        for i in range(c):
            for j in range(len(B)):
                _, P[j, i], E[j, i], _ = UNB3MM(B[j], H[j], doy[i])

    return E, P

def distr_T_gpt2RT(BLh_pudel_num: np.ndarray, dmj: int,pathgrd, epoch, grdmodel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate temperature distribution inside tomography model (using GPT2).

    Parameters
    ----------
    BLh_pudel_num : ndarray
        Voxel centers [3D grid: lat, lon, h]
    dmj : int
        Day of year

    Returns
    -------
    T : ndarray
        Temperature [K], flattened
    T3D : ndarray
        Temperature [K], reshaped to 3D grid
    undu : ndarray
        Geoid undulation
    """
   
    wi2, ki2, wa2 = BLh_pudel_num.shape
    BLh_pudel_num_2D = np.zeros((3,wa2* ki2))
    cc = BLh_pudel_num[1,:,:]
    cc = cc.T.reshape(1,ki2*wa2)
    dd = BLh_pudel_num[0,:,0]
    dd = np.tile(dd, wa2)
    ee = BLh_pudel_num[2,0,:]
    ee = np.tile(ee,ki2)
    BLh_pudel_num_2D[0,:] = dd
    BLh_pudel_num_2D[1,:] = cc
    BLh_pudel_num_2D[2,:] = np.sort(ee)
    
    B = BLh_pudel_num_2D[0, :].T  # latitude
    L = BLh_pudel_num_2D[1, :].T  # longitude
    h = BLh_pudel_num_2D[2, :].T  # height

    # Convert degrees to radians
    B = np.deg2rad(B)
    L = np.deg2rad(L)

    # Call GPT2 model (must be implemented separately)
    dmj = dmj-2400000.5
    p, t, dT, E, ah, aw, undu, grdmodel = gpt2(dmj, B, L, h, len(B), 0, epoch, pathgrd, grdmodel)

    # Convert Celsius to Kelvin
    T = t + 273.15
    T = T.T

    # Reshape to 3D grid
    T3D = T.reshape(1, ki2, wa2)

    return T, T3D, undu, grdmodel

def eT2Nw(e, T):
    """
    Calculate wet refractivity (Nw) in a voxel.
    
    Parameters
    ----------
    e : array_like
        Water vapor partial pressure [mbar], shape (n, 1) or (n,)
    T : array_like
        Temperature [K], same shape as e

    Returns
    -------
    Nw : np.ndarray
        Wet refractivity, shape (n, 1)
    Nw3D : np.ndarray
        Placeholder for 3D refractivity, currently empty
    """
    k2 = 72
    k3 = 370100

    e = np.asarray(e).reshape(-1)
    T = np.asarray(T).reshape(-1)

    Nw = np.zeros_like(e, dtype=float)

    for i in range(len(e)):
        Zv = 1 + e[i] * (1 + 3.7e-4 * e[i]) * (
            -2.37321e-3 + 2.23366 / T[i] - 710.792 / T[i]**2 + 7.76147e4 / T[i]**3
        )
        Nw[i] = (k2 * e[i] / T[i] + k3 * e[i] / T[i]**2) * Zv

    Nw = Nw.reshape(-1, 1)
    Nw3D = np.array([]) 
    
    return Nw, Nw3D