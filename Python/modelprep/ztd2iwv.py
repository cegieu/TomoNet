import numpy as np

def ztd2iwv(ZTD, P_s, T_s, lat, h, vmf1w=None):
    """
    Convert Zenith Total Delay (ZTD) to Integrated Water Vapour (IWV),
    and optionally to Slant IWV using VMF1 wet mapping factor.

    Parameters
    ----------
    ZTD : float or array
        Zenith Total Delay [m]
    P_s : float or array
        Surface pressure [hPa]
    T_s : float or array
        Surface temperature [K]
    lat : float
        Latitude [deg]
    h   : float
        Ellipsoidal height [m]
    vmf1w : float or array, optional
        VMF1 wet mapping factor

    Returns
    -------
    IWV : float or ndarray
        Zenith IWV [mm]
    SIWV : float or ndarray or None
        Slant IWV [mm] if vmf1w is given, else None
    """
    phi = np.deg2rad(lat)
    H = h / 1000.0
    ZHD = 0.0022768 * P_s / (1 - 0.00266*np.cos(2*phi) - 0.00028*H)

    ZWD = ZTD - ZHD

    T_m = 70.2 + 0.72 * T_s

    k2_prime = 22.1
    k3 = 3.739e3
    R_v = 461.5
    rho_w = 1000.0

    Pi = 1e-6 * (rho_w * R_v * ((k3 / T_m) + k2_prime))

    IWV = Pi * (ZWD * 1000.0)

    SIWV = IWV * vmf1w if vmf1w is not None else None

    return IWV, SIWV