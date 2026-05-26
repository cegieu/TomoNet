import numpy as np

def UNB3M(lat_rad: float, height_m: float, doy: float, elev_rad: float):
    """
    UNB3M: compute slant neutral atmosphere delays using UNB3m model.

    Inputs:
        lat_rad  : geodetic latitude in radians
        height_m : orthometric height in meters
        doy      : day-of-year (1..365.25)
        elev_rad : elevation angle in radians

    Returns:
        RTROP : total slant delay (m)
        HZD   : hydrostatic zenith delay (m)
        HMF   : hydrostatic Niell mapping function
        WZD   : wet (non-hydrostatic) zenith delay (m)
        WMF   : wet Niell mapping function
    """
    # Lookup tables (rows: lat=15,30,45,60,75)
    AVG = np.array([
        [15.0, 1013.25, 299.65, 75.00, 6.30, 2.77],
        [30.0, 1017.25, 294.15, 80.00, 6.05, 3.15],
        [45.0, 1015.75, 283.15, 76.00, 5.58, 2.57],
        [60.0, 1011.75, 272.15, 77.50, 5.39, 1.81],
        [75.0, 1013.00, 263.65, 82.50, 4.53, 1.55]
    ])

    AMP = np.array([
        [15.0,  0.00,  0.00,  0.00,  0.00, 0.00],
        [30.0, -3.75,  7.00,  0.00,  0.25, 0.33],
        [45.0, -2.25, 11.00, -1.00,  0.32, 0.46],
        [60.0, -1.75, 15.00, -2.50,  0.81, 0.74],
        [75.0, -0.50, 14.50,  2.50,  0.62, 0.30]
    ])

    ABC_AVG = np.array([
        [15.0, 1.2769934e-3, 2.9153695e-3, 62.610505e-3],
        [30.0, 1.2683230e-3, 2.9152299e-3, 62.837393e-3],
        [45.0, 1.2465397e-3, 2.9288445e-3, 63.721774e-3],
        [60.0, 1.2196049e-3, 2.9022565e-3, 63.824265e-3],
        [75.0, 1.2045996e-3, 2.9024912e-3, 64.258455e-3]
    ])

    ABC_AMP = np.array([
        [15.0, 0.0,          0.0,          0.0],
        [30.0, 1.2709626e-5, 2.1414979e-5, 9.0128400e-5],
        [45.0, 2.6523662e-5, 3.0160779e-5, 4.3497037e-5],
        [60.0, 3.4000452e-5, 7.2562722e-5, 84.795348e-5],
        [75.0, 4.1202191e-5, 11.723375e-5, 170.37206e-5]
    ])

    ABC_W2P0 = np.array([
        [15.0, 5.8021897e-4, 1.4275268e-3, 4.3472961e-2],
        [30.0, 5.6794847e-4, 1.5138625e-3, 4.6729510e-2],
        [45.0, 5.8118019e-4, 1.4572752e-3, 4.3908931e-2],
        [60.0, 5.9727542e-4, 1.5007428e-3, 4.4626982e-2],
        [75.0, 6.1641693e-4, 1.7599082e-3, 5.4736038e-2]
    ])

    # constants
    EXCEN2 = 6.6943799901413e-03
    MD = 28.9644
    MW = 18.0152
    K1 = 77.604
    K2 = 64.79
    K3 = 3.776e5
    R = 8314.34
    C1 = 2.2768e-03
    K2PRIM = K2 - K1 * (MW / MD)
    RD = R / MD
    DTR = 1.745329251994329e-02  # not used explicitly later, kept for completeness
    DOY2RAD = 2.0 * np.pi / 365.25

    A_HT = 2.53e-5
    B_HT = 5.49e-3
    C_HT = 1.14e-3
    HT_TOPCON = 1 + A_HT / (1 + B_HT / (1 + C_HT))

    # Latitude deg
    lat_deg = lat_rad * 180.0 / np.pi

    # Southern hemisphere adjustment for seasonal phase
    tdoy = doy
    if lat_deg < 0:
        tdoy = tdoy + 182.625
    cosphs = np.cos((tdoy - 28.0) * DOY2RAD)

    # absolute latitude
    LAT = abs(lat_deg)

    # Choose table indices (0-based)
    if LAT >= 75.0:
        p1 = p2 = 4
        m = 0.0
    elif LAT <= 15.0:
        p1 = p2 = 0
        m = 0.0
    else:
        # MATLAB: P1 = fix((LAT - 15)/15) + 1
        # Python 0-based:
        p1 = int((LAT - 15.0) / 15.0)        # yields 0..3
        p2 = p1 + 1
        # interpolation fraction M
        lat1 = AVG[p1, 0]
        lat2 = AVG[p2, 0]
        m = (LAT - lat1) / (lat2 - lat1)

    # Interpolate AVG values
    # AVG columns: [lat, P, T, RH, beta, lambda]
    PAVG = m * (AVG[p2, 1] - AVG[p1, 1]) + AVG[p1, 1]
    TAVG = m * (AVG[p2, 2] - AVG[p1, 2]) + AVG[p1, 2]
    EAVG = m * (AVG[p2, 3] - AVG[p1, 3]) + AVG[p1, 3]
    BETAAVG = m * (AVG[p2, 4] - AVG[p1, 4]) + AVG[p1, 4]
    LAMBDAAVG = m * (AVG[p2, 5] - AVG[p1, 5]) + AVG[p1, 5]

    # Interpolate AMP values
    PAMP = m * (AMP[p2, 1] - AMP[p1, 1]) + AMP[p1, 1]
    TAMP = m * (AMP[p2, 2] - AMP[p1, 2]) + AMP[p1, 2]
    EAMP = m * (AMP[p2, 3] - AMP[p1, 3]) + AMP[p1, 3]
    BETAAMP = m * (AMP[p2, 4] - AMP[p1, 4]) + AMP[p1, 4]
    LAMBDAAMP = m * (AMP[p2, 5] - AMP[p1, 5]) + AMP[p1, 5]

    # Surface tropo values (seasonal)
    P0 = PAVG - PAMP * cosphs
    T0 = TAVG - TAMP * cosphs
    E0 = EAVG - EAMP * cosphs
    BETA = BETAAVG - BETAAMP * cosphs
    BETA = BETA / 1000.0     # original code divides by 1000
    LAMBDA = LAMBDAAVG - LAMBDAAMP * cosphs

    # Transform relative humidity to water vapor pressure (IERS 2003)
    # ES formula:
    # ES = 0.01 * exp(1.2378847e-5 * (T0 ^ 2) - 1.9121316e-2 * T0 + 3.393711047e1 - 6.3431645e3 * (T0 ^ -1));
    ES = 0.01 * np.exp(1.2378847e-5 * (T0 ** 2) - 1.9121316e-2 * T0 + 3.393711047e1 - 6.3431645e3 * (1.0 / T0))
    FW = 1.00062 + 3.14e-6 * P0 + 5.6e-7 * ((T0 - 273.15) ** 2)
    E0 = (E0 / 100.0) * ES * FW

    # Compute power value for pressure & water vapour
    EP = 9.80665 / 287.054 / BETA

    # Scale surface values to required height
    T = T0 - BETA * height_m
    # avoid negative or zero T/T0 - but model assumes reasonable heights
    # compute P and E
    P = P0 * (T / T0) ** EP
    E = E0 * (T / T0) ** (EP * (LAMBDA + 1.0))

    # Acceleration at mass center of vertical column
    geolat = np.arctan((1.0 - EXCEN2) * np.tan(lat_rad))
    dgref = 1.0 - 2.66e-03 * np.cos(2.0 * geolat) - 2.8e-07 * height_m
    gm = 9.784 * dgref
    den = (LAMBDA + 1.0) * gm

    # Mean temperature of water vapor
    TM = T * (1.0 - BETA * RD / den)

    # Zenith hydrostatic delay
    HZD = C1 / dgref * P

    # Zenith wet delay
    WZD = 1.0e-6 * (K2PRIM + K3 / TM) * RD * E / den

    # NMF(H) coefficients interpolation (ABC_AVG and ABC_AMP)
    A_AVG = m * (ABC_AVG[p2, 1] - ABC_AVG[p1, 1]) + ABC_AVG[p1, 1]
    B_AVG = m * (ABC_AVG[p2, 2] - ABC_AVG[p1, 2]) + ABC_AVG[p1, 2]
    C_AVG = m * (ABC_AVG[p2, 3] - ABC_AVG[p1, 3]) + ABC_AVG[p1, 3]

    A_AMP = m * (ABC_AMP[p2, 1] - ABC_AMP[p1, 1]) + ABC_AMP[p1, 1]
    B_AMP = m * (ABC_AMP[p2, 2] - ABC_AMP[p1, 2]) + ABC_AMP[p1, 2]
    C_AMP = m * (ABC_AMP[p2, 3] - ABC_AMP[p1, 3]) + ABC_AMP[p1, 3]

    A = A_AVG - A_AMP * cosphs
    B = B_AVG - B_AMP * cosphs
    C = C_AVG - C_AMP * cosphs

    # compute sine of elevation
    SINE = np.sin(elev_rad)
    # avoid zero division:
    if SINE <= 0.0:
        # for negative or zero elevation, set mapping to large number (practical choice)
        SINE = 1e-8

    # NMF(H)
    ALPHA = B / (SINE + C)
    GAMMA = A / (SINE + ALPHA)
    TOPCON = (1.0 + A / (1.0 + B / (1.0 + C)))
    HMF = TOPCON / (SINE + GAMMA)

    # height correction for hydrostatic mapping function
    ALPHA = B_HT / (SINE + C_HT)
    GAMMA = A_HT / (SINE + ALPHA)
    HT_CORR_COEF = 1.0 / SINE - HT_TOPCON / (SINE + GAMMA)
    HT_CORR = HT_CORR_COEF * height_m / 1000.0  # height in km in original formula
    HMF = HMF + HT_CORR

    # NMF(W) coefficients interpolation (ABC_W2P0)
    A = m * (ABC_W2P0[p2, 1] - ABC_W2P0[p1, 1]) + ABC_W2P0[p1, 1]
    B = m * (ABC_W2P0[p2, 2] - ABC_W2P0[p1, 2]) + ABC_W2P0[p1, 2]
    C = m * (ABC_W2P0[p2, 3] - ABC_W2P0[p1, 3]) + ABC_W2P0[p1, 3]

    ALPHA = B / (SINE + C)
    GAMMA = A / (SINE + ALPHA)
    TOPCON = (1.0 + A / (1.0 + B / (1.0 + C)))
    WMF = TOPCON / (SINE + GAMMA)

    # Total slant delay
    RTROP = HZD * HMF + WZD * WMF

    return float(RTROP), float(HZD), float(HMF), float(WZD), float(WMF)

def UNB3MM(LATRAD: float, HEIGHTM: float, DAYOYEAR: float):
    """
    Python implementation of the UNB3m model (UNB3MM).

    Parameters
    ----------
    LATRAD : float
        Station geodetic latitude (radians)
    HEIGHTM : float
        Station orthometric height (m)
    DAYOYEAR : float
        Day of year

    Returns
    -------
    T : float
        Surface temperature (K)
    P : float
        Surface pressure (mbar)
    E : float
        Surface water vapor pressure (mbar)
    TM : float
        Mean temperature of water vapor (K)
    """

    # --- Lookup tables ---
    AVG = np.array([
        [15.0, 1013.25, 299.65, 75.00, 6.30, 2.77],
        [30.0, 1017.25, 294.15, 80.00, 6.05, 3.15],
        [45.0, 1015.75, 283.15, 76.00, 5.58, 2.57],
        [60.0, 1011.75, 272.15, 77.50, 5.39, 1.81],
        [75.0, 1013.00, 263.65, 82.50, 4.53, 1.55]
    ])
    AMP = np.array([
        [15.0,   0.00,   0.00,   0.00, 0.00, 0.00],
        [30.0,  -3.75,   7.00,   0.00, 0.25, 0.33],
        [45.0,  -2.25,  11.00,  -1.00, 0.32, 0.46],
        [60.0,  -1.75,  15.00,  -2.50, 0.81, 0.74],
        [75.0,  -0.50,  14.50,   2.50, 0.62, 0.30]
    ])

    # --- Constants ---
    EXCEN2 = 6.6943799901413e-03
    MD     = 28.9644
    MW     = 18.0152
    K1     = 77.604
    K2     = 64.79
    K3     = 3.776e5
    R      = 8314.34
    RD     = R / MD
    DTR    = np.pi / 180.0
    DOY2RAD = 2.0 * np.pi / 365.25

    # --- Latitude in degrees ---
    LATDEG = LATRAD * 180.0 / np.pi

    # --- Adjust for southern hemisphere ---
    TD_O_Y = DAYOYEAR
    if LATDEG < 0:
        TD_O_Y += 182.625
    COSPHS = np.cos((TD_O_Y - 28) * DOY2RAD)

    # --- Interpolation pointers ---
    LAT = abs(LATDEG)
    if LAT >= 75:
        P1 = P2 = 4
        M = 0.0
    elif LAT <= 15:
        P1 = P2 = 0
        M = 0.0
    else:
        P1 = int((LAT - 15) // 15)
        P2 = P1 + 1
        M = (LAT - AVG[P1, 0]) / (AVG[P2, 0] - AVG[P1, 0])

    # --- Average values (interpolated) ---
    PAVG, TAVG, EAVG, BETAAVG, LAMBDAAVG = AVG[P1, 1:] + \
        M * (AVG[P2, 1:] - AVG[P1, 1:])
    PAMP, TAMP, EAMP, BETAAMP, LAMBDAAMP = AMP[P1, 1:] + \
        M * (AMP[P2, 1:] - AMP[P1, 1:])

    # --- Surface values ---
    P0 = PAVG - PAMP * COSPHS
    T0 = TAVG - TAMP * COSPHS
    E0 = EAVG - EAMP * COSPHS
    BETA = (BETAAVG - BETAAMP * COSPHS) / 1000.0
    LAMBDA = LAMBDAAVG - LAMBDAAMP * COSPHS

    # --- Water vapor pressure (IERS 2003) ---
    ES = 0.01 * np.exp(
        1.2378847e-5 * T0**2
        - 1.9121316e-2 * T0
        + 33.93711047
        - 6343.1645 / T0
    )
    FW = 1.00062 + 3.14e-6 * P0 + 5.6e-7 * (T0 - 273.15)**2
    E0 = (E0 / 100.0) * ES * FW

    # --- Power for scaling ---
    EP = 9.80665 / 287.054 / BETA

    # --- Scale to station height ---
    T = T0 - BETA * HEIGHTM
    P = P0 * (T / T0) ** EP
    E = E0 * (T / T0) ** (EP * (LAMBDA + 1.0))

    # --- Mean temperature of water vapor ---
    GEOLAT = np.arctan((1.0 - EXCEN2) * np.tan(LATRAD))
    DGREF = 1.0 - 2.66e-3 * np.cos(2.0 * GEOLAT) - 2.8e-7 * HEIGHTM
    GM = 9.784 * DGREF
    DEN = (LAMBDA + 1.0) * GM
    TM = T * (1.0 - BETA * RD / DEN)

    return T, P, E, TM