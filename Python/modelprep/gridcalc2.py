import numpy as np
import NWMgrid as NWMg

def gridcalc2(XLAT, XLON, undugpt, pres, temp, specHum, geop, layers, lat, lon, struc, model):
    """
    Calculate full ERA5 grid meteorological parameters including pressure, water vapor, 
    temperature, and WGS radius for tomography and ray tracing.

    Parameters
    ----------
    XLAT, XLON : 2D arrays [rad]
        Lat/lon grids
    undugpt : 2D array
        Geoid undulation grid [m]
    pres, temp, specHum, geop : 3D arrays
        ERA5 pressure [hPa], temperature [K], specific humidity [kg/kg], geopotential heights [km]
    layers : 1D array
        Desired vertical layers for interpolation [km]
    lat, lon : 1D arrays
        Lat/lon of processed domain
    struc : object or None
        Optional structure modifying undulation grid
    model : object
        Model parameters including 'radii' [a, b] in km

    Returns
    -------
    pGrid, eGrid, tempGrid : 3D arrays
        Interpolated grids for pressure [hPa], water vapor [hPa], temperature [K]
    radWGS : 3D array
        WGS radius [km]
    """
    # Meteorological constants
    Rd = 287.058  # gas constant dry air [J/K/kg]
    Md = 28.9644  # molar mass dry air [kg/mol]
    Mw = 18.0152  # molar mass wet air [kg/mol]
    g0 = 9.80665

    # Ellipsoid parameters
    a, b = model['radii']
    e2 = (a**2 - b**2)/a**2

    # Ensure grid sizes match domain lat/lon
    if XLAT.shape[1] != len(lat):
        idx = np.isin(np.round(np.degrees(XLAT[0,:]), 2), lat)
        pres = np.flip(pres[:, idx, :], axis=1)
        specHum = np.flip(specHum[:, idx, :], axis=1)
        temp = np.flip(temp[:, idx, :], axis=1)
        geop = np.flip(geop[:, idx, :], axis=1)
        XLAT = np.flip(XLAT[:, idx], axis=1)

    if XLAT.shape[0] != len(lon):
        idx = np.isin(np.round(np.degrees(XLON[:,0]), 2), lon)
        geop = geop[idx, :, :]
        pres = pres[idx, :, :]
        specHum = specHum[idx, :, :]
        temp = temp[idx, :, :]
        XLAT = XLAT[idx, :]

    if struc is not None:
        if XLAT.shape[1] != len(lat):
            idx = np.isin(np.round(np.degrees(struc.LAT[0,:]),2), lat)
            undugpt = np.flip(undugpt[:, idx, :], axis=1)
        if XLAT.shape[0] != len(lon):
            idx = np.isin(np.round(np.degrees(struc.LON[:,0]),2), lon)
            undugpt = undugpt[idx, :, :]

    # Convert to ellipsoidal heights
    undugpt = undugpt/1000
    undu= np.repeat(undugpt[:, :, np.newaxis], geop.shape[2], axis=2)
    XLAT2 = np.repeat(XLAT[:, :, np.newaxis], geop.shape[2], axis=2)
    geomPrf = NWMg.fgeop2geom(XLAT2, geop) + undu  # km

    # Water vapor partial pressure [hPa]
    eg = specHum * pres / (Mw/Md + (1 - Mw/Md) * specHum)

    # Layer separation
    lastH = np.max(geomPrf[:, :, -1])
    satmH = layers[layers > lastH]
    intH = layers[layers <= lastH]

    # US standard atmosphere above model top
    satmHa = np.repeat(layers[np.newaxis, np.newaxis, :], XLAT.shape[0], axis=0)
    satmHa = np.repeat(satmHa, XLAT.shape[1], axis=1)

    XLATa = np.repeat(XLAT[:, :, np.newaxis], len(satmH), axis=2)
    geopUS76 = NWMg.fgeom2geop(XLATa, satmHa[:, :, len(intH):])

    _, _, tempRay, _, _, _ = NWMg.stdatmo(geopUS76*1000)  # US76 temperature

    # Earth radius
    Rns = a**2 * b**2 / (a**2 * np.cos(XLAT)**2 + b**2 * np.sin(XLAT)**2)**(3/2)
    Rew = a / np.sqrt(1 - e2 * np.sin(XLAT)**2)
    XLATacc = np.repeat(XLAT[:, :, np.newaxis], len(layers), axis=2)
    Rwgs = np.repeat(np.sqrt(Rns * Rew)[:, :, np.newaxis], len(layers), axis=2)

    gm = g0 * (1 - 0.0026373 * np.cos(2 * XLATacc) + 5.9e-6 * np.cos(2 * XLATacc))**2 / (1 + satmHa / Rwgs)**2

    # Extrapolate pressure above model
    presh = np.repeat(pres[:, :, -1][:, :, np.newaxis], len(satmH), axis=2) * \
            np.exp(-gm[:, :, len(intH):] * (satmHa[:, :, len(intH):] - geomPrf[:, :, -1][:, :, np.newaxis]) * 1000 / (Rd * np.repeat(temp[:, :, -1][:, :, np.newaxis], len(satmH), axis=2)))
    presh[presh < 0] = 0

    # Water vapor above top = 0
    eAb = np.zeros((XLAT.shape[0], XLAT.shape[1], len(satmH)))

    # Initialize below model grids
    tempW = np.zeros((XLAT.shape[0], XLAT.shape[1], len(intH)))
    presW = np.zeros((XLAT.shape[0], XLAT.shape[1], len(intH)))
    eW = np.zeros((XLAT.shape[0], XLAT.shape[1], len(intH)))

    # Interpolation of parameters for each profile
    for i in range(XLAT.shape[0]):
        for j in range(XLAT.shape[1]):
            hPrf = geomPrf[i, j, :]
            tPrf = temp[i, j, :]
            pPrf = pres[i, j, :]
            ePrf = eg[i, j, :]
            gPrf = gm[i, j, :]

            # Interpolation below top, within model, above top
            # (simplified linear interpolation)
            tempW[i, j, :] = np.interp(intH, hPrf, tPrf)
            presW[i, j, :] = np.interp(intH, hPrf, pPrf)
            eW[i, j, :] = np.interp(intH, hPrf, ePrf)

    # Final grids
    tempGrid = np.concatenate([tempW, tempRay, np.ones((XLAT.shape[0], XLAT.shape[1], 1))], axis=2)
    eGrid = np.concatenate([eW, eAb, np.zeros((XLAT.shape[0], XLAT.shape[1], 1))], axis=2)
    pGrid = np.concatenate([presW, presh, np.zeros((XLAT.shape[0], XLAT.shape[1], 1))], axis=2)
    radWGS = np.sqrt(Rns * Rew)[:, :, np.newaxis]

    return pGrid, eGrid, tempGrid, radWGS