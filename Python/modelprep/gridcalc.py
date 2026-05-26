import numpy as np
import NWMgrid as NWMg

def gridcalc(XLAT, XLON, undugpt, pres, temp, specHum, geop, layers, lat, lon, struc, model):
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
            # extract vertical profiles (1D arrays)
            hPrf = geomPrf[i, j, :].ravel()    # profile heights
            tPrf = temp[i, j, :].ravel()
            pPrf = pres[i, j, :].ravel()
            ePrf = eg[i, j, :].ravel()
            gPrf = gm[i, j, :].ravel()[:layers.size]   # gm for each layer level
            # If hPrf contains NaN or not strictly increasing, handle later
            if np.all(np.isnan(hPrf)):
                continue

            # heights inside ERA layers where we want to interpolate: Hbet
            # find intH values between hPrf[0] and hPrf[-1]
            Hbet_mask = (intH >= hPrf[0]) & (intH <= hPrf[-1])
            Hbet = intH[Hbet_mask]

            # below model heights
            Hbl_mask = intH < hPrf[0]
            Hbl = intH[Hbl_mask]

            # above model heights (indices into intH)
            hr_inds = np.where(intH > hPrf[-1])[0]

            # find layers where want to interpolate below model
            indB_inds = np.where(layers <= hPrf[0])[0]

            # If there are RT heights below model
            if indB_inds.size > 0:
                # Virtual temperature Tv at first layer
                Tv0 = tPrf[0] * pPrf[0] / (pPrf[0] - (1 - Mw/Md) * ePrf[0])
                # Extrapolate pressure from model's 1st layer
                # gPrf for indices indB_inds (gm needs correspond levels)
                # use gPrf at those layers if available, else use gPrf[0]
                # We'll create g_profile for indB_inds by indexing gm array; fallback to first
                g_profile = gPrf[indB_inds] if indB_inds.size <= gPrf.size else np.full(indB_inds.size, gPrf[0])
                presW[i, j, indB_inds] = pPrf[0] * np.exp(-g_profile * (layers[indB_inds] - hPrf[0]) * 1000.0 / (Rd * Tv0))
                # Keep water vapor pressure unchanged below model
                eW[i, j, indB_inds] = ePrf[0]
                # Linear (simple) temperature extrapolation below using lapse rate 6.5 K/km
                tbl = tPrf[0] + (hPrf[0] - Hbl) * 6.5
                # place into tempW temporarily at indices indB_inds
                tempW[i, j, indB_inds] = tbl

            # Linear interpolation of temperature inside model (Hbet)
            if Hbet.size > 0:
                # numpy.interp expects increasing x; ensure hPrf is increasing
                try:
                    tempbt = np.interp(Hbet, hPrf, tPrf)
                except Exception:
                    tempbt = np.full_like(Hbet, np.nan)
            else:
                tempbt = np.array([])

            # Prepare Cea for vapor extrapolation between model layers
            # Cea = (hPrf[1:] - hPrf[:-1]) * 1000 / log(ePrf[1:]/ePrf[:-1])
            with np.errstate(divide='ignore', invalid='ignore'):
                denom_log = np.log(np.divide(ePrf[1:], ePrf[:-1]))
                denom_log[~np.isfinite(denom_log)] = np.nan
                Cea = (hPrf[1:] - hPrf[:-1]) * 1000.0 / denom_log

            # Virtual temperature per level
            Tv = tPrf * pPrf / (pPrf - (1 - Mw/Md) * ePrf)

            # Interpolate/extrapolate between each pair of model levels
            for hw in range(len(hPrf) - 1):
                # layers between hPrf[hw] and hPrf[hw+1]
                indW_inds = np.where((layers > hPrf[hw]) & (layers <= hPrf[hw+1]))[0]
                if indW_inds.size == 0:
                    continue
                # compute presw1 and presw2
                # Use broadcasting: layers[indW_inds] is (k,)
                delta1 = (layers[indW_inds] - hPrf[hw]) * 1000.0
                delta2 = (layers[indW_inds] - hPrf[hw+1]) * 1000.0
                # gPrf for those intermediate layers: attempt to index gPrf by indW_inds but length mismatch possible;
                # For simplicity use gPrf[indW_inds] if shapes align else use gm at corresponding layer indexes
                g_for_indW = gPrf[indW_inds] if indW_inds.size <= gPrf.size else np.full(indW_inds.size, gPrf[hw])
                presw1 = pPrf[hw] * np.exp(- g_for_indW * delta1 / (Rd * Tv[hw]))
                presw2 = pPrf[hw+1] * np.exp(- g_for_indW * delta2 / (Rd * Tv[hw+1]))
                presw1[~np.isfinite(presw1)] = 0.0
                presw2[~np.isfinite(presw2)] = 0.0
                presW[i, j, indW_inds] = (presw1 + presw2) / 2.0

                # water vapor extrapolation
                if np.all(np.isfinite(Cea[hw])):
                    eh1 = ePrf[hw] * np.exp((layers[indW_inds] - hPrf[hw]) * 1000.0 / Cea[hw])
                    eh2 = ePrf[hw+1] * np.exp((layers[indW_inds] - hPrf[hw+1]) * 1000.0 / Cea[hw])
                else:
                    eh1 = np.zeros(indW_inds.size)
                    eh2 = np.zeros(indW_inds.size)

                eh1[eh1 < 0] = 0.0
                eh2[eh2 < 0] = 0.0
                avg_e = (eh1 + eh2) / 2.0
                # guard against complex or nan
                avg_e = np.real(avg_e)
                avg_e[~np.isfinite(avg_e)] = 0.0
                eW[i, j, indW_inds] = avg_e

            # Heights above top model layer
            if hr_inds.size > 0:
                Hab = intH[hr_inds]
                # Convert to geopotential height layers above [km]
                XLATa_single = np.repeat(XLAT[i, j], Hab.size)[np.newaxis, :]
                geopAbove = NWMg.fgeom2geop(XLATa_single, Hab)  # shape (1, len(Hab)) or (len(Hab),)
                # US standard atmosphere for temperature above only (geopotential height)
                _, _, tempab, _, _, _ = NWMg.stdatmo(geopAbove * 1000.0)
                # extrapolate pressure above
                # use pPrf[-1] and tPrf[-1] and geom heights
                top_p = pPrf[-1]
                top_t = tPrf[-1]
                geom_top_val = hPrf[-1]
                # gm for hr_inds
                gm_hr = gPrf[hr_inds] if hr_inds.size <= gPrf.size else np.full(hr_inds.size, gPrf[-1])
                presW[i, j, hr_inds] = top_p * np.exp(- gm_hr * (Hab - geom_top_val) * 1000.0 / (Rd * top_t))
                # temp correction between last model layer and first RT height above the model
                geopTemp = NWMg.fgeom2geop(XLAT[i, j], np.array([geom_top_val]))
                _, _, tempTemp, _, _, _ = NWMg.stdatmo(geopTemp * 1000.0)
                corr = tPrf[-1] - tempTemp
                tempab = tempab + corr

                # store tempab in a slice - we'll assemble final tempW below

            # assemble temperature vertical profile for this node
            # tempW slice is size len(intH)
            # position sub-slices:
            # below model: we already assigned for indB_inds
            # inside model: assign tempbt into locations Hbet_mask
            # above model: assign tempab into hr_inds
            if Hbet.size > 0:
                # find indices where Hbet_mask True
                idxs = np.where(Hbet_mask)[0]
                # but Hbet is subset, need to map to intH positions
                pos_in_intH = np.where(Hbet_mask)[0]
                tempW[i, j, Hbet_mask] = tempbt

            if hr_inds.size > 0:
                tempW[i, j, hr_inds] = tempab

    # Fill NaNs in eW by 2D neighbor mean (simple approach)
    for w in range(eW.shape[2]):
        layer = eW[:, :, w]
        nan_mask = np.isnan(layer)
        if np.any(nan_mask):
            rows, cols = np.where(nan_mask)
            for r, c in zip(rows, cols):
                candidates = []
                # neighbor indices
                for rr, cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0 <= rr < (XLAT.shape[0]) and 0 <= cc < (XLAT.shape[1]):
                        val = layer[rr, cc]
                        if np.isfinite(val):
                            candidates.append(val)
                if candidates:
                    eW[r, c] = np.mean(candidates)
                else:
                    eW[r, c] = 0.0

    presW[presW < 0] = 0.0

    # create additional layer for satellite param
    zeroGrid = np.zeros((XLAT.shape[0], XLAT.shape[1], 1))
    onesGrid = np.ones((XLAT.shape[0], XLAT.shape[1], 1))

    # concatenate along vertical axis: tempW (intH), tempRay (satmH), onesGrid
    tempGrid = np.concatenate((tempW, tempRay, onesGrid), axis=2) if tempRay.size > 0 else np.concatenate((tempW, onesGrid), axis=2)
    eGrid = np.concatenate((eW, eAb, zeroGrid), axis=2)
    pGrid = np.concatenate((presW, presh, zeroGrid), axis=2)

    # radWGS same shape as sqrt(Rns * Rew)
    radWGS = np.sqrt(Rns * Rew)
    # Final grids
    tempGrid = np.concatenate([tempW, tempRay, np.ones((XLAT.shape[0], XLAT.shape[1], 1))], axis=2)
    eGrid = np.concatenate([eW, eAb, np.zeros((XLAT.shape[0], XLAT.shape[1], 1))], axis=2)
    pGrid = np.concatenate([presW, presh, np.zeros((XLAT.shape[0], XLAT.shape[1], 1))], axis=2)
    radWGS = np.sqrt(Rns * Rew)[:, :, np.newaxis]

    return pGrid, eGrid, tempGrid, radWGS