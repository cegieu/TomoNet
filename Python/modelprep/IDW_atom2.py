import numpy as np

def IDW_atom2(LAT, LON, Tlayer, Hlayer, Player, Nlayer, Nwlayer, rWGS, llRay, hlayers, settings):
    """
    IDW interpolation on ray point coordinates to calculate refractivity values 
    from temperature, pressure, and water vapor pressure.
    
    Parameters
    ----------
    LAT, LON : 2D arrays [rad]
        Ellipsoidal latitude and longitude of refractivity nodes
    Tlayer, Hlayer, Player : 3D arrays
        Temperature [K], water vapor pressure [hPa], pressure [hPa] at all levels
    rWGS : 2D array
        Earth radius matrix [km]
    llRay : 1D array
        Ray point coordinates [lat, lon, height]
    hlayers : 1D array
        Vertical layers heights [km]
    settings : dict
        Settings placeholder
    
    Returns
    -------
    N : dict
        Refractivity information at ray point
        N['Nh'], N['Nw'], N['Nt'], N['T'], N['p'], N['h']
    """
    # Constants
    Rd = 287.058
    Md = 28.9644
    Mw = 18.0152
    g0 = 9.80665

    # Distances to all nodes
    AD = np.sqrt((np.pi/2 - llRay[0] - LAT)**2 + (llRay[1] - LON)**2)

    # Find 2 lowest distances for IDW
    sdmin = np.sort(np.min(AD, axis=0))[:2]

    node = []
    Tprf, Hprf, Pprf, rWGprf = [], [], [], []
    N_coord = []

    k = 0
    for i in range(2):
        col = np.argwhere(AD == sdmin[i])[0][1]  # pick first if multiple
        row_sorted = np.argsort(AD[:, col])[:2]
        for j in row_sorted:
            node.append([j, col, AD[j, col]])
            Tprf.append(Tlayer[j, col, :])
            Hprf.append(Hlayer[j, col, :])
            Pprf.append(Player[j, col, :])
            rWGprf.append(rWGS[j, col])
            N_coord.append([j, col])
            k += 1

    Tprf = np.array(Tprf).T  # shape: (levels, 4)
    Hprf = np.array(Hprf).T
    Pprf = np.array(Pprf).T
    rWGprf = np.array(rWGprf)  # shape: (4,)
    N_coord = np.array(N_coord)

    # Weights: inverse squared distance
    wNode = np.array([n[2] for n in node])**-2

    geomRay = llRay[2] + 20

    # Layer above/below ray
    dgeom = geomRay - hlayers
    nodev = [np.max(np.where(dgeom >= 0)[0]), np.min(np.where(dgeom < 0)[0])]

    # Interpolate for 4 nodes and 2 layers
    Tprf2 = Tprf[nodev, :]
    Hprf2 = Hprf[nodev, :]
    Pprf2 = Pprf[nodev, :]

    if geomRay < 181:
        # Water vapor pressure
        Ce = (hlayers[nodev[1]] - hlayers[nodev[0]]) / np.log(Hprf2[1,:] / Hprf2[0,:])
        eh1 = Hprf2[0,:] * np.exp((geomRay - hlayers[nodev[0]]) / Ce)
        eh2 = Hprf2[1,:] * np.exp((geomRay - hlayers[nodev[1]]) / Ce)
        eh = np.sum((eh1 + eh2)/2 * wNode) / np.sum(wNode)
        if np.isnan(eh) and (np.sum(Hprf2[1,:]) == 0 or np.sum(Hprf2[0,:]) == 0):
            eh = 0

        # Temperature
        Ctemp = (Tprf2[1,:] - Tprf2[0,:]) / (hlayers[nodev[1]] - hlayers[nodev[0]])
        temph1 = Tprf2[0,:] + (geomRay - hlayers[nodev[0]]) * Ctemp
        temph2 = Tprf2[1,:] + (geomRay - hlayers[nodev[1]]) * Ctemp
        temph = np.sum((temph1 + temph2)/2 * wNode) / np.sum(wNode)

        # Pressure
        presh = np.zeros(2)
        gm = np.zeros(2)
        Tv = np.zeros((2,4))
        presh_nodes = np.zeros((2,4))
        for i in range(2):
            gm[i] = g0*(1 - 0.0026373*np.cos(2*llRay[1]*np.pi/180) + 5.9e-6*np.cos(2*llRay[1]*np.pi/180))**2 * (1 / (1 + (geomRay/1000)/rWGprf)**2)
            Tv[i,:] = Tprf2[i,:]*Pprf2[i,:]/(Pprf2[i,:] - (1 - Mw/Md)*Hprf2[i,:])
            presh_nodes[i,:] = Pprf2[i,:] * np.exp(-gm[i]*(geomRay - hlayers[nodev[i]])/(Rd*Tv[i,:]))
        presh = np.sum(np.mean(presh_nodes, axis=0) * wNode) / np.sum(wNode)
    else:
        presh = np.sum((Pprf2[0,:] + Pprf2[1,:])/2 * wNode) / np.sum(wNode)
        temph = np.sum((Tprf2[0,:] + Tprf2[1,:])/2 * wNode) / np.sum(wNode)
        eh = np.sum((Hprf2[0,:] + Hprf2[1,:])/2 * wNode) / np.sum(wNode)

    Nh, Nw = refcalc(presh, temph, eh, 'c')

    N = {
        'Nh': Nh,
        'Nw': Nw,
        'Nt': Nh + Nw,
        'T': temph,
        'p': presh,
        'h': eh
    }

    return N