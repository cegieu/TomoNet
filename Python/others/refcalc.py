import numpy as np

def refcalc(pres, temp, e, solution='c'):
    """
    Calculate refractivity based on pressure, temperature, and water vapor pressure.
    
    Parameters
    ----------
    pres : float or ndarray
        Total pressure [hPa].
    temp : float or ndarray
        Temperature [K].
    e : float or ndarray
        Water vapor pressure [hPa].
    solution : str
        Calculation method: 'a', 'b', or 'c'.
        
    Returns
    -------
    Nh : float or ndarray
        Hydrostatic refractivity [-].
    Nw : float or ndarray
        Wet refractivity [-].
    """
    
    # Constants
    Rd = 287.058       # gas constant dry air [J/(K*kg)]
    Rw = 461.525       # gas constant wet air [J/(K*kg)]
    
    # Bevis (1994) coefficients
    k1 = 77.689
    k2 = 71.295
    k3 = 375463
    k2p = k2 - k1*Rd/Rw
    
    pres = np.asarray(pres, dtype=float)
    temp = np.asarray(temp, dtype=float)
    e = np.asarray(e, dtype=float)
    
    # Compute refractivities
    if solution == 'a':
        # Use total pressure and k2' coefficient: largest Nh
        Nh = k1 * pres / temp
        Nw = k2p * e / temp + k3 * e / temp**2
        
    elif solution == 'b':
        # Dry and vapor parts separately: smallest Nh, largest Nw
        Nh = k1 * (pres - e) / temp
        Nw = k2 * e / temp + k3 * e / temp**2
        
    elif solution == 'c':
        # Hydrostatic and non-hydrostatic (moderate Nh)
        Dd = (pres - e) / (Rd * temp)
        Dw = e / (Rw * temp)
        D = Dd + Dw
        Nh = k1 * Rd * D
        Nw = k2p * e / temp + k3 * e / temp**2
        
    else:
        raise ValueError("Invalid solution option. Choose 'a', 'b', or 'c'.")
    
    return Nh, Nw