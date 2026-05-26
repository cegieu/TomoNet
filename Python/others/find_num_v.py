import numpy as np
def find_num_v(phi_p, lam_p, phi_v, lam_v):
    """
    Determines in which voxel a point (phi_p, lam_p) is located.
    
    Parameters
    ----------
    phi_p : float
        Latitude of the point [deg]
    lam_p : float
        Longitude of the point [deg]
    phi_v : array-like
        Voxel latitudes [deg], sorted ascending
    lam_v : array-like
        Voxel longitudes [deg], sorted ascending
    
    Returns
    -------
    i_pos : int
        Voxel ID (1D index)
    """
    phi_v = np.array(phi_v)
    lam_v = np.array(lam_v)

    # Compute differences
    dphi = phi_p - phi_v
    dlam = lam_p - lam_v

    # Get index of voxel
    n_lat1 = np.where(dphi >= 0)[0]
    n_lon1 = np.where(dlam >= 0)[0]
    
    n_lat1 = n_lat1[-1] if len(n_lat1) > 0 else 0
    n_lon1 = n_lon1[-1] if len(n_lon1) > 0 else 0

    # Correct index if outside
    if n_lon1 >= len(lam_v)-1:
        n_lon1 = len(lam_v)-2
    if n_lat1 >= len(phi_v)-1:
        n_lat1 = len(phi_v)-2

    # Compute voxel ID (flattened 1D)
    i_pos = n_lon1 + (len(lam_v)-1) * n_lat1
    return i_pos