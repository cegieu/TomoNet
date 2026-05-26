#!/usr/bin/env python3
"""
TomoNet v.1.1

Designed by Adam Cegla at ETH Zurich, Chair of Space Geodesy.
15.05.2026

This software is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users should
refer to the GNU General Public License for more details.
"""

import numpy as np
import time
import scipy.io as sio
from pathlib import Path
import voxel_dist as vd
import voxel_dist_2D as vd2

def raytracing(obs, model, epoch, paths, switches):
    """
    Ray tracing function to build design matrix A and observation vectors.
    
    Parameters
    ----------
    obs : list of dicts
        GNSS observation data per epoch.
    model : object
        Tomography and ray tracing model parameters.
        Attributes needed: num_lon_TOMO, num_lat_TOMO, num_levels_TOMO, LAT, LON,
        h_RT, BLh, refrNw, refrN
    epoch : int
        Current epoch index.
    paths : object
        Attributes: pathTOMO
    switches : object
        Attributes: solution (string, 'REAL' or other)
        
    Returns
    -------
    A : ndarray
        Design matrix
    SD : list
        Observed slant delays
    SDtest : list
        Calculated slant delays for testing
    elev : list
        Elevation angles
    """
    save_filename = Path(paths['pathTOMO']) / f'amtrix_{epoch+1}.mat'
    if not save_filename.exists():
        start_time = time.time()
        
        n = model['num_lon_TOMO'] * model['num_lat_TOMO'] * model['num_levels_TOMO']
        
        # Initialize variables
        A = np.zeros((1, n))
        SD = []
        elev = []
        SDtest = []
        n_SD = 0
        refNw = model['refrNw'][epoch+1]
        for r, h_obs in enumerate(obs[epoch]['h']):
            if switches['solution'] == ['REAL']:
                SD_sel = np.array([sat['SWD'] for sat in h_obs['satellite']])
            
            los_r = np.array([
                [sat['azi'] for sat in h_obs['satellite']],
                [sat['elevation'] for sat in h_obs['satellite']]
            ])
            
            n_obs = los_r.shape[1]
            for s in range(n_obs):
                if not np.isnan(los_r[1, s]):
                    # Choose ray tracer depending on elevation
                    if los_r[1, s] > 10:
                        ray = vd.voxel_dist(
                            model['LAT'][0,:], model['LON'][0,:], model['h_RT'],
                            model['BLh'][r,1], model['BLh'][r,2], model['BLh'][r,3],
                            los_r[1,s], los_r[0,s]
                        )
                    else:
                        ray = vd2.voxel_dist_2D(
                            model['refrN'][epoch+1] - refNw, refNw,
                            model['LAT'][0,:], model['LON'][0,:], model['h_RT'].T,
                            model['BLh'][r,1], model['BLh'][r,2], model['BLh'][r,3],
                            los_r[1,s], los_r[0,s]
                        )
                    
                    refvox = []
                    for i in range(len(ray['d_voxel'])):
                        temp_num = ray['n_voxel'][i];
                        num_v = temp_num[0] + (model['num_lat_TOMO']*model['num_lon_TOMO'])*(temp_num[1])
                        if A.shape[0] <= n_SD:
                            A = np.vstack([A, np.zeros((1, n))])
                        A[n_SD, num_v] = ray['d_voxel'][i]/1000
                        refvox.append(refNw[temp_num[1], temp_num[0]])
                    
                    if switches['solution'] == ['REAL']:
                        SD.append(SD_sel[s])
                        SDtest.append(np.dot(refvox, np.array(ray['d_voxel'])/1000)/1000)

                    else:
                        SD.append(np.dot(refvox, np.array(ray['d_voxel'])/1000)/1000)
                    elev.append(los_r[1,s])
                    n_SD += 1
                    
        sio.savemat(save_filename, {'A': A, 'elev': elev, 'SD': SD, 'SDtest': SDtest})
        end_time = time.time()
        print(f"Epoch calculated: {epoch} (took {end_time - start_time:.2f} s)")
    else:
        data = sio.loadmat(save_filename)
        A = data['A']
        elev = data['elev']
        SD = data['SD']
        SDtest = data['SDtest']

    return A, SD, SDtest, elev