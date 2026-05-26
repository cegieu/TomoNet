import numpy as np
import os
import glob
import pandas as pd
from datetime import datetime


def readtxtOBS(pathATM, NAME, observation_set):
    """
    Read ZTD and gradients from GNSS observation files.
    
    Parameters
    ----------
    pathATM : str
        Path to GNSS observation files.
    NAME : list of str
        List of GNSS station names.
    observation_set : ndarray
        Observation epochs [YYYY, DOY, ...].
    
    Returns
    -------
    ZTDA : ndarray
        Zenith Total Delay values.
    MZTDA : ndarray
        Uncertainties of ZTD.
    DGNA : ndarray
        North gradient of ZTD.
    MDGNA : ndarray
        Uncertainty of north gradient.
    DGEA : ndarray
        East gradient of ZTD.
    MDGEA : ndarray
        Uncertainty of east gradient.
    NAMES : list of str
        Available GNSS station names.
    test : ndarray
        Boolean mask for stations with available observations.
    """
    
    # Find files
    filelist = sorted(glob.glob(os.path.join(pathATM, '*')))
    epoch = -1
    if len(filelist) == 0:
        print("No matching files were found")
        return ([], [], [], [], [], [], [], [])

    # Determine first matching file by date
    A = datetime(int(observation_set[0, 2]), int(observation_set[0, 6]), int(observation_set[0, 7]))
    s = None
    for idx, filepath in enumerate(filelist):
        filename = os.path.basename(filepath)
        year = int(filename[0:4])
        month = int(filename[4:6])
        day = int(filename[6:8])
        B = datetime(year, month, day)
        if A == B:
            s = idx
            break

    if s is None:
        print("No matching files for the first observation epoch")
        return ([], [], [], [], [], [], [], [])

    # Determine number of days
    if observation_set.shape[0] > 24:
        days = observation_set.shape[0] // 24
    else:
        days = 1

    # Initialize output arrays
    ZTDA, MZTDA, DGNA, DGEA, MDGNA, MDGEA = [], [], [], [], [], []
    
    
    for ep in range(days):
        filepath = filelist[ep]
        # Read tab-delimited data
        try:
            # Read CSV without headers
            df = pd.read_csv(filepath, delimiter='\t', dtype=str)

            # Assign proper column names
            columnsnum = ['B','L','Helips','Hnorm','ZTD', 'mZTD', 'GradN', 'GradE', 'mGradN', 'mGradE']
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")

        # Convert relevant columns to numeric
        for col in columnsnum:
            df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')



        # Filter rows by NAME
        df = df[df['ID'].isin(NAME)]
        df = df.sort_values(by='Date')

        for k in range(24):  # hours
            epoch = epoch + 1
            epoch_dt = datetime(
                int(observation_set[epoch, 2]),  # year
                int(observation_set[epoch, 6]),  # month
                int(observation_set[epoch, 7]),  # day
                int(observation_set[epoch, 8])   # hour
            )
            # Match the epoch
            try:
                df_epoch = df[pd.to_datetime(df['Date'], errors='coerce') == epoch_dt]
            except Exception:
                df_epoch = pd.DataFrame(columns=df.columns)

            # Map station names
            ZTD = np.full(len(NAME), np.nan)
            MZTD = np.full(len(NAME), np.nan)
            DGN = np.full(len(NAME), np.nan)
            DGE = np.full(len(NAME), np.nan)
            MDGN = np.full(len(NAME), np.nan)
            MDGE = np.full(len(NAME), np.nan)

            for i, name in enumerate(NAME):
                row = df_epoch[df_epoch['ID'] == name]
                if not row.empty:
                    ZTD[i] = row['ZTD'].values[0]
                    MZTD[i] = row['mZTD'].values[0]
                    DGN[i] = row['GradN'].values[0]
                    DGE[i] = row['GradE'].values[0]
                    MDGN[i] = row['mGradN'].values[0]
                    MDGE[i] = row['mGradE'].values[0]

            ZTDA.append(ZTD)
            MZTDA.append(MZTD)
            DGNA.append(DGN)
            DGEA.append(DGE)
            MDGNA.append(MDGN)
            MDGEA.append(MDGE)

    # Convert lists to arrays
    ZTDA = np.vstack(ZTDA)
    MZTDA = np.vstack(MZTDA)
    DGNA = np.vstack(DGNA)
    DGEA = np.vstack(DGEA)
    MDGNA = np.vstack(MDGNA)
    MDGEA = np.vstack(MDGEA)
    NAMES = list(df['ID'].unique())
    test = np.array([name in NAMES for name in NAME])

    # Filter missing stations
    if len(NAMES) != len(NAME):
        ZTDA = ZTDA[:, test]
        MZTDA = MZTDA[:, test]
        DGNA = DGNA[:, test]
        MDGNA = MDGNA[:, test]
        DGEA = DGEA[:, test]
        MDGEA = MDGEA[:, test]
        print("Warning: Missing observational total delays for some GNSS stations. Removed from processing.")

    return ZTDA, MZTDA, DGNA, MDGNA, DGEA, MDGEA, NAMES, test


def screen_ztd(ZTDA, MZTDA, DGNA, MDGNA, DGEA, MDGEA, miss_stat, model):
    """
    Screen ZTD and related matrices for outliers and NaN values.

    Parameters
    ----------
    ZTDA : ndarray
        Matrix of zenith total delays.
    MZTDA : ndarray
        Matrix of uncertainties of zenith total delays.
    DGNA, MDGNA, DGEA, MDGEA : ndarray
        Matrices of N/E gradients and their uncertainties.
    miss_stat : list or array-like
        Indices of stations with available observations.
    model : dict-like
        Parameters of tomography and ray tracing models. Must contain 'BLh', 'BLH', 'NAME'.

    Returns
    -------
    ZTDA, MZTDA, DGNA, MDGNA, DGEA, MDGEA : ndarray
        Filtered matrices with outliers set to NaN.
    model : dict-like
        Updated model containing only stations in miss_stat.
    """

    # Find non-NaN indices in MZTDA
    nonan = ~np.isnan(MZTDA)
    
    # Mean of non-NaN MZTDA values
    mean_MZTDA = np.nanmean(MZTDA)
    
    # Identify outliers (> 1.5 * mean)
    outliers = MZTDA > 1.5 * mean_MZTDA
    
    # Set outliers to NaN
    ZTDA[outliers] = np.nan
    MZTDA[outliers] = np.nan
    DGNA[outliers] = np.nan
    MDGNA[outliers] = np.nan
    DGEA[outliers] = np.nan
    MDGEA[outliers] = np.nan
    
    # Update model to include only stations in miss_stat
    model['BLh'] = model['BLh'][miss_stat, :]
    model['BLH'] = model['BLH'][miss_stat, :]
    model['Pstat'] = model['Pstat'][:,miss_stat]
    model['Tstat'] = model['Tstat'][:,miss_stat]
    model['NAME'] = model['NAME'][miss_stat]
    ZHD_dict = model["ZHD"]

    n_epochs = len(ZHD_dict)
    n_stations = len(next(iter(ZHD_dict.values())))

    ZHD_matrix = np.empty((n_epochs, n_stations))
    ZHD_matrix[:] = np.nan  # optional initialization
    
    for epoch, key in enumerate(sorted(ZHD_dict.keys())):   
        ZHD_matrix[epoch, :] = ZHD_dict[key].flatten()
    model["ZHD"] = ZHD_matrix[:,miss_stat]
    
    return ZTDA, MZTDA, DGNA, MDGNA, DGEA, MDGEA, model