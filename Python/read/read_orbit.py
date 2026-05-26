from pathlib import Path
import numpy as np
import os
import ftplib
from typing import List
from poly import orb2poly, polyval

def readsp3(filename: str):
    """
    Read precise satellite orbit data from an SP3 file.

    Parameters
    ----------
    filename : str
        Path to the SP3 file.

    Returns
    -------
    SP3data : ndarray
        Array with columns [GPSweek*7*24*3600 + GPSsec, PRN, X, Y, Z, clk].
        X, Y, Z are in km, clk in microseconds.
    numsat : int
        Number of satellites in SP3 file (from header).
    header : list[str]
        Lines of the SP3 file header.
    """

    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"SP3 file not found: {filename}")

    SP3data = []
    header = []
    time = None
    noom = 0

    # ---- Read header ----
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("* "):  # start of data
                break
            header.append(line.rstrip("\n"))

    n_header = len(header)

    # ---- Read data ----
    with open(filename, "r") as f:
        # Skip header lines
        for _ in range(n_header):
            f.readline()

        for line in f:
            line = line.strip("\n")

            if len(line) >= 60 and line[0] == "P":
                # Extract PRN and orbit parameters
                prn = int(line[2:4])
                x = float(line[4:18])
                y = float(line[18:32])
                z = float(line[32:46])
                clk = float(line[46:60])

                if time is not None:
                    gps_week, gps_sec = utc2gps(time, 0)
                    SP3data.append([
                        gps_week * 7 * 24 * 3600 + gps_sec,
                        prn, x, y, z, clk
                    ])

            elif len(line) == 31 and line.startswith("*"):
                # Epoch time line
                noom += 1
                # Parse time from SP3 epoch line: "*  yyyy mm dd hh mm ss.ssssssss"
                parts = line[1:].split()
                year, month, day, hour, minute, sec = map(float, parts)
                time = [int(year), int(month), int(day),
                        int(hour), int(minute), sec]

    # Convert to numpy array
    SP3data = np.array(SP3data, dtype=float)

    # Extract numsat from header if available
    numsat = None
    for h in header:
        if h.startswith("+ "):  # Satellite list line
            try:
                numsat = int(h[4:6])
            except ValueError:
                pass
            break

    return SP3data, numsat, header


def utc2gps(time_list, leap_seconds: int = 0):
    """
    Convert UTC calendar date to GPS week and seconds of week.

    Parameters
    ----------
    time_list : list
        [year, month, day, hour, minute, second]
    leap_seconds : int, optional
        Leap second offset, default=0.

    Returns
    -------
    gps_week : int
    gps_sec : float
    """
    import datetime

    dt = datetime.datetime(
        int(time_list[0]),
        int(time_list[1]),
        int(time_list[2]),
        int(time_list[3]),
        int(time_list[4]),
        int(time_list[5])
    )

    gps_epoch = datetime.datetime(1980, 1, 6)
    delta = (dt - gps_epoch).total_seconds() - leap_seconds

    gps_week = int(delta // (7 * 24 * 3600))
    gps_sec = delta % (7 * 24 * 3600)
    return gps_week, gps_sec

def readSP3dat(obs_set, pathORB):
    """
    Reads SP3 orbit data corresponding to the observation set.

    Parameters
    ----------
    obs_set : object
        Observation set structure with attribute 'observation_set_SP3' (numpy array)
    pathORB : str
        Path to the SP3 orbit files

    Returns
    -------
    SP3data : ndarray
        Combined SP3 data from all relevant files
    """
    print("SP3 processing")
    SP3data = []

    # Extract GPS week and day
    obs_SP3 = obs_set
    gps_week_day = np.unique(obs_SP3[:, [3, 5]], axis=0)  

    # Handle next day case
    if obs_SP3 [-1, 5] == 7:
        gps_week_next = int(obs_SP3[-1, 3]+1)
        gps_week_day_s = 0
    else:
        gps_week_next = int(obs_SP3[-1, 3])   
        gps_week_day_s = int(obs_SP3[-1, 5]+1)
    
    gps_week_day = np.vstack([gps_week_day,[gps_week_next, gps_week_day_s]])
    
    # Handle next day case
    if obs_SP3 [0, 5] == 0:
        gps_week_next = int(obs_SP3[0, 3]-1)
        gps_week_day_s = 7
    else:
        gps_week_next = int(obs_SP3[0, 3])   
        gps_week_day_s = int(obs_SP3[0, 5]-1)
    
    gps_week_day = np.vstack([gps_week_day,[gps_week_next, gps_week_day_s]])


    # Loop over GPS week/day combinations
    for gw, gd in gps_week_day:
        filenames = [
            f"igs{int(gw)}{int(gd)}.sp3",
        ]

        file_ORB = None
        for fname in filenames:
            full_path = os.path.join(pathORB, fname)
            if os.path.isfile(full_path):
                file_ORB = full_path
                break

        if file_ORB is not None:
            tempSP3data, _, _ = readsp3(file_ORB)  
            SP3data.append(tempSP3data)

    if SP3data:
        SP3data = np.vstack(SP3data)
    else:
        SP3data = np.array([])

    return SP3data

def download_orb(path_orb: Path, observation_set_SP3: np.ndarray) -> None:
    """
    Download SP3 orbit files from igs.bkg.bund.de FTP server.

    Parameters
    ----------
    path_orb : Path
        Local path where files will be downloaded
    observation_set_SP3 : ndarray
        Observation set matrix containing time [JD, ..., GPS week/day info]
    """

    # Ensure target directory exists
    path_orb.mkdir(parents=True, exist_ok=True)

    # Unique [gps_week, gps_day] pairs
    gps_week_day = np.unique(
        np.column_stack((observation_set_SP3[:, 3], observation_set_SP3[:, 5])),
        axis=0
    )

    # Handle next day case
    if observation_set_SP3[-1, 5] == 7:
        gps_week_next = int(observation_set_SP3[-1, 3]+1)
        gps_week_day_s = 0
    else:
        gps_week_next = int(observation_set_SP3[-1, 3])   
        gps_week_day_s = int(observation_set_SP3[-1, 5]+1)
    
    gps_week_day = np.vstack([gps_week_day,[gps_week_next, gps_week_day_s]])
    
    # Handle next day case
    if observation_set_SP3[0, 5] == 0:
        gps_week_next = int(observation_set_SP3[0, 3]-1)
        gps_week_day_s = 7
    else:
        gps_week_next = int(observation_set_SP3[0, 3])   
        gps_week_day_s = int(observation_set_SP3[0, 5]-1)
    
    gps_week_day = np.vstack([gps_week_day,[gps_week_next, gps_week_day_s]])
    
    # Loop over GPS weeks/days
    for week, day in gps_week_day:
        gps_names: List[str] = [
            f"igs{int(week)}{int(day)}.sp3",
        ]

        for gps_name in gps_names:
            local_file = path_orb / gps_name
            if local_file.exists():
                break  # Already downloaded

            ftp = ftplib.FTP("igs.bkg.bund.de")
            ftp.login()

            remote_dir = f"/IGS/products/orbits/{week}/"
            remote_file = gps_name + ".Z"

            try:
                ftp.cwd(remote_dir)
                file_list = ftp.nlst()

                if remote_file in file_list:
                    local_z_file = local_file.with_suffix(".sp3.Z")
                    with open(local_z_file, "wb") as f:
                        ftp.retrbinary("RETR " + remote_file, f.write)

                    import subprocess
                    subprocess.run(["uncompress", str(local_z_file)], check=True)

                    break
            except Exception as e:
                print(f"Failed to download {remote_file}: {e}")
            finally:
                ftp.quit()
                
def interSP3(SP3data, observation_set):
    """
    Interpolate SP3 satellite positions to specified observation times.

    Parameters
    ----------
    SP3data : ndarray
        SP3 orbit data [time, satellite_id, X, Y, Z, ...]
    observation_set : ndarray
        Observation epochs and data, columns 4 and 5 used for time

    Returns
    -------
    SP3X, SP3Y, SP3Z : ndarrays
        Interpolated coordinates [obs x satellite]
    PRN : ndarray
        Satellite IDs repeated per observation
    observation_set : ndarray
        Possibly trimmed observation set based on SP3 coverage
    """
    # Check SP3 and observation times
    startSP3 = SP3data[:,0].min()
    endSP3 = SP3data[:,0].max()
    startOBS = observation_set[0,3]*7*24*3600 + observation_set[0,4]
    endOBS = observation_set[-1,3]*7*24*3600 + observation_set[-1,4]

    # Time overlap checks
    if startOBS > endSP3:
        print(f"Check your orbits! Starting observation time {startOBS}, SP3 ends at {endSP3}")
        print(f"Difference: {(endSP3 - startOBS)/3600:.2f} hours")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if startSP3 > endOBS:
        print(f"Check your orbits! Observation ends at {endOBS}, SP3 starts at {startSP3}")
        print(f"Difference: {(startSP3 - endOBS)/3600:.2f} hours")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Trim observations based on SP3 coverage
    obs_times = observation_set[:,3]*7*24*3600 + observation_set[:,4]
    if startOBS >= startSP3 and endSP3 < endOBS:
        endOBS = endSP3
        observation_set = observation_set[obs_times <= endOBS]
    elif startSP3 >= startOBS and endOBS < endSP3:
        startOBS = startSP3
        observation_set = observation_set[obs_times >= startOBS]
    elif startSP3 > startOBS and endOBS > endSP3:
        startOBS = startSP3
        endOBS = endSP3
        observation_set = observation_set[(obs_times >= startOBS) & (obs_times <= endOBS)]

    # Prepare satellite IDs and arrays
    satellites = np.unique(SP3data[:,1]).astype(int)
    maxp = satellites.max()
    n_obs = observation_set.shape[0]

    SP3X = np.full((n_obs, maxp), np.nan)
    SP3Y = np.full((n_obs, maxp), np.nan)
    SP3Z = np.full((n_obs, maxp), np.nan)
    PRN = np.tile(np.arange(1, maxp+1), (n_obs, 1))

    # Estimate time resolution
    first_sat = SP3data[SP3data[:,1] == satellites[0]]
    resolutionTIME = np.median(np.diff(first_sat[:,0]))
    # Generate polynomials
    poly, fr, to = orb2poly(SP3data, 16, resolutionTIME*12*3, resolutionTIME*12)

    # Interpolate each satellite for each observation
    for j, sat in enumerate(satellites):
        for i in range(n_obs):
            timeCURRENT = observation_set[i,3]*7*24*3600 + observation_set[i,4]
            test = np.column_stack((fr - timeCURRENT, to - timeCURRENT))
            w = np.where((test[:,0] < 0.5) & (test[:,1] >= 0.5))[0]

            if w.size > 0 and not np.any(np.isnan(poly[sat][w[0]]['X'])):
                SP3X[i, sat-1], _ = polyval(poly[sat][w[0]]['X'], timeCURRENT,
                                      poly[sat][w[0]]['X_s'], poly[sat][w[0]]['X_mu'])
                SP3Y[i, sat-1], _ = polyval(poly[sat][w[0]]['Y'], timeCURRENT,
                                      poly[sat][w[0]]['Y_s'], poly[sat][w[0]]['Y_mu'])
                SP3Z[i, sat-1], _ = polyval(poly[sat][w[0]]['Z'], timeCURRENT,
                                      poly[sat][w[0]]['Z_s'], poly[sat][w[0]]['Z_mu'])
                PRN[i, sat-1] = sat

    return SP3X, SP3Y, SP3Z, PRN, observation_set