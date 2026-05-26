import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ObsSet:
    observation_set: np.ndarray
    observation_set_SP3: np.ndarray
    observation_set_NWP: np.ndarray = None
    interpolation_set: np.ndarray = None

def find_epochs(switches, observation_start, observation_end,
                est_interval_TOMO, obs_interval_SP3,
                obs_interval_ZTD, obs_interval_NWP,
                int_interval_METEO):
    """
    Generates matrices with epochs for tomography, SP3, NWP, and interpolation
    """
    # Convert start and end times to datetime
    t_start = datetime(*observation_start[:6])
    t_end = datetime(*observation_end[:6])

    # TOMO epochs
    observation_set = generate_epoch_matrix(t_start, t_end, est_interval_TOMO)

    # SP3 epochs
    observation_set_SP3 = generate_epoch_matrix(t_start, t_end, obs_interval_SP3)

    # Interpolation epochs
    interpolation_set = generate_epoch_matrix(t_start, t_end, int_interval_METEO)

    # NWP epochs
    observation_set_NWP = None
    if switches.get('aprModel') == 'WRF':
        observation_set_NWP = generate_epoch_matrix(t_start, t_end, obs_interval_NWP)
        # Mark forecast hours 0,6,12,18
        hours = observation_set_NWP[:, 8]
        minutes = observation_set_NWP[:, 9]
        mark = ((hours == 0) | (hours == 6) | (hours == 12) | (hours == 18)) & (minutes == 0)
        observation_set_NWP[mark, 10] = 1

    # Mark SP3 at 00:00
    hours_sp3 = observation_set_SP3[:, 8]
    minutes_sp3 = observation_set_SP3[:, 9]
    observation_set_SP3[(hours_sp3 == 0) & (minutes_sp3 == 0), 10] = 1

    return ObsSet(
        observation_set=observation_set,
        observation_set_SP3=observation_set_SP3,
        observation_set_NWP=observation_set_NWP,
        interpolation_set=interpolation_set
    )


def generate_epoch_matrix(t_start: datetime, t_end: datetime, interval_seconds: float) -> np.ndarray:
    """
    Create a matrix of epochs with calendar/GPS info
    """
    # Number of intervals
    total_seconds = (t_end - t_start).total_seconds()
    n = int(total_seconds // interval_seconds) + 1

    # Initialize matrix: 11 columns as in original code
    jdmat = np.zeros((n, 11))

    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    for i in range(n):
        t = t_start + timedelta(seconds=i*interval_seconds)
        # Julian Date
        jdmat[i, 0] = julian_date(t)
        # Day of year
        jdmat[i, 1] = t.timetuple().tm_yday
        # Year
        jdmat[i, 2] = t.year
        # GPS week and seconds of week
        delta_gps = (t - gps_epoch).total_seconds()
        jdmat[i, 3] = np.floor(delta_gps // (7 * 86400))  # GPS week
        jdmat[i, 4] = delta_gps % (7 * 86400)            # Seconds since week start
        # GPS weekday (Sunday=0)
        jdmat[i, 5] = (t.weekday() + 1) % 7
        # Month, Day, Hour, Minute
        jdmat[i, 6] = t.month
        jdmat[i, 7] = t.day
        jdmat[i, 8] = t.hour
        jdmat[i, 9] = t.minute
        jdmat[i, 10] = 0

    return jdmat


def julian_date(dt: datetime) -> float:
    """
    Convert a datetime object to Julian Date.
    """
    year = dt.year
    month = dt.month
    day = dt.day + dt.hour/24 + dt.minute/1440 + dt.second/86400

    if month <= 2:
        year -= 1
        month += 12

    A = np.floor(year / 100)
    B = 2 - A + np.floor(A / 4)

    JD = np.floor(365.25 * (year + 4716)) + np.floor(30.6001 * (month + 1)) + day + B - 1524.5
    return JD