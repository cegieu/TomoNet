import numpy as np

def cal2time(day):
    """
    Convert fractional day to hours, minutes, and seconds.
    
    Parameters
    ----------
    day : float or ndarray
        Day of month with fractional part.
    
    Returns
    -------
    hh, mm, sec : int or ndarray
        Hours, minutes, seconds corresponding to the fractional day.
    """
    seconds = np.round((day - np.floor(day)) * 24 * 60 * 60).astype(int)
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    sec = seconds % 60
    return hh, mm, sec

def jd2cal(jd):
    """
    Convert Julian date to calendar date (year, month, day with fraction)
    
    Parameters
    ----------
    jd : float or ndarray
        Julian date
    
    Returns
    -------
    yr, mn, dy : int or float or ndarray
        Year, month, day (day may include fraction)
    """
    jd = np.asarray(jd)
    a = np.floor(jd + 0.5).astype(int)
    
    yr = np.zeros_like(jd, dtype=int)
    mn = np.zeros_like(jd, dtype=int)
    dy = np.zeros_like(jd, dtype=float)
    
    mask = a < 2299161
    c = np.where(mask, a + 1524, 0)
    b = np.where(~mask, np.floor((a - 1867216.25) / 36524.25).astype(int), 0)
    c = np.where(~mask, a + b - np.floor(b / 4).astype(int) + 1525, c)
    
    d = np.floor((c - 122.1) / 365.25).astype(int)
    e = np.floor(365.25 * d).astype(int)
    f = np.floor((c - e) / 30.6001).astype(int)
    dy = c - e - np.floor(30.6001 * f) + (jd + 0.5 - a)
    mn = f - 1 - 12 * np.floor(f / 14).astype(int)
    yr = d - 4715 - np.floor((7 + mn) / 10).astype(int)
    
    return yr, mn, dy

def jd2doy(jd):
    """
    Convert Julian date to year and day-of-year
    
    Parameters
    ----------
    jd : float or ndarray
        Julian date
    
    Returns
    -------
    doy : int or ndarray
        Day of year
    yr : int or ndarray
        Year
    """
    yr, mn, dy = jd2cal(jd)
    jd_start = cal2jd(yr, 1, 1)  # JD at the start of the year
    doy = jd - jd_start
    return doy, yr

def jd2gps2(jd):
    """
    Convert Julian Date to year, day-of-year, and seconds of day
    
    Parameters
    ----------
    jd : float or ndarray
        Julian date
    
    Returns
    -------
    yr : int or ndarray
        Year
    doy : int or ndarray
        Day of the year
    sec : float or ndarray
        Seconds of the day
    """
    doy_float, yr = jd2doy(jd)
    doy = np.floor(doy_float).astype(int)
    yr2, mm, day_frac = jd2cal(jd)
    hh, mm_time, ss = cal2time(day_frac)
    sec = ss + mm_time * 60 + hh * 3600
    return yr, doy, sec

def cal2jd(yr: int, mn: int, dy: float) -> float:
    """
    Converts calendar date to Julian date using algorithm
    from "Practical Ephemeris Calculations" by Oliver Montenbruck
    (Springer-Verlag, 1989).
    
    Astronomical year numbering is used for BC dates (2 BC = -1).
    
    Parameters
    ----------
    yr : int
        Calendar year (4-digit including century, negative for BC).
    mn : int
        Calendar month (1–12).
    dy : float
        Calendar day (can include fractional day).
    
    Returns
    -------
    jd : float
        Julian date.
    """

    # --- input checks ---
    if not (1 <= mn <= 12):
        raise ValueError("Invalid input month")

    if dy < 1:
        raise ValueError("Invalid input day")

    if (mn == 2 and dy > 29) or (mn in [3, 5, 9, 11] and dy > 30) or (dy > 31):
        raise ValueError("Invalid input day")

    # --- month/year adjustment ---
    if mn > 2:
        y = yr
        m = mn
    else:
        y = yr - 1
        m = mn + 12

    # --- Gregorian calendar reform (1582) ---
    date1 = 4 + 31 * (10 + 12 * 1582)    # Last Julian date
    date2 = 15 + 31 * (10 + 12 * 1582)   # First Gregorian date
    date = dy + 31 * (mn + 12 * yr)

    if date <= date1:
        b = -2
    elif date >= date2:
        b = (y // 400) - (y // 100)
    else:
        raise ValueError("Dates between October 5 and 15, 1582 do not exist")

    # --- Julian date calculation ---
    if y > 0:
        jd = (int(365.25 * y) +
              int(30.6001 * (m + 1)) +
              b + 1720996.5 + dy)
    else:
        jd = (int(365.25 * y - 0.75) +
              int(30.6001 * (m + 1)) +
              b + 1720996.5 + dy)

    return jd