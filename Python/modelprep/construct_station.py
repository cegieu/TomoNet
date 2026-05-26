import numpy as np
from math import pi, cos, sin
from typing import Any
from gpt2 import gpt2
import construct_station_func as csf
from ztd2iwv import ztd2iwv


def construct_station(
    BLh: np.ndarray,
    BLH: np.ndarray,
    ZTD: np.ndarray,
    M_ZTD: np.ndarray,
    ZHD: np.ndarray,
    NAME: list[str],
    PRN: np.ndarray,
    SP3X: np.ndarray,
    SP3Y: np.ndarray,
    SP3Z: np.ndarray,
    Pstat,
    Tstat,
    observation_set: np.ndarray,
    cut_off_angle: float,
    pathgrd,
    grdmodel,
    switches: dict,
    *args: Any
) -> list[dict]:
    """
    Convert ZTD data into station structural dictionary for tomography model.

    Parameters
    ----------
    BLh : ndarray
        Station coordinates (id, lat, lon, h, ...)
    BLH : ndarray
        Station coordinates (alternative format)
    ZTD, M_ZTD, ZHD : ndarray
        Zenith delays and errors
    NAME : list[str]
        Station names
    PRN : ndarray
        Satellite PRN numbers [doy, sat]
    SP3X, SP3Y, SP3Z : ndarray
        Satellite XYZ positions (km)
    observation_set : ndarray
        Observation metadata [?, doy, year]
    cut_off_angle : float
        Elevation cutoff [deg]
    switches : dict
        Configuration switches
    *args : optional
        If solution=="REAL", can be:
          (DNA, DEA, MDNA, MDEA) or (DNA, DEA, MDNA, MDEA, GdN, GdE)

    Returns
    -------
    station : list[dict]
        station[t]["h"][s] contains station/satellite data
    """

    station: list[dict] = []
    idx: list[int] = []

    if switches["solution"]== ["REAL"]:
        if len(args) == 4:
            DNA, DEA, MDNA, MDEA = args
            GdN = GdE = None
        elif len(args) == 6:
            DNA, DEA, MDNA, MDEA, GdN, GdE = args
        else:
            DNA = DEA = MDNA = MDEA = GdN = GdE = None

    # loop epochs
    for t in range(SP3X.shape[0]):
        station_t = {"h": []}

        for s in range(len(BLh)):
            h_dict: dict[str, Any] = {"satellite": []}

            if switches.get("solution") == ["REAL"]:
                ztd_val = ZTD[t, s] / 1000
                dna_val = DNA[t, s] / 1000 if DNA is not None else np.nan
                dea_val = DEA[t, s] / 1000 if DEA is not None else np.nan

                h_dict["ZTD"] = [ztd_val, dna_val, dea_val]

                if any(np.isnan(h_dict["ZTD"])):
                    idx.append(s)

            # loop satellites
            b = 0
            for nr in range(PRN.shape[1]):
                doy_num = int(np.floor(observation_set[0, 1] - np.floor(observation_set[0, 1]))) + 1
                sat = PRN[doy_num - 1, nr]

                # receiver XYZ (m) vs satellite XYZ (km → m)
                lat_s = float(np.asarray(BLh[s, 1]).squeeze())
                lon_s = float(np.asarray(BLh[s, 2]).squeeze())
                h_s = float(np.asarray(BLh[s, 3]).squeeze())

                X, Y, Z = csf.BLH2XYZ(lat_s * pi / 180, lon_s * pi / 180, h_s)
                sat_xyz = np.array([SP3X[t, nr], SP3Y[t, nr], SP3Z[t, nr]], dtype=float) * 1000.0
                rec_xyz = np.array([X, Y, Z], dtype=float).squeeze()
                elev, azi = csf.xyzSP32elaz(rec_xyz, sat_xyz)

                elev = elev * 180 / pi
                azi = azi * 180 / pi

                # filter by cutoff
                if elev > cut_off_angle:
                    b += 1

                    # VMF1 with GPT2
                    mjdday = csf.doy2jd(int(observation_set[t, 2]), int(np.floor(observation_set[t, 1])))
                    mjdday = mjdday-2400000.5;

                    # for first call: load mapping function coefficients
                    if t == 0 and s == 0:
                        ah = aw = None
                        mjdday_old = 0
                    if mjdday > mjdday_old:
                        _, _, _, _, ah, aw, _,grdmodel = gpt2(mjdday, BLh[s, 1] * pi / 180, BLh[s, 2] * pi / 180, BLh[s, 3], 1, 0,0,pathgrd,grdmodel)
                        mjdday_old = mjdday

                    vmf1h, vmf1w = csf.vmf1_ht(ah, aw, mjdday, BLh[s, 1] * pi / 180, BLh[s, 3], (pi / 2 - elev * pi / 180))

                    sat_dict = {
                        "PRN": sat,
                        "elevation": elev,
                        "azi": azi,
                        "vmf1h": vmf1h,
                        "vmf1w": vmf1w,
                    }

                    if switches["solution"] == ["REAL"]:
                        zhd = ZHD[t][s]
                        std = vmf1w * (ZTD[t, s] / 1000 - zhd) + vmf1h * zhd + 1./np.tan(elev * pi / 180) * (
                            dna_val * cos(azi * pi / 180) + dea_val * sin(azi * pi / 180)
                        )
                        swd = vmf1w * (ZTD[t, s] / 1000 - zhd) + 1./np.tan(elev * pi / 180) * (
                            (dna_val - (GdN[t, s] / 1000 if GdN is not None else 0)) * cos(azi * pi / 180)
                            + (dea_val - (GdE[t, s] / 1000 if GdE is not None else 0)) * sin(azi * pi / 180)
                        )
                        m_swd = vmf1w * M_ZTD[t, s] / 1000
                        m_std = m_swd + 0.2 * m_swd
                        iwv, siwv = ztd2iwv(ZTD[t,s]/1000, Pstat[t,s], Tstat[t,s], BLh[s,1], BLh[s,3], vmf1w)
                        siwv = siwv/1000
                        m_siwv = np.cos(elev*np.pi/180)*siwv/1000
                        sat_dict.update(
                            {
                                "SWD": swd,
                                "M_SWD": m_swd,
                                "STD": std,
                                "M_STD": m_std,
                                "SIWV": siwv,
                                "M_SIWV": m_siwv,
                            }
                        )

                    h_dict["parameters"] = [*BLh[s, 0:3], BLh[s, 3], BLH[s, 3]]
                    h_dict["name"] = NAME[s]
                    h_dict["satellite"].append(sat_dict)

            station_t["h"].append(h_dict)

        # remove invalid stations
        if idx:
            station_t["h"] = [h for j, h in enumerate(station_t["h"]) if j not in idx]
        idx = []

        station.append(station_t)

    return station