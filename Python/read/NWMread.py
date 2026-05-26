#!/usr/bin/env python3
"""
TomoNet v.1.1

Designed by Adam Cegla at ETH Zurich, Chair of Space Geodesy.
15.05.2026

This software is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users should
refer to the GNU General Public License for more details.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import warnings
import numpy as np

import NWMgrid as NWMg
from timerecalc import cal2jd
from gridcalc import gridcalc
from gridcalc2 import gridcalc2
from pBLh2ZHDRT import pBLh2ZHDRT
from refcalc import refcalc

try:
    from netCDF4 import Dataset
except Exception:
    Dataset = None


def _switch(value, default=None):
    if value is None:
        return default
    if isinstance(value, list):
        return value[0]
    return value

def _check_era_domain(lat_var, lon_var, model, ERA_path):
    lat_min, lat_max = float(np.min(lat_var)), float(np.max(lat_var))
    lon_min, lon_max = float(np.min(lon_var)), float(np.max(lon_var))

    tomo_lat_min = float(np.min(model["lat_TOMO"]))
    tomo_lat_max = float(np.max(model["lat_TOMO"]))
    tomo_lon_min = float(np.min(model["lon_TOMO"]))
    tomo_lon_max = float(np.max(model["lon_TOMO"]))

    errors = []

    if tomo_lat_min < lat_min or tomo_lat_max > lat_max:
        errors.append(
            f"Latitude mismatch: TOMO lat range [{tomo_lat_min}, {tomo_lat_max}] "
            f"but ERA5 file covers [{lat_min}, {lat_max}]"
        )

    if tomo_lon_min < lon_min or tomo_lon_max > lon_max:
        errors.append(
            f"Longitude mismatch: TOMO lon range [{tomo_lon_min}, {tomo_lon_max}] "
            f"but ERA5 file covers [{lon_min}, {lon_max}]"
        )

    if errors:
        raise ValueError(
            "\n[ERROR] ERA5 file does not cover tomography domain.\n"
            f"ERA5 file: {ERA_path}\n"
            + "\n".join(errors)
            + "\nPlease use an ERA5 file covering the full tomography domain "
              "or reduce model['lat_TOMO']/model['lon_TOMO'] in tomo_conf.py."
        )


def _check_grid_shapes(LAT, LON, undugrid_sub, pGrid_sub, tempGrid_sub, Spechum_sub, Geoph_sub):
    shapes = {
        "LAT": LAT.shape,
        "LON": LON.shape,
        "undugrid_sub": undugrid_sub.shape,
        "pGrid_sub horizontal": pGrid_sub.shape[:2],
        "tempGrid_sub horizontal": tempGrid_sub.shape[:2],
        "Spechum_sub horizontal": Spechum_sub.shape[:2],
        "Geoph_sub horizontal": Geoph_sub.shape[:2],
    }

    if LAT.shape != LON.shape:
        raise ValueError(f"LAT/LON shape mismatch: {shapes}")

    if LAT.shape != pGrid_sub.shape[:2]:
        raise ValueError(
            "\n[ERROR] ERA5 grid shape mismatch before gridcalc/gridcalc2.\n"
            f"Shapes: {shapes}\n"
            "This usually means the ERA5 file grid/resolution does not match "
            "model['GRIDboundaries'] or the selected TOMO domain."
        )

    if undugrid_sub.shape != LAT.shape:
        raise ValueError(
            "\n[ERROR] Undulation grid shape mismatch.\n"
            f"Shapes: {shapes}\n"
            "Delete the cached undulation .npy file or regenerate it for the "
            "current TOMO domain/resolution."
        )


def NWMread(
    ERAname: str,
    model: Dict[str, Any],
    tempgrid: Dict[str, Any],
    boundRT: np.ndarray,
    date0: str,
    pathERA: str,
    pathCONF: str,
    obs_start: np.ndarray,
    epoch: int,
    unduFile: str,
    switches: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    if Dataset is None:
        raise RuntimeError("netCDF4 is not available. Install with: pip install netCDF4")

    pathCONF_p = Path(pathCONF)
    pathERA_p = Path(pathERA)
    ERA_path = pathERA_p / ERAname

    if not ERA_path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {ERA_path}")

    globalepoch = int(epoch)

    required_model_keys = [
        "lat_TOMO", "lon_TOMO", "levels_TOMO",
        "GRIDboundaries"
    ]
    for key in required_model_keys:
        if key not in model:
            raise KeyError(f"Missing model['{key}'] required by NWMread")

    boundRT = np.asarray(boundRT, dtype=float).ravel()
    if boundRT.size < 6:
        raise ValueError(
            "boundRT must contain at least 6 values: "
            "[lon_min, lon_max, resolution, lat_min, lat_max, resolution]"
        )

    # ------------------------------------------------------------------
    # BLh pudel / tomography grid structure
    # ------------------------------------------------------------------
    if globalepoch == 1:
        BLHstruc = NWMg.struc(
            model,
            model["lat_TOMO"],
            model["lon_TOMO"],
            model["levels_TOMO"],
            switches,
        )
        tempgrid["BLHstruc"] = BLHstruc
    else:
        BLHstruc = tempgrid.get("BLHstruc", None)

    # ------------------------------------------------------------------
    # Undulation grid
    # ------------------------------------------------------------------
    if isinstance(unduFile, list):
        undu_name = unduFile[0]
    else:
        undu_name = unduFile

    undu_path = pathCONF_p / undu_name
    if undu_path.suffix != ".npy":
        undu_npy_path = undu_path.with_suffix(".npy")
    else:
        undu_npy_path = undu_path

    if not undu_npy_path.exists():
        try:
            undugrid = NWMg.undu(
                boundRT[3], boundRT[4],
                boundRT[0], boundRT[1],
                boundRT[2],
                pathCONF,
                "unduera5",
            )
        except Exception as exc:
            warnings.warn(
                f"Could not compute undulation grid with NWMg.undu; "
                f"falling back to GPT2 undulation. Original error: {exc}"
            )

            obs_start_arr = np.asarray(obs_start).ravel()
            if obs_start_arr.size < 6:
                raise ValueError(
                    "obs_start must contain at least 6 values: "
                    "year, month, day, hour, minute, second"
                )

            start_time = (
                obs_start_arr[3] / 24.0
                + obs_start_arr[4] / 60.0 / 24.0
                + obs_start_arr[5] / 3600.0 / 24.0
            )

            jd_start = cal2jd(
                int(obs_start_arr[0]),
                int(obs_start_arr[1]),
                int(obs_start_arr[2]) + start_time,
            )

            _, _, undugrid = NWMg.distr_T_gpt2RT(
                BLHstruc,
                jd_start - 2400000.5,
            )

            undugrid = np.asarray(undugrid).reshape(
                len(model["lat_TOMO"]),
                len(model["lon_TOMO"]),
                len(model["levels_TOMO"]),
            )

            undugrid = np.flip(np.squeeze(undugrid[:, :, 0]).T, axis=1)

        np.save(undu_npy_path, undugrid)

    else:
        undugrid = np.load(undu_npy_path)

    undugrid = np.asarray(undugrid, dtype=float)

    # ------------------------------------------------------------------
    # RT grid boundaries
    # ------------------------------------------------------------------
    lam_N = np.arange(boundRT[0], boundRT[1] + 1e-12, boundRT[2])
    phi_N = np.arange(boundRT[3], boundRT[4] + 1e-12, boundRT[2])

    if lam_N.size == 0 or phi_N.size == 0:
        raise ValueError("lam_N or phi_N is empty. Check boundRT values.")

    g0 = 9.80665
    hpre = np.asarray(model["levels_TOMO"], dtype=float).ravel() / 1000.0

    # ------------------------------------------------------------------
    # ERA5 epoch index
    # ------------------------------------------------------------------
    dt64 = np.datetime64(date0, "s")
    dt = dt64.astype("O")
    date_array = np.array(
        [[dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]],
        dtype=float,
    )

    eNwp, *_ = NWMg.fera5Epoch(date_array)
    ueNwp = np.unique(eNwp, axis=0)
    era_epoch = int(ueNwp[0, 4])

    if era_epoch < 1:
        raise ValueError(f"Invalid ERA5 epoch index: {era_epoch}")

    time_index = era_epoch - 1

    # ------------------------------------------------------------------
    # Read ERA5 file
    # ------------------------------------------------------------------
    with Dataset(str(ERA_path), "r") as nc:
        required_vars = ["latitude", "longitude", "q", "t", "z"]
        for var in required_vars:
            if var not in nc.variables:
                raise KeyError(f"ERA5 file missing variable '{var}': {ERA_path}")

        lat_var = np.asarray(nc.variables["latitude"][:], dtype=float)
        lon_var = np.asarray(nc.variables["longitude"][:], dtype=float)

        _check_era_domain(lat_var, lon_var, model, ERA_path)

        q_var = nc.variables["q"]
        t_var = nc.variables["t"]
        z_var = nc.variables["z"]

        if time_index >= q_var.shape[0]:
            raise IndexError(
                f"ERA5 time index {time_index} out of range for q variable "
                f"with shape {q_var.shape}"
            )

        Spechum = np.asarray(q_var[time_index, :, :, :], dtype=float)
        tempGrid = np.asarray(t_var[time_index, :, :, :], dtype=float)
        Geop = np.asarray(z_var[time_index, :, :, :], dtype=float)

        if "level" in nc.variables:
            iso = np.asarray(nc.variables["level"][:], dtype=float)

            # MATLAB:
            # Spechum = flip(Spechum,3)
            # tempGrid = flip(tempGrid,3)
            # iso = flip(iso)
            Spechum = np.flip(Spechum, axis=0)
            tempGrid = np.flip(tempGrid, axis=0)
            iso = np.flip(iso)

            # MATLAB later always does:
            # Geoph = flip(Geop,3)./(1000*g0)
            Geop_for_geoph = np.flip(Geop, axis=0)

        elif "pressure_level" in nc.variables:
            iso = np.asarray(nc.variables["pressure_level"][:], dtype=float)

            # MATLAB catch:
            # Geop = flip(Geop,3)
            # then later:
            # Geoph = flip(Geop,3)./(1000*g0)
            # This double-flips Geop effectively back.
            Geop = np.flip(Geop, axis=0)
            Geop_for_geoph = np.flip(Geop, axis=0)

        else:
            raise KeyError("ERA5 file missing both 'level' and 'pressure_level'.")

    # Convert from Python netCDF order: usually time, level, lat, lon
    # to MATLAB/gridcalc-like order: lon, lat, level
    if Spechum.ndim != 3:
        raise ValueError(f"Expected q to be 3D after time slicing, got {Spechum.shape}")

    Spechum = np.transpose(Spechum, (2, 1, 0))
    tempGrid = np.transpose(tempGrid, (2, 1, 0))
    Geoph = np.transpose(Geop_for_geoph / (1000.0 * g0), (2, 1, 0))

    pGrid = np.tile(
        iso.reshape(1, 1, -1),
        (len(lon_var), len(lat_var), 1),
    ).astype(float)

    if pGrid.shape != tempGrid.shape:
        raise ValueError(
            f"Grid shape mismatch: pGrid={pGrid.shape}, tempGrid={tempGrid.shape}. "
            "Check ERA5 dimension order."
        )

    # ------------------------------------------------------------------
    # Decimation factor, matching MATLAB n and n1 logic
    # ------------------------------------------------------------------
    grid_res = float(np.asarray(model["GRIDboundaries"]).ravel()[2])

    if grid_res != 0.25:
        n = int(round(grid_res / 0.25))
        if n < 1:
            raise ValueError(f"Invalid decimation factor n={n} from grid_res={grid_res}")
    else:
        n = 1

    n1 = n

    if len(model["lat_TOMO"]) == undugrid.shape[0]:
        n1 = 1

    # ------------------------------------------------------------------
    # LAT/LON grid
    # ------------------------------------------------------------------
    if globalepoch == 1:
        LAT, LON = np.meshgrid(lat_var, lon_var)
        LAT = np.deg2rad(LAT)
        LON = np.deg2rad(LON)

        LAT = LAT[::n, ::n]
        LON = LON[::n, ::n]

        tempgrid["LAT"] = LAT
        tempgrid["LON"] = LON
        tempgrid["lam_N"] = lam_N
        tempgrid["phi_N"] = phi_N
    else:
        for key in ["LAT", "LON", "lam_N", "phi_N"]:
            if key not in tempgrid:
                raise KeyError(
                    f"tempgrid['{key}'] missing at epoch {globalepoch}. "
                    "Run epoch 1 first or preserve tempgrid between epochs."
                )

        LAT = tempgrid["LAT"]
        LON = tempgrid["LON"]
        lam_N = tempgrid["lam_N"]
        phi_N = tempgrid["phi_N"]

    # ------------------------------------------------------------------
    # Subsample grids
    # ------------------------------------------------------------------
    pGrid_sub = pGrid[::n, ::n, :]
    tempGrid_sub = tempGrid[::n, ::n, :]
    Spechum_sub = Spechum[::n, ::n, :]
    Geoph_sub = Geoph[::n, ::n, :]
    undugrid_sub = undugrid[::n1, ::n1]

    _check_grid_shapes(
        LAT,
        LON,
        undugrid_sub,
        pGrid_sub,
        tempGrid_sub,
        Spechum_sub,
        Geoph_sub,
    )

    # ------------------------------------------------------------------
    # Interpolate ERA5 to RT grid
    # ------------------------------------------------------------------
    RTstruc = {}

    model_interp = _switch(switches.get("modelInterp", "accurate"), "accurate")

    if model_interp == "accurate":
        pGrid3D, eGrid3D, tempGrid3D, RTstruc["rWGS"] = gridcalc(
            LAT,
            LON,
            undugrid_sub,
            pGrid_sub,
            tempGrid_sub,
            Spechum_sub,
            Geoph_sub,
            hpre,
            phi_N,
            lam_N,
            BLHstruc,
            model,
        )
    else:
        warnings.warn(
            f"Using gridcalc2 because switches['modelInterp']={model_interp!r}."
        )

        if BLHstruc is None:
            raise ValueError(
                "\n[ERROR] gridcalc2 requires BLHstruc, but it is missing.\n"
                "Run epoch 1 first, preserve tempgrid between epochs, or use "
                "switches['modelInterp']=['accurate']."
            )

        pGrid3D, eGrid3D, tempGrid3D, RTstruc["rWGS"] = gridcalc2(
            LAT,
            LON,
            undugrid_sub,
            pGrid_sub,
            tempGrid_sub,
            Spechum_sub,
            Geoph_sub,
            hpre,
            phi_N,
            lam_N,
            BLHstruc,
            model,
        )
    pGrid3D = np.rot90(pGrid3D, 1, axes=(0, 1))
    eGrid3D = np.rot90(eGrid3D, 1, axes=(0, 1))
    tempGrid3D = np.rot90(tempGrid3D, 1, axes=(0, 1))
    RTstruc["rWGS"] = np.rot90(RTstruc["rWGS"], 1, axes=(0, 1))

    if pGrid3D.shape[2] < hpre.size:
        raise ValueError(
            f"pGrid3D has only {pGrid3D.shape[2]} vertical levels, "
            f"but hpre requires {hpre.size}."
        )

    model["pres3D"] = pGrid3D[:, :, : hpre.size]
    model["temp3D"] = tempGrid3D[:, :, : hpre.size]

    # ------------------------------------------------------------------
    # REAL solution: station pressure, temperature, ZHD
    # ------------------------------------------------------------------
    if _switch(switches.get("solution")) == "REAL":
        if "BLh" not in model:
            warnings.warn(
                "switches['solution']='REAL' but model['BLh'] is missing. "
                "pBLh2ZHDRT may fail."
            )

        ZHD, Pstat, Tstat = pBLh2ZHDRT(model)

        model.setdefault("ZHD", {})
        model["ZHD"][globalepoch] = np.asarray(ZHD).ravel()

        if "Pstat" in model:
            model["Pstat"][globalepoch - 1, :] = np.asarray(Pstat).ravel()
        else:
            model.setdefault("Pstat_dict", {})[globalepoch] = np.asarray(Pstat).ravel()
            warnings.warn("model['Pstat'] missing; stored values in model['Pstat_dict'].")

        if "Tstat" in model:
            model["Tstat"][globalepoch - 1, :] = np.asarray(Tstat).ravel()
        else:
            model.setdefault("Tstat_dict", {})[globalepoch] = np.asarray(Tstat).ravel()
            warnings.warn("model['Tstat'] missing; stored values in model['Tstat_dict'].")

    # ------------------------------------------------------------------
    # Reshape to RTstruc format
    # ------------------------------------------------------------------
    RTstruc["rWGS"] = RTstruc["rWGS"].reshape(
        1,
        RTstruc["rWGS"].shape[0] * RTstruc["rWGS"].shape[1],
    )

    pGrid3D_perm = np.transpose(model["pres3D"], axes=(1, 0, 2))
    RTstruc["pGrid3D"] = pGrid3D_perm.reshape(
        pGrid3D_perm.shape[0] * pGrid3D_perm.shape[1],
        pGrid3D_perm.shape[2],
    ).T

    eGrid3D = eGrid3D[:, :, : hpre.size]
    eGrid3D_perm = np.transpose(eGrid3D, axes=(1, 0, 2))
    RTstruc["eGrid3D"] = eGrid3D_perm.reshape(
        eGrid3D_perm.shape[0] * eGrid3D_perm.shape[1],
        eGrid3D_perm.shape[2],
    ).T

    tempGrid3D = tempGrid3D[:, :, : hpre.size]
    tempGrid3D_perm = np.transpose(tempGrid3D, axes=(1, 0, 2))
    RTstruc["tempGrid3D"] = tempGrid3D_perm.reshape(
        tempGrid3D_perm.shape[0] * tempGrid3D_perm.shape[1],
        tempGrid3D_perm.shape[2],
    ).T

    LAT_rep = np.tile(phi_N.reshape(-1, 1), (1, len(lam_N))).T
    LAT_rep = LAT_rep.reshape(1, LAT_rep.shape[0] * LAT_rep.shape[1])

    RTstruc["LAT"] = np.tile(LAT_rep, (len(hpre), 1))
    RTstruc["LON"] = np.tile(lam_N, (len(hpre), len(phi_N)))

    # ------------------------------------------------------------------
    # Refractivity
    # MATLAB uses refcalc(..., 'c'), not 'b'
    # ------------------------------------------------------------------
    Nh, Nw = refcalc(
        RTstruc["pGrid3D"],
        RTstruc["tempGrid3D"],
        RTstruc["eGrid3D"],
        "c",
    )

    RTstruc["N3D"] = Nw
    RTstruc["N3D_RT"] = Nw + Nh

    # ------------------------------------------------------------------
    # Save variables
    # ------------------------------------------------------------------
    model.setdefault("temp", {})[globalepoch] = RTstruc["tempGrid3D"]
    model.setdefault("pres", {})[globalepoch] = RTstruc["pGrid3D"]
    model.setdefault("wvpr", {})[globalepoch] = RTstruc["eGrid3D"]
    model.setdefault("refrNw", {})[globalepoch] = RTstruc["N3D"]
    model.setdefault("refrN", {})[globalepoch] = RTstruc["N3D_RT"]

    if globalepoch == 1:
        model["BLh_pudel_rad"] = BLHstruc
        model["rWGS"] = RTstruc["rWGS"]
        model["LAT"] = RTstruc["LAT"]
        model["LON"] = RTstruc["LON"]

    return model, tempgrid