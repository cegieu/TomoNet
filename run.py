#!/usr/bin/env python3
"""
TomoNet v.1.1

Designed by Adam Cegla at ETH Zurich, Chair of Space Geodesy.
15.05.2026

This software is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users should
refer to the GNU General Public License for more details.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import tomo_conf as tc


def _as_scalar(value):
    if isinstance(value, list):
        return value[0]
    return value


def _require_keys(container, keys, name):
    for key in keys:
        if key not in container:
            raise KeyError(f"Missing {name}['{key}'].")


def _ensure_dirs(paths, keys):
    for key in keys:
        path = Path(paths[key])
        path.mkdir(parents=True, exist_ok=True)


def _check_array(name, arr, ndim=None, min_size=1):
    arr = np.asarray(arr)

    if arr.size < min_size:
        raise ValueError(f"{name} is empty.")

    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {arr.shape}.")

    try:
        if np.any(~np.isfinite(arr.astype(float, copy=False))):
            warnings.warn(f"{name} contains NaN or infinite values.")
    except Exception:
        warnings.warn(f"{name} could not be checked for finite numeric values.")

    return arr


def _validate_after_config(model, switches, paths, dates):
    _require_keys(
        paths,
        [
            "mainpath",
            "modeldata",
            "station_file",
            "pathCONF",
            "pathTOMO",
            "pathORB",
            "pathMETEO",
            "pathATM",
            "pathSAVE",
            "pathEXPORT",
        ],
        "paths",
    )

    _require_keys(
        switches,
        [
            "solution",
            "method",
            "aprModel",
            "setwieghts",
            "modelInterp",
            "stat_range",
            "unduFile",
        ],
        "switches",
    )

    _require_keys(
        model,
        [
            "radii",
            "lat_TOMO",
            "lon_TOMO",
            "levels_TOMO",
            "h_RT",
            "GRIDboundaries",
            "cut_off_angle",
            "num_lat_TOMO",
            "num_lon_TOMO",
            "num_levels_TOMO",
        ],
        "model",
    )

    solution = _as_scalar(switches["solution"])
    method = _as_scalar(switches["method"])
    apr_model = _as_scalar(switches["aprModel"])

    if solution not in {"REAL", "SYNTHETIC"}:
        raise ValueError("switches['solution'] must be REAL or SYNTHETIC.")

    if method not in {"LSQ", "KALMAN", "NN"}:
        raise ValueError("switches['method'] must be LSQ, KALMAN or NN.")

    if apr_model not in {"ERA5", "DETER"}:
        raise ValueError("switches['aprModel'] must be ERA5 or DETER.")

    if solution == "REAL" and apr_model == "DETER":
        warnings.warn(
            "REAL observations with DETER apriori are enabled. "
            "This is allowed, but check that this is intended."
        )

    n_vox = (
        model["num_lat_TOMO"]
        * model["num_lon_TOMO"]
        * model["num_levels_TOMO"]
    )

    if n_vox <= 0:
        raise ValueError("Invalid tomography voxel count.")

    if model["cut_off_angle"] < 0 or model["cut_off_angle"] >= 90:
        raise ValueError("model['cut_off_angle'] must be in range [0, 90).")

    _ensure_dirs(
        paths,
        ["pathTOMO", "pathORB", "pathMETEO", "pathATM", "pathEXPORT"],
    )


def _validate_stations(NAME, BLh_ori):
    if len(NAME) == 0:
        raise ValueError("No GNSS stations found in station file.")

    BLh_ori = _check_array("BLh_ori", BLh_ori, ndim=2)

    if BLh_ori.shape[1] < 5:
        raise ValueError(
            f"BLh_ori must have at least 5 columns, got shape {BLh_ori.shape}."
        )

    if BLh_ori.shape[0] != len(NAME):
        raise ValueError(
            f"Station name count {len(NAME)} does not match coordinate rows "
            f"{BLh_ori.shape[0]}."
        )

    return BLh_ori


def _validate_obs_set(obs_set):
    for key in ["observation_set", "observation_set_SP3"]:
        if not hasattr(obs_set, key):
            raise AttributeError(f"obs_set is missing attribute '{key}'.")

    if obs_set.observation_set.size == 0:
        raise ValueError("obs_set.observation_set is empty.")

    if obs_set.observation_set_SP3.size == 0:
        raise ValueError("obs_set.observation_set_SP3 is empty.")


def _validate_apriori(apriori, model, n_epochs):
    n_vox = (
        model["num_lat_TOMO"]
        * model["num_lon_TOMO"]
        * model["num_levels_TOMO"]
    )

    for key in ["Nw_DETER", "Nw_ERA5"]:
        if key not in apriori:
            continue

        if len(apriori[key]) != n_epochs:
            warnings.warn(
                f"apriori['{key}'] has {len(apriori[key])} epochs, "
                f"expected {n_epochs}."
            )

        for i, arr in enumerate(apriori[key]):
            arr = np.asarray(arr).ravel()
            if arr.size != n_vox:
                raise ValueError(
                    f"apriori['{key}'][{i}] has {arr.size} elements, "
                    f"expected {n_vox}."
                )


def _install_paths(mainpath, modeldata):
    sys.path.insert(0, mainpath)
    sys.path.insert(0, os.path.join(mainpath, modeldata))
    sys.path.insert(0, os.path.join(mainpath, "Python/others"))
    sys.path.insert(0, os.path.join(mainpath, "Python/raytracer"))
    sys.path.insert(0, os.path.join(mainpath, "Python/read"))
    sys.path.insert(0, os.path.join(mainpath, "Python/tomo"))
    sys.path.insert(0, os.path.join(mainpath, "Python/model_train"))


def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # Load configuration inside main to avoid Python local/global scoping issues
    # ------------------------------------------------------------------
    model, switches, project_name, paths, dates = tc.tomo_config()
    _validate_after_config(model, switches, paths, dates)

    mainpath = paths["mainpath"]
    modeldata = paths["modeldata"]

    _install_paths(mainpath, modeldata)

    # ------------------------------------------------------------------
    # Imports depending on TomoNet paths
    # ------------------------------------------------------------------
    import stat_read as st
    import NWMread as NWM
    import gpt2 as gpt2
    import construct_station as cs
    import raytracing as rt
    import find_epochs as fe
    from nn_tomography import run_nn_model

    from read_orbit import readSP3dat, download_orb, interSP3
    from readZTD import readtxtOBS, screen_ztd
    from tomography import tomography

    solution = _as_scalar(switches["solution"])
    method = _as_scalar(switches["method"])
    apr_model = _as_scalar(switches["aprModel"])

    print("[INFO] TomoNet run started")
    print(f"[INFO] Project: {project_name}")
    print(f"[INFO] Solution: {solution}")
    print(f"[INFO] Method: {method}")
    print(f"[INFO] Apriori model: {apr_model}")

    # ------------------------------------------------------------------
    # Read GNSS stations
    # ------------------------------------------------------------------
    station_path = Path(paths["pathCONF"]) / paths["station_file"]

    if not station_path.exists():
        raise FileNotFoundError(f"Station file not found: {station_path}")

    try:
        NAME, BLh_ori = st.read_stat(str(station_path))
    except Exception as exc:
        raise RuntimeError(
            f"Wrong format of GNSS station coordinate file: {station_path}"
        ) from exc

    BLh_ori = _validate_stations(NAME, BLh_ori)

    print(f"[INFO] Loaded {len(NAME)} GNSS stations from {station_path}")

    # ------------------------------------------------------------------
    # Convert station coordinates to XYZ
    # ------------------------------------------------------------------
    Xsta = np.zeros((3, BLh_ori.shape[0]), dtype=float)

    for i in range(BLh_ori.shape[0]):
        lat_deg = float(BLh_ori[i, 2])
        lon_deg = float(BLh_ori[i, 1])
        h_km = float(BLh_ori[i, 4]) / 1000.0

        Xsta[:, i] = st.geodetic_to_ecef(
            lat_deg,
            lon_deg,
            h_km,
            model["radii"][0],
            (model["radii"][0] - model["radii"][1]) / model["radii"][0],
        )

    X, Y, Z, lat, lon, h, H, NAME = st.boundingTOMO(
        model["east_limit_TOMO"],
        model["west_limit_TOMO"],
        model["north_limit_TOMO"],
        model["south_limit_TOMO"],
        Xsta,
        BLh_ori,
        NAME,
    )

    if len(NAME) == 0:
        raise ValueError(
            "No stations remain after boundingTOMO. "
            "Check station coordinates and TOMO boundaries."
        )

    X, Y, Z, lat, lon, h, H, NAME = st.removeStat(
        X, Y, Z, lat, lon, h, H, NAME, switches["stat_range"]
    )

    if len(NAME) == 0:
        raise ValueError(
            "No stations remain after removeStat. "
            "Reduce switches['stat_range'] or check station distribution."
        )

    BLh = np.vstack(
        (
            np.arange(1, len(lat) + 1),
            np.rad2deg(lat),
            np.rad2deg(lon),
            h,
        )
    ).T

    BLh_ori = np.vstack(
        (
            np.arange(1, len(lat) + 1),
            np.rad2deg(lat),
            np.rad2deg(lon),
            H,
            h,
        )
    ).T

    model["BLh"] = BLh
    model["BLH"] = BLh_ori[:, :4]
    model["NAME"] = NAME

    print(f"[INFO] Stations after filtering: {len(NAME)}")

    # ------------------------------------------------------------------
    # Find processing epochs
    # ------------------------------------------------------------------
    obs_set = fe.find_epochs(
        switches,
        dates["observation_start_TOMO"],
        dates["observation_end_TOMO"],
        dates["estimation_interval_TOMO"],
        dates["observation_interval_SP3"],
        dates["observation_interval_ZTD"],
        dates["observation_interval_NWP"],
        dates["interpolation_interval_METEO"],
    )

    _validate_obs_set(obs_set)

    n_epochs = obs_set.observation_set.shape[0]

    print(f"[INFO] Number of processing epochs: {n_epochs}")

    # ------------------------------------------------------------------
    # Create or load NWM/apriori model
    # ------------------------------------------------------------------
    NWM_file_path = Path(paths["pathTOMO"]) / "modelNWM.npy"

    rows = obs_set.observation_set.shape[0]
    cols = BLh.shape[0]

    model["Pstat"] = np.zeros((rows, cols), dtype=float)
    model["Tstat"] = np.zeros((rows, cols), dtype=float)

    if not NWM_file_path.exists():
        print("[INFO] NWM model file not found. Building modelNWM.npy")

        tempgrid = {}

        for epoch in range(n_epochs):
            try:
                date0 = datetime(
                    int(obs_set.observation_set[epoch, 2]),
                    int(obs_set.observation_set[epoch, 6]),
                    int(obs_set.observation_set[epoch, 7]),
                    int(obs_set.observation_set[epoch, 8]),
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not build datetime for epoch {epoch} from observation_set."
                ) from exc

            ERAname = (
                f"ERA5_{int(obs_set.observation_set[epoch, 2])}-"
                f"{int(obs_set.observation_set[epoch, 6])}-"
                f"{int(obs_set.observation_set[epoch, 7])}.nc"
            )

            ERA_path = Path(paths["pathMETEO"]) / ERAname

            if not ERA_path.exists():
                warnings.warn(
                    f"ERA5 file not found before NWMread call: {ERA_path}. "
                    "NWMread may fail unless it can locate/download it internally."
                )

            model, tempgrid = NWM.NWMread(
                ERAname,
                model,
                tempgrid,
                model["GRIDboundaries"],
                date0,
                paths["pathMETEO"],
                paths["pathCONF"],
                dates["observation_start_TOMO"],
                epoch + 1,
                switches["unduFile"],
                switches,
            )

            print(f"[INFO] NWM epoch {epoch + 1}/{n_epochs}: {ERAname}")

        if "pres3D" in model:
            del model["pres3D"]

        np.save(NWM_file_path, model)
        print(f"[INFO] Saved NWM model: {NWM_file_path}")

    else:
        model = np.load(NWM_file_path, allow_pickle=True).item()
        print(f"[INFO] Loaded existing NWM model: {NWM_file_path}")

    if "BLh_pudel_rad" not in model:
        raise KeyError(
            "model['BLh_pudel_rad'] missing after NWM step. "
            "Check NWMread and modelNWM.npy."
        )

    # ------------------------------------------------------------------
    # Download and process SP3 orbit files
    # ------------------------------------------------------------------
    pathORB = paths["pathORB"]

    download_orb(pathORB, obs_set.observation_set_SP3)

    SP3data = readSP3dat(obs_set.observation_set_SP3, pathORB)

    SP3Xn, SP3Yn, SP3Zn, PRNn, obs_set.observation_set_SP3 = interSP3(
        SP3data,
        obs_set.observation_set_SP3,
    )

    SP3Xn = _check_array("SP3Xn", SP3Xn, ndim=2)
    SP3Yn = _check_array("SP3Yn", SP3Yn, ndim=2)
    SP3Zn = _check_array("SP3Zn", SP3Zn, ndim=2)
    PRNn = _check_array("PRNn", PRNn, ndim=2)

    if SP3Xn.shape != SP3Yn.shape or SP3Xn.shape != SP3Zn.shape:
        raise ValueError(
            f"SP3 coordinate shape mismatch: "
            f"X={SP3Xn.shape}, Y={SP3Yn.shape}, Z={SP3Zn.shape}"
        )

    paths["pathORB"] = pathORB

    print(f"[INFO] SP3 interpolation complete. Shape: {SP3Xn.shape}")

    # ------------------------------------------------------------------
    # Read ZTD for REAL solution
    # ------------------------------------------------------------------
    ZTDA = MZTDA = DGNA = MDGNA = DGEA = MDGEA = None

    if solution == "REAL":
        pathATM = paths["pathATM"]

        ZTDA, MZTDA, DGNA, MDGNA, DGEA, MDGEA, NAMES, MISS_STAT = readtxtOBS(
            pathATM,
            model["NAME"],
            obs_set.observation_set,
        )

        ZTDA, MZTDA, DGNA, MDGNA, DGEA, MDGEA, model = screen_ztd(
            ZTDA,
            MZTDA,
            DGNA,
            MDGNA,
            DGEA,
            MDGEA,
            MISS_STAT,
            model,
        )

        if ZTDA.shape[0] != n_epochs:
            raise ValueError(
                f"ZTDA epoch count {ZTDA.shape[0]} does not match {n_epochs}."
            )

        print("[INFO] ZTD observations loaded and screened")

    # ------------------------------------------------------------------
    # DETER/GPT2 apriori
    # ------------------------------------------------------------------
    apriori = {}
    grdmodel = {}

    for epoch in range(n_epochs):
        T, _, undu, grdmodel = gpt2.distr_T_gpt2RT(
            model["BLh_pudel_rad"],
            obs_set.observation_set[epoch, 0],
            os.path.join(mainpath, modeldata),
            epoch,
            grdmodel,
        )

        E, _ = gpt2.distr_e_unb3RT(
            model["BLh_pudel_rad"],
            undu,
            obs_set.observation_set[epoch, 0],
        )

        NwT, _ = gpt2.eT2Nw(E, T)

        apriori.setdefault("Nw_DETER", []).append(np.asarray(NwT).ravel())

        R = 461.525
        WVT = (
            100.0
            * np.asarray(E).ravel()
            / (R * np.asarray(T).ravel())
            * 1000.0
        )
        apriori.setdefault("WVT_DETER", []).append(WVT)

    print("[INFO] DETER/GPT2 apriori created")

    # ------------------------------------------------------------------
    # ERA5 apriori
    # ------------------------------------------------------------------
    if "refrNw" not in model or "wvpr" not in model or "temp" not in model:
        warnings.warn(
            "ERA5 apriori fields are missing from model. "
            "apriori['Nw_ERA5'] will not be created."
        )
    else:
        for epoch in range(n_epochs):
            key = epoch + 1

            if key not in model["refrNw"]:
                raise KeyError(f"model['refrNw'][{key}] missing.")

            NwT1_num = model["refrNw"][key]
            E1_num = model["wvpr"][key]
            T1_num = model["temp"][key]

            NwT_num = np.asarray(NwT1_num).reshape(-1)
            E_num = np.asarray(E1_num).reshape(-1)
            T_num = np.asarray(T1_num).reshape(-1)

            apriori.setdefault("Nw_ERA5", []).append(NwT_num)

            R = 461.525
            WVT_num = 100.0 * E_num / (R * T_num) * 1000.0
            apriori.setdefault("WV_ERA5", []).append(WVT_num)

        print("[INFO] ERA5 apriori assigned")

    _validate_apriori(apriori, model, n_epochs)

    if apr_model == "ERA5" and "Nw_ERA5" not in apriori:
        raise KeyError(
            "switches['aprModel']='ERA5', but apriori['Nw_ERA5'] was not created."
        )

    if apr_model == "DETER" and "Nw_DETER" not in apriori:
        raise KeyError(
            "switches['aprModel']='DETER', but apriori['Nw_DETER'] was not created."
        )

    # ------------------------------------------------------------------
    # Create or load observation structure
    # ------------------------------------------------------------------
    obs_file_path = Path(paths["pathTOMO"]) / "obs.npy"

    if not obs_file_path.exists():
        print("[INFO] obs.npy not found. Constructing station observation structure.")

        if solution == "SYNTHETIC":
            obs = cs.construct_station(
                model["BLh"],
                model["BLH"],
                0,
                0,
                0,
                model["NAME"],
                PRNn,
                SP3Xn,
                SP3Yn,
                SP3Zn,
                0,
                0,
                obs_set.observation_set_SP3,
                model["cut_off_angle"],
                os.path.join(mainpath, modeldata),
                grdmodel,
                switches,
                0,
                0,
                0,
                0,
            )

        elif solution == "REAL":
            if ZTDA is None:
                raise RuntimeError("REAL solution requires ZTD observations, but ZTDA is None.")

            if "ZHD" not in model:
                raise KeyError("REAL solution requires model['ZHD'].")

            obs = cs.construct_station(
                model["BLh"],
                model["BLH"],
                ZTDA,
                MZTDA,
                model["ZHD"],
                model["NAME"],
                PRNn,
                SP3Xn,
                SP3Yn,
                SP3Zn,
                model["Pstat"],
                model["Tstat"],
                obs_set.observation_set_SP3,
                model["cut_off_angle"],
                os.path.join(mainpath, modeldata),
                grdmodel,
                switches,
                DGNA,
                MDGNA,
                DGEA,
                MDGEA,
            )

        else:
            raise ValueError(f"Unsupported solution: {solution}")

        np.save(obs_file_path, obs)
        print(f"[INFO] Saved observation structure: {obs_file_path}")

    else:
        obs = np.load(obs_file_path, allow_pickle=True)
        print(f"[INFO] Loaded existing observation structure: {obs_file_path}")

    if len(obs) == 0:
        raise ValueError("Observation structure obs is empty.")

    if len(obs) != n_epochs:
        warnings.warn(
            f"obs has {len(obs)} epochs, but obs_set has {n_epochs}. "
            "Check cached obs.npy."
        )

    # ------------------------------------------------------------------
    # Ray tracing
    # ------------------------------------------------------------------
    raytracing_results = []

    for epoch in range(len(obs)):
        A, SD, SDtest, elev = rt.raytracing(obs, model, epoch, paths, switches)

        A = np.asarray(A)
        SD = np.asarray(SD).ravel()
        elev = np.asarray(elev).ravel()

        if A.size == 0:
            warnings.warn(f"Raytracing returned empty A at epoch {epoch}.")

        if A.ndim != 2:
            raise ValueError(f"Raytracing A must be 2D at epoch {epoch}, got {A.shape}.")

        if A.shape[0] != SD.size:
            raise ValueError(
                f"Raytracing mismatch at epoch {epoch}: "
                f"A rows={A.shape[0]}, SD length={SD.size}."
            )

        if elev.size != A.shape[0]:
            warnings.warn(
                f"Elevation count mismatch at epoch {epoch}: "
                f"elev={elev.size}, A rows={A.shape[0]}."
            )

        raytracing_results.append(
            {
                "A": A,
                "SD": SD,
                "SDtest": SDtest,
                "elev": elev,
            }
        )

        print(f"[INFO] Raytracing epoch {epoch + 1}/{len(obs)} complete")

    # ---------------------------z---------------------------------------
    # Tomography
    # ------------------------------------------------------------------
    if method == "NN":
        output = run_nn_model(raytracing_results,apriori,model,paths,switches,project_name)
    else:
        output = tomography(raytracing_results, apriori, obs, model, paths, switches)

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    out_dir = Path(paths["pathSAVE"]) / project_name / "OUT"
    out_dir.mkdir(parents=True, exist_ok=True)

    tomo_file_path = out_dir / f"out_{solution}_{method}_{apr_model}.npy"

    np.save(tomo_file_path, output)

    print(f"[INFO] Saved tomography output: {tomo_file_path}")
    print(f"[INFO] Total runtime: {time.time() - start_time:.2f} sec")

    return output, raytracing_results, model, switches, paths


if __name__ == "__main__":
    main()