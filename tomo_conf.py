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
import warnings
import numpy as np


# ==================================================================
# Function definitions (scroll down for configuration script)
# ==================================================================

def _as_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _as_scalar(value):
    if isinstance(value, list):
        return value[0]
    return value


def _validate_config(model, switches, paths, dates):
    required_switches = [
        "time_mode",
        "coord",
        "stat_range",
        "aprModel",
        "observations",
        "apriori",
        "constraints",
        "method",
        "solution",
        "unduFile",
        "modelInterp",
        "setwieghts",
    ]

    for key in required_switches:
        if key not in switches:
            raise KeyError(f"Missing switches['{key}'] in tomo_conf.py")

    solution = _as_scalar(switches["solution"])
    method = _as_scalar(switches["method"])
    apr_model = _as_scalar(switches["aprModel"])
    model_interp = _as_scalar(switches["modelInterp"])
    set_weights = _as_scalar(switches["setwieghts"])

    if solution not in {"REAL", "SYNTHETIC"}:
        raise ValueError("switches['solution'] must be 'REAL' or 'SYNTHETIC'.")

    if method not in {"LSQ", "KALMAN", "NN"}:
        raise ValueError("switches['method'] must be 'LSQ' or 'KALMAN'.")

    if apr_model not in {"ERA5", "DETER"}:
        raise ValueError("switches['aprModel'] must be 'ERA5' or 'DETER'.")

    if model_interp not in {"simple", "accurate"}:
        raise ValueError("switches['modelInterp'] must be 'simple' or 'accurate'.")

    if set_weights not in {"YES", "NO"}:
        raise ValueError("switches['setwieghts'] must be 'YES' or 'NO'.")

    if solution == "SYNTHETIC" and apr_model == "ERA5":
        warnings.warn(
            "SYNTHETIC solution with ERA5 apriori is allowed, but make sure "
            "ERA5 files are available and NWMread is executed."
        )

    required_model = [
        "radii",
        "lat_TOMO",
        "lon_TOMO",
        "h_RT",
        "cut_off_angle",
        "GRIDboundaries",
        "num_lat_TOMO",
        "num_lon_TOMO",
        "num_levels_TOMO",
        "levels_TOMO",
    ]

    for key in required_model:
        if key not in model:
            raise KeyError(f"Missing model['{key}'] in tomo_conf.py")

    if model["lat_TOMO"].size < 2:
        raise ValueError("model['lat_TOMO'] must contain at least 2 values.")

    if model["lon_TOMO"].size < 2:
        raise ValueError("model['lon_TOMO'] must contain at least 2 values.")

    if model["levels_TOMO"].size < 2:
        raise ValueError("model['levels_TOMO'] must contain at least 2 values.")

    if not (0 <= model["cut_off_angle"] < 90):
        raise ValueError("model['cut_off_angle'] must be in range [0, 90).")

    if np.any(np.diff(model["lat_TOMO"]) <= 0):
        raise ValueError("model['lat_TOMO'] must be strictly increasing.")

    if np.any(np.diff(model["lon_TOMO"]) <= 0):
        raise ValueError("model['lon_TOMO'] must be strictly increasing.")

    if np.any(np.diff(model["h_RT"]) <= 0):
        raise ValueError("model['h_RT'] must be strictly increasing.")

    if dates["observation_end_TOMO"] <= dates["observation_start_TOMO"]:
        raise ValueError("observation_end_TOMO must be later than observation_start_TOMO.")

    if dates["estimation_interval_TOMO"] <= 0:
        raise ValueError("dates['estimation_interval_TOMO'] must be positive.")

    required_paths = [
        "PATH_INSTALL",
        "PATH_EXTERNALSAVE",
        "pathSAVE",
        "pathCONF",
        "pathTOMO",
        "pathORB",
        "pathMETEO",
        "pathATM",
        "pathEXPORT",
        "mainpath",
        "modeldata",
        "station_file",
    ]

    for key in required_paths:
        if key not in paths:
            raise KeyError(f"Missing paths['{key}'] in tomo_conf.py")
    # ==================================================================
    # Configuration script start
    # ==================================================================

def tomo_config(mainpath: str | None = None) -> tuple[dict, dict, str, dict, dict]:
    """
    TomoNet manual configuration script.

    Returns
    -------
    model : dict
        Tomography and ray tracing model parameters.
    switches : dict
        Processing switches.
    PROJECT_NAME : str
        Project name.
    paths : dict
        Paths and filenames used by the processing chain.
    dates : dict
        Processing dates and intervals.
    """

    # ==================================================================
    # 0. Main installation settings
    # ==================================================================
    # Main path
    if mainpath is None:
        mainpath = "/scratch/AWARE/RayTracing/"

    PATH_INSTALL = Path(mainpath).expanduser().resolve()
    PATH_EXTERNALSAVE = PATH_INSTALL / "DATA"

    # Main project folder name
    PROJECT_NAME = "TOMO_2020_07_07"

    #Model Preparation function archive (do not modify)
    modeldata = "Python/modelprep"

    # Name of the file with GNSS station coordinates
    station_file = "stations_new.txt"

    # ==================================================================
    # 1. Time settings
    # ==================================================================
    dates: dict = {}
    # Start/End dates of tomography processing [YY, MM, DD, hh, mm, ss]
    dates["observation_start_TOMO"] = [2020, 7, 7, 0, 0, 0]
    dates["observation_end_TOMO"] = [2020, 7, 7, 23, 59, 59]
    # Time resolution for tomography processing [s]
    dates["estimation_interval_TOMO"] = 3600

    # Automatically derived intervals
    dates["observation_start_SP3"] = dates["observation_start_TOMO"]
    dates["observation_end_SP3"] = dates["observation_end_TOMO"]
    dates["observation_interval_SP3"] = dates["estimation_interval_TOMO"]

    dates["observation_start_ZTD"] = dates["observation_start_TOMO"]
    dates["observation_end_ZTD"] = dates["observation_end_TOMO"]
    dates["observation_interval_ZTD"] = dates["estimation_interval_TOMO"]

    dates["observation_start_METEO"] = dates["observation_start_TOMO"]
    dates["observation_end_METEO"] = dates["observation_end_TOMO"]
    dates["observation_interval_METEO"] = dates["estimation_interval_TOMO"]

    dates["interpolation_interval_METEO"] = dates["estimation_interval_TOMO"]
    dates["observation_interval_NWP"] = dates["estimation_interval_TOMO"]

    # ==================================================================
    # 2. Model settings
    # ==================================================================
    model: dict = {}
    # Tomography model horizontal resolution
    res = 0.25
    # Earth radius [km]
    model["radii"] = np.array([6378.137, 6356.752314245])
    # Tomography model latitudes [deg]
    model["lat_TOMO"] = np.arange(49.0, 52.5 + 1e-12, res)
    # Tomography model longitudes [deg]
    model["lon_TOMO"] = np.arange(16.5, 20.0 + 1e-12, res)
    # Tomography model altitudes [m]
    model["h_RT"] = np.array(
        [0,500,1000,1500,2000,2500,3000,4500,6000,7500,9000,14500,],dtype=float,)
    # Cut off elevation angle for ray tracing
    model["cut_off_angle"] = 10.0

    # =================================================================
    # 3. Processing switches
    # ==================================================================
    switches: dict = {}
    # Processing type (for now POSTPROCESSING only)
    switches["time_mode"] = "POSTPROCESSING"
    # Inpout coordinates file type for GNSS stations
    switches["coord"] = ["FORMATTED"]
    # Filtering range for GNSS station [km]
    switches["stat_range"] = 40
    # Apriori source: ERA5 or DETER
    switches["aprModel"] = "DETER"
    # Apriori source: SWD or/and IWV processing
    switches["observations"] = ["SWD", ""]
    # not working
    switches["apriori"] = ["INNER", "OUTER", "TOP", "BOTTOM"]
    # not working yet
    switches["constraints"] = ["", ""]
    # LSQ, KALMAN, NN
    switches["method"] = ["KALMAN"]
    # REAL or SYNTHETIC
    switches["solution"] = ["REAL"]
    # Undulation file name
    switches["unduFile"] = ["Undu"]
    # ERA5 interpolation of the near ground wet refractivities
    switches["modelInterp"] = ["simple"]
    # Set observation based weights
    switches["setwieghts"] = ["YES"]
    # If NN model is chosen
    switches["NN"] = {
        "model_type": "double_branch",  # one_branch / double_branch
        "checkpoint_name": "tomonet_model.pt", # name of the pretrained model
        "results_name": "nn_output.npy", # processing output name
    }
    # ==================================================================
    # 4. Automatically derived paths
    # ==================================================================
    paths: dict = {}

    paths["PATH_INSTALL"] = PATH_INSTALL
    paths["PATH_EXTERNALSAVE"] = PATH_EXTERNALSAVE
    paths["pathSAVE"] = PATH_EXTERNALSAVE
    paths["pathCONF"] = PATH_INSTALL / "CONF"
    paths["pathTOMO"] = PATH_INSTALL / "DATA" / PROJECT_NAME / "WORK"
    paths["pathORB"] = PATH_INSTALL / "DATA" / PROJECT_NAME / "ORB"
    paths["pathMETEO"] = PATH_INSTALL / "DATA" / PROJECT_NAME / "METEO"
    paths["pathATM"] = PATH_INSTALL / "DATA" / PROJECT_NAME / "ATM"
    paths["pathEXPORT"] = PATH_EXTERNALSAVE / PROJECT_NAME / "OUT"

    paths["mainpath"] = str(PATH_INSTALL)
    paths["modeldata"] = modeldata
    paths["station_file"] = station_file

    # ==================================================================
    # 5. Automatically derived model geometry
    # ==================================================================
    lam1 = float(model["lon_TOMO"][0])
    lam2 = float(model["lon_TOMO"][-1])
    lat1 = float(model["lat_TOMO"][0])
    lat2 = float(model["lat_TOMO"][-1])

    model["GRIDboundaries"] = np.array([lam1, lam2, res, lat1, lat2, res])

    model["num_lat_TOMO"] = model["lat_TOMO"].size
    model["num_lon_TOMO"] = model["lon_TOMO"].size
    model["res"] = res

    midpoints = model["h_RT"][:-1] + np.diff(model["h_RT"]) / 2.0
    model["levels_TOMO"] = np.concatenate(([0.0], midpoints))
    model["num_levels_TOMO"] = model["levels_TOMO"].size

    model["lat_RT"] = np.concatenate(
        (model["lat_TOMO"], [model["lat_TOMO"][-1] + res / 2.0])
    )

    model["lon_RT"] = np.concatenate(
        (model["lon_TOMO"] - res / 2.0, [model["lon_TOMO"][-1] + res / 2.0])
    )

    model["west_limit_TOMO"] = float(model["lon_TOMO"][0])
    model["east_limit_TOMO"] = float(model["lon_TOMO"][-1])
    model["south_limit_TOMO"] = float(model["lat_TOMO"][0])
    model["north_limit_TOMO"] = float(model["lat_TOMO"][-1])

    _validate_config(model, switches, paths, dates)

    print(f"[INFO] Project: {PROJECT_NAME}")
    print(f"[INFO] Main path: {paths['mainpath']}")
    print(f"[INFO] Model data folder: {paths['modeldata']}")
    print(f"[INFO] Station file: {paths['station_file']}")
    print(        "[INFO] Tomography voxels: "
        f"{model['num_lat_TOMO']} x {model['num_lon_TOMO']} x "
        f"{model['num_levels_TOMO']} = "
        f"{model['num_lat_TOMO'] * model['num_lon_TOMO'] * model['num_levels_TOMO']}"    )

    return model, switches, PROJECT_NAME, paths, dates