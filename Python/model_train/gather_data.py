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
import warnings
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse


DEFAULT_SYNTHETIC_FILE = "out_SYNTHETIC_LSQ25_det.mat"
DEFAULT_MODEL_FILE = "modelTOMO.mat"
DEFAULT_WORK_DIR = "WORK"


def _require_file(path: str, description: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def _get_required_key(mat_dict: dict, key: str, mat_path: str):
    value = mat_dict.get(key, None)
    if value is None:
        available = sorted(k for k in mat_dict.keys() if not k.startswith("__"))
        raise KeyError(
            f"Missing variable '{key}' in {mat_path}. "
            f"Use exactly this variable name: '{key}'. Available variables: {available}"
        )
    return value


def _get_required_attr(obj, attr: str, mat_path: str, parent_name: str):
    value = getattr(obj, attr, None)
    if value is None:
        available = sorted(k for k in dir(obj) if not k.startswith("_")) if obj is not None else []
        raise KeyError(
            f"Missing field '{parent_name}.{attr}' in {mat_path}. "
            f"Use exactly this field name: '{attr}'. Available fields in '{parent_name}': {available}"
        )
    return value


def _validate_epoch(A, SWD, Nref, Napr, elev, fpath: str) -> None:
    if A is None:
        raise KeyError(f"Missing variable 'A' in {fpath}. Use exactly this variable name: 'A'.")
    if SWD is None:
        raise KeyError(f"Missing variable 'SD' in {fpath}. Use exactly this variable name: 'SD'.")
    if elev is None:
        raise KeyError(f"Missing variable 'elev' in {fpath}. Use exactly this variable name: 'elev'.")

    A_shape = np.asarray(A).shape
    if len(A_shape) != 2:
        raise ValueError(f"Variable 'A' in {fpath} must be a 2-D observation matrix; got shape {A_shape}.")
    m, n = A_shape
    if np.asarray(SWD).size != m:
        raise ValueError(f"Variable 'SD' in {fpath} has {np.asarray(SWD).size} values, but A has {m} rows.")
    if np.asarray(elev).size != m:
        raise ValueError(f"Variable 'elev' in {fpath} has {np.asarray(elev).size} values, but A has {m} rows.")
    if np.asarray(Napr).size != n:
        raise ValueError(f"Field 'apriori.Nw_ERA5' has {np.asarray(Napr).size} values, but A has {n} columns.")
    if Nref is not None and np.asarray(Nref).size != n:
        raise ValueError(f"Field 'apriori.Nw_DETER' has {np.asarray(Nref).size} values, but A has {n} columns.")


def process_all_data(
    pathDATA: str,
    years: Optional[Iterable[int]] = None,
    months: Optional[Iterable[int]] = None,
    synthetic_filename: str = DEFAULT_SYNTHETIC_FILE,
    model_filename: str = DEFAULT_MODEL_FILE,
    work_dirname: str = DEFAULT_WORK_DIR,
    matrix_template: str = "amtrix_{epoch}.mat",
    strict: bool = True,
) -> Tuple[list, list, list, list, list]:
    """
    Load tomography epochs for TomoNet training/evaluation.

    Input variables
    ---------------
    pathDATA : str
        Root data directory containing year/month subdirectories.
    years : iterable of int
        Years to load, for example [2023] or range(2022, 2024).
    months : iterable of int
        Month numbers to load, for example range(1, 13).
    synthetic_filename : str
        Name of the monthly synthetic/reference MATLAB file. Default:
        'out_SYNTHETIC_LSQ25_det.mat'. This file must contain variable 'output'
        with field 'Nw'.
    model_filename : str
        Name of the apriori MATLAB file in the WORK directory. Default:
        'modelTOMO.mat'. This file must contain variable 'apriori' with fields
        'Nw_DETER' and 'Nw_ERA5'.
    work_dirname : str
        Name of the work subdirectory containing model_filename and epoch matrix files.
    matrix_template : str
        Filename template for each epoch. It must include '{epoch}', for example
        'amtrix_{epoch}.mat'. Each file must contain variables 'A', 'SD', and 'elev'.
    strict : bool
        If True, stop at the first missing/invalid file or variable. If False,
        warn and skip the problematic epoch/month.

    Output variables
    ----------------
    A_list : list of scipy.sparse.csr_matrix
        Tomography design matrices, one matrix per epoch, shape (n_observations, n_voxels).
    SWD_list : list of numpy.ndarray
        Slant wet delay observation vectors named 'SD' in the MATLAB files.
    Nref_list : list of numpy.ndarray
        Reference wet refractivity fields from 'apriori.Nw_DETER'.
    Napr_list : list of numpy.ndarray
        Apriori wet refractivity fields from 'apriori.Nw_ERA5'.
    Elev_list : list of numpy.ndarray
        Elevation vectors from variable 'elev', aligned with rows of A and SD.
    """
    if not pathDATA or not os.path.isdir(pathDATA):
        raise NotADirectoryError(f"pathDATA does not exist or is not a directory: {pathDATA}")
    if years is None:
        raise ValueError("years must be provided, for example years=[2023].")
    if months is None:
        months = range(1, 13)
    if "{epoch}" not in matrix_template:
        raise ValueError("matrix_template must include '{epoch}', e.g. 'amtrix_{epoch}.mat'.")

    A_list, SWD_list, Nref_list, Napr_list, Elev_list = [], [], [], [], []
    errors = []

    for year in years:
        year_path = os.path.join(pathDATA, str(year))
        if not os.path.isdir(year_path):
            msg = f"Missing year directory: {year_path}"
            if strict:
                raise FileNotFoundError(msg)
            warnings.warn(msg)
            continue

        for month in months:
            month_path = os.path.join(year_path, str(month))
            work_path = os.path.join(month_path, work_dirname)
            mat_path_temp = os.path.join(month_path, synthetic_filename)
            mat_path = os.path.join(work_path, model_filename)

            try:
                _require_file(mat_path_temp, f"synthetic/reference file '{synthetic_filename}'")
                _require_file(mat_path, f"apriori model file '{model_filename}'")

                data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
                data_temp = loadmat(mat_path_temp, squeeze_me=True, struct_as_record=False)

                output_temp = _get_required_key(data_temp, "output", mat_path_temp)
                output = _get_required_key(data, "apriori", mat_path)

                Napr_temp = _get_required_attr(output_temp, "Nw", mat_path_temp, "output")
                Nref_all = _get_required_attr(output, "Nw_DETER", mat_path, "apriori")
                Napr_all = _get_required_attr(output, "Nw_ERA5", mat_path, "apriori")

                n_epochs = int(np.asarray(Napr_temp).shape[0])
                if n_epochs <= 0:
                    raise ValueError(f"No epochs found in 'output.Nw' in {mat_path_temp}.")

                for epoch in range(1, n_epochs + 1):
                    fpath = os.path.join(work_path, matrix_template.format(epoch=epoch))
                    if not os.path.isfile(fpath):
                        msg = f"Missing epoch matrix file: {fpath}"
                        if strict:
                            raise FileNotFoundError(msg)
                        warnings.warn(msg)
                        continue

                    work_data = loadmat(fpath, squeeze_me=True, struct_as_record=False)
                    A = work_data.get("A", None)
                    SWD = work_data.get("SD", None)
                    elev = work_data.get("elev", None)
                    Nref = np.asarray(Nref_all[epoch - 1], dtype=np.float32)
                    Napr = np.asarray(Napr_all[epoch - 1], dtype=np.float32)
                    _validate_epoch(A, SWD, Nref, Napr, elev, fpath)

                    A_list.append(csr_matrix(np.asarray(A, dtype=np.float32)))
                    SWD_list.append(np.asarray(SWD, dtype=np.float32).ravel())
                    Nref_list.append(Nref.ravel())
                    Napr_list.append(Napr.ravel())
                    Elev_list.append(np.asarray(elev, dtype=np.float32).ravel())

            except Exception as exc:
                msg = f"Error while loading year={year}, month={month}: {exc}"
                if strict:
                    raise type(exc)(msg) from exc
                warnings.warn(msg)
                errors.append(msg)
                continue

    if len(A_list) == 0:
        raise RuntimeError("No valid tomography epochs were loaded. Check pathDATA, years, months, and MATLAB variable names.")

    print(f"[INFO] Loaded {len(A_list)} total epochs across selected years/months.")
    if errors:
        print(f"[WARN] Completed with {len(errors)} skipped month/epoch errors because strict=False.")
    return A_list, SWD_list, Nref_list, Napr_list, Elev_list


def summarize_data(A_list, SWD_list, Nref_list, Napr_list):
    """
    Print summary statistics for the loaded TomoNet input/output arrays.

    Input variables are the lists returned by process_all_data.
    Output is printed to stdout; the function returns None.
    """
    def summarize_list(name, data_list):
        if not data_list:
            print(f"[WARN] {name}: empty list")
            return
        vals = []
        for arr in data_list:
            if arr is None:
                continue
            x = arr.data if issparse(arr) else np.asarray(arr).ravel()
            if x.size > 0:
                vals.append(x)
        if not vals:
            print(f"[WARN] {name}: all arrays empty")
            return
        flat = np.concatenate(vals)
        print(f"[{name}] min={flat.min():.4e}, max={flat.max():.4e}, mean={flat.mean():.4e}, std={flat.std():.4e}, n={flat.size}")

    print("\n=== Dataset Summary ===")
    summarize_list("A_all", A_list)
    summarize_list("SWD_all", SWD_list)
    summarize_list("Nref_all", Nref_list)
    summarize_list("Napr_all", Napr_list)
    print("========================\n")
