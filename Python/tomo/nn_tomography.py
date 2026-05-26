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
import importlib
import warnings
import numpy as np
import train_one
import train_double


def _as_scalar(value):
    """Return first element if value is stored as a one-element list."""
    if isinstance(value, list):
        return value[0]
    return value


def _get_nn_config(switches):
    """
    Read NN configuration from switches.

    Expected minimal structure in tomo_conf.py:

        switches["method"] = ["NN"]

        switches["NN"] = {
            "model_type": "double_branch",
            "checkpoint_name": "tomonet_model.pt",
            "results_name": "nn_output.npy",
        }
    """

    nn_conf = switches.get("NN", {})

    model_type = nn_conf.get("model_type", "double_branch")
    checkpoint_name = nn_conf.get("checkpoint_name", "tomonet_model.pt")
    results_name = nn_conf.get("results_name", "nn_output.npy")

    if model_type not in {"one_branch", "double_branch"}:
        raise ValueError(
            f"Invalid NN model_type='{model_type}'. "
            "Expected 'one_branch' or 'double_branch'."
        )

    if not isinstance(checkpoint_name, str) or checkpoint_name.strip() == "":
        raise ValueError("switches['NN']['checkpoint_name'] must be a non-empty string.")

    if not isinstance(results_name, str) or results_name.strip() == "":
        raise ValueError("switches['NN']['results_name'] must be a non-empty string.")

    return {
        "model_type": model_type,
        "checkpoint_name": checkpoint_name,
        "results_name": results_name,
    }


def _get_train_module_name(model_type):
    if model_type == "one_branch":
        return "train_one"

    if model_type == "double_branch":
        return "train_double"

    raise ValueError(
        f"Invalid NN model_type='{model_type}'. "
        "Expected 'one_branch' or 'double_branch'."
    )


def _get_apriori_key(switches):
    apr_model = _as_scalar(switches.get("aprModel", "DETER"))

    if apr_model == "ERA5":
        return "Nw_ERA5", apr_model

    if apr_model == "DETER":
        return "Nw_DETER", apr_model

    raise ValueError(
        f"Invalid switches['aprModel']='{apr_model}'. "
        "Expected 'ERA5' or 'DETER'."
    )


def _get_output_paths(paths, project_name, checkpoint_name, results_name):
    if "pathSAVE" not in paths:
        raise KeyError("Missing paths['pathSAVE'].")

    if "mainpath" not in paths:
        raise KeyError("Missing paths['mainpath'].")

    mainpath = Path(paths["mainpath"])

    checkpoint_path = mainpath / "Python" / "model_train" / checkpoint_name

    out_dir = Path(paths["pathSAVE"]) / project_name / "OUT"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / results_name

    return out_dir, checkpoint_path, results_path

def _validate_model_geometry(model):
    required = [
        "num_lat_TOMO",
        "num_lon_TOMO",
        "num_levels_TOMO",
    ]

    for key in required:
        if key not in model:
            raise KeyError(f"Missing model['{key}'].")

    n_vox = (
        int(model["num_lat_TOMO"])
        * int(model["num_lon_TOMO"])
        * int(model["num_levels_TOMO"])
    )

    if n_vox <= 0:
        raise ValueError(f"Invalid tomography voxel count: {n_vox}")

    return n_vox


def _validate_numeric_vector(name, arr, expected_len=None):
    arr = np.asarray(arr, dtype=float).ravel()

    if arr.size == 0:
        raise ValueError(f"{name} is empty.")

    if expected_len is not None and arr.size != expected_len:
        raise ValueError(
            f"{name} has length {arr.size}, expected {expected_len}."
        )

    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")

    return arr


def _validate_design_matrix(name, A, n_vox):
    A = np.asarray(A, dtype=float)

    if A.size == 0:
        raise ValueError(f"{name} is empty.")

    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}.")

    if A.shape[1] != n_vox:
        raise ValueError(
            f"{name} has {A.shape[1]} columns, expected {n_vox} tomography voxels."
        )

    if np.any(~np.isfinite(A)):
        raise ValueError(f"{name} contains NaN or Inf.")

    return A


def _prepare_nn_inputs(raytracing_results, apriori, switches, model):
    """
    Convert raytracing/apriori structures into NN input lists.

    Returns
    -------
    A_list : list[np.ndarray]
    SWD_list : list[np.ndarray]
        Slant wet delay in mm.
    Napr_list : list[np.ndarray]
    Elev_list : list[np.ndarray]
    apr_model : str
    """

    n_vox = _validate_model_geometry(model)
    apr_key, apr_model = _get_apriori_key(switches)

    if apr_key not in apriori:
        raise KeyError(
            f"Missing apriori['{apr_key}'] required for NN tomography."
        )

    if len(raytracing_results) == 0:
        raise ValueError("raytracing_results is empty.")

    if len(apriori[apr_key]) != len(raytracing_results):
        raise ValueError(
            f"Apriori epoch count mismatch: apriori['{apr_key}'] has "
            f"{len(apriori[apr_key])} epochs, while raytracing_results has "
            f"{len(raytracing_results)} epochs."
        )

    A_list = []
    SWD_list = []
    Napr_list = []
    Elev_list = []

    for epoch, rt_epoch in enumerate(raytracing_results):
        if not isinstance(rt_epoch, dict):
            raise TypeError(
                f"raytracing_results[{epoch}] must be a dict, got {type(rt_epoch)}."
            )

        for key in ["A", "SD", "elev"]:
            if key not in rt_epoch:
                raise KeyError(f"raytracing_results[{epoch}] missing key '{key}'.")

        A = _validate_design_matrix(
            f"raytracing_results[{epoch}]['A']",
            rt_epoch["A"],
            n_vox,
        )

        SWD = _validate_numeric_vector(
            f"raytracing_results[{epoch}]['SD']",
            rt_epoch["SD"],
            expected_len=A.shape[0],
        )

        # Keep same convention as LSQ/tomography code: SD/SWD converted to mm.
        SWD = SWD * 1000.0

        Elev = _validate_numeric_vector(
            f"raytracing_results[{epoch}]['elev']",
            rt_epoch["elev"],
            expected_len=A.shape[0],
        )

        Napr = _validate_numeric_vector(
            f"apriori['{apr_key}'][{epoch}]",
            apriori[apr_key][epoch],
            expected_len=n_vox,
        )

        A_list.append(A)
        SWD_list.append(SWD)
        Napr_list.append(Napr)
        Elev_list.append(Elev)

    return A_list, SWD_list, Napr_list, Elev_list, apr_model


def _load_nn_module(model_type):
    train_module_name = _get_train_module_name(model_type)

    try:
        mt = importlib.import_module(train_module_name)
    except ImportError as exc:
        raise ImportError(
            f"Could not import NN module '{train_module_name}'. "
            "Make sure train_one.py/train_double.py is available on sys.path."
        ) from exc

    return mt, train_module_name


def _load_pretrained_model(mt, train_module_name, checkpoint_path, input_dim):
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained NN checkpoint not found: {checkpoint_path}"
        )

    if not hasattr(mt, "TomoNet"):
        raise RuntimeError(
            f"Module '{train_module_name}' does not define TomoNet class."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mt.TomoNet(input_dim).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"[INFO] Loaded pretrained NN model from: {checkpoint_path}")
    print(f"[INFO] NN device: {device}")

    return model

def _predict_with_model(mt, train_module_name, nn_model, A_list, SWD_list, Napr_list, Elev_list):
    import torch
    import numpy as np

    device = next(nn_model.parameters()).device

    predictions = []
    branch_choice = []
    resid_rms = []

    nn_model.eval()

    with torch.no_grad():
        for epoch, (A, SWD, Napr, Elev) in enumerate(
            zip(A_list, SWD_list, Napr_list, Elev_list)
        ):
            A_t = torch.tensor(np.asarray(A, dtype=np.float32), device=device)
            SWD_t = torch.tensor(np.asarray(SWD, dtype=np.float32), device=device)
            Napr_t = torch.tensor(np.asarray(Napr, dtype=np.float32), device=device)
            Elev_t = torch.tensor(np.asarray(Elev, dtype=np.float32), device=device)

            outputs = nn_model(A_t, SWD_t, Napr_t, Elev_t)

            if isinstance(outputs, (tuple, list)) and len(outputs) == 4:
                # double-branch model:
                # N_up, N_down, A_eff, P
                N_up, N_down, A_eff, _ = outputs

                N_up = torch.clamp(N_up, min=0.0)
                N_down = torch.clamp(N_down, min=0.0)

                SWD_up = torch.matmul(A_eff, N_up.unsqueeze(-1)).squeeze(-1)
                SWD_down = torch.matmul(A_eff, N_down.unsqueeze(-1)).squeeze(-1)

                rms_up = torch.sqrt(torch.mean((SWD_up - SWD_t) ** 2)).item()
                rms_down = torch.sqrt(torch.mean((SWD_down - SWD_t) ** 2)).item()

                if rms_up <= rms_down:
                    N_pred = N_up
                    chosen = "up"
                    rms = rms_up
                else:
                    N_pred = N_down
                    chosen = "down"
                    rms = rms_down

            elif isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                # one-branch model:
                # Npred, A_eff, P
                N_pred, A_eff, _ = outputs
                N_pred = torch.clamp(N_pred, min=0.0)

                SWD_pred = torch.matmul(A_eff, N_pred.unsqueeze(-1)).squeeze(-1)
                rms = torch.sqrt(torch.mean((SWD_pred - SWD_t) ** 2)).item()
                chosen = "single"

            elif isinstance(outputs, (tuple, list)):
                N_pred = outputs[0]
                N_pred = torch.clamp(N_pred, min=0.0)
                rms = np.nan
                chosen = "single"

            else:
                N_pred = outputs
                N_pred = torch.clamp(N_pred, min=0.0)
                rms = np.nan
                chosen = "single"

            predictions.append(N_pred.detach().cpu().numpy().astype(np.float32))
            branch_choice.append(chosen)
            resid_rms.append(rms)

            print(
                f"[INFO] NN epoch {epoch + 1}/{len(A_list)}: "
                f"chosen={chosen}, RMS={rms:.4f}"
            )

    return np.asarray(predictions), resid_rms, branch_choice


def _validate_prediction(Nw_pred, n_epochs, n_vox):
    Nw_pred = np.asarray(Nw_pred, dtype=float)

    if Nw_pred.size == 0:
        raise ValueError("NN prediction output is empty.")

    if Nw_pred.ndim == 1:
        if n_epochs != 1:
            raise ValueError(
                f"NN prediction is 1D with length {Nw_pred.size}, but "
                f"{n_epochs} epochs were processed."
            )
        Nw_pred = Nw_pred.reshape(1, -1)

    if Nw_pred.ndim != 2:
        raise ValueError(
            f"NN prediction must be 2D [epochs, voxels], got shape {Nw_pred.shape}."
        )

    if Nw_pred.shape[0] != n_epochs:
        raise ValueError(
            f"NN prediction epoch mismatch: got {Nw_pred.shape[0]}, "
            f"expected {n_epochs}."
        )

    if Nw_pred.shape[1] != n_vox:
        raise ValueError(
            f"NN prediction voxel mismatch: got {Nw_pred.shape[1]}, "
            f"expected {n_vox}."
        )

    if np.any(~np.isfinite(Nw_pred)):
        raise ValueError("NN prediction contains NaN or Inf.")

    return Nw_pred


def run_nn_model(raytracing_results, apriori, model, paths, switches, project_name):
    """
    Run pretrained TomoNet neural-network tomography.

    Parameters
    ----------
    raytracing_results : list[dict]
        Output from raytracing stage. Each epoch must contain:
            - A
            - SD
            - elev

    apriori : dict
        Apriori fields. Uses either:
            - apriori["Nw_ERA5"]
            - apriori["Nw_DETER"]

        depending on switches["aprModel"].

    model : dict
        Tomography model structure.

    paths : dict
        TomoNet path dictionary from tomo_conf.py.

    switches : dict
        Processing switches. Expected:
            switches["method"] = ["NN"]
            switches["aprModel"] = "ERA5" or "DETER"
            switches["NN"] = {
                "model_type": "double_branch",
                "checkpoint_name": "tomonet_model.pt",
                "results_name": "nn_output.npy",
            }

    project_name : str
        TomoNet project name.

    Returns
    -------
    output : dict
        Dictionary with NN tomography output:
            output["Nw"]
            output["Nwerr"]
            output["method"]
            output["model_type"]
            output["aprModel"]
            output["checkpoint"]
    """

    method = _as_scalar(switches.get("method", None))

    if method != "NN":
        warnings.warn(
            f"run_nn_model called while switches['method']={method!r}. "
            "Expected 'NN'. Continuing anyway."
        )

    nn_conf = _get_nn_config(switches)
    model_type = nn_conf["model_type"]
    checkpoint_name = nn_conf["checkpoint_name"]
    results_name = nn_conf["results_name"]

    out_dir, checkpoint_path, results_path = _get_output_paths(
        paths,
        project_name,
        checkpoint_name,
        results_name,
    )

    print("================================================")
    print("[INFO] Running TomoNet NN tomography")
    print("================================================")
    print(f"[INFO] NN model type: {model_type}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Results file: {results_path}")

    A_list, SWD_list, Napr_list, Elev_list, apr_model = _prepare_nn_inputs(
        raytracing_results,
        apriori,
        switches,
        model,
    )

    n_epochs = len(A_list)
    n_vox = Napr_list[0].size

    print(f"[INFO] NN epochs: {n_epochs}")
    print(f"[INFO] NN voxels: {n_vox}")

    mt, train_module_name = _load_nn_module(model_type)

    input_dim = Napr_list[0].size

    nn_model = _load_pretrained_model(
        mt,
        train_module_name,
        checkpoint_path,
        input_dim,
    )

    Nw_pred, resid_rms, branch_choice = _predict_with_model(
        mt,
        train_module_name,
        nn_model,
        A_list,
        SWD_list,
        Napr_list,
        Elev_list,
    )

    Nw_pred = _validate_prediction(
        Nw_pred,
        n_epochs=n_epochs,
        n_vox=n_vox,
    )

    output = {
        "Nw": Nw_pred,
        "Nwerr": None,
        "method": "NN",
        "model_type": model_type,
        "aprModel": apr_model,
        "checkpoint": str(checkpoint_path),
        "results_path": str(results_path),
        "resid_rms": resid_rms,
        "branch_choice": branch_choice,
    }

    print("================================================")
    print("[INFO] TomoNet NN tomography finished successfully")
    print(f"[INFO] Saved NN output: {results_path}")
    print("================================================")

    return output