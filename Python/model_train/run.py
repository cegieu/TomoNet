#!/usr/bin/env python3
"""
TomoNet v.1.1
Designed by Adam Cegla at ETH Zurich, Chair of Space Geodesy.
15.05.2026

This software is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users should
refer to the GNU General Public License for more details.
"""

import argparse
import importlib
import os
import sys
import time
import warnings

import numpy as np

import gather_data as gd
import stats as ms


def parse_int_list(value: str):
    """Parse comma-separated integers and simple inclusive ranges such as '2022:2024'."""
    out = []
    for part in str(value).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            bounds = [int(x.strip()) for x in part.split(':')]
            if len(bounds) != 2:
                raise argparse.ArgumentTypeError(f"Invalid range '{part}'. Use start:end.")
            start, end = bounds
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("At least one integer must be provided.")
    return out


def thin_observations(A_list, SWD_list, Elev_list, drop_frac=0.10, seed=1234, min_keep=5):
    """
    Randomly remove observation rows from each epoch.

    Input variables
    ---------------
    A_list : list
        Tomography matrices with shape (n_observations, n_voxels).
    SWD_list : list
        Observation vectors aligned with rows of A_list.
    Elev_list : list
        Elevation vectors aligned with rows of A_list.
    drop_frac : float
        Fraction of observations to remove from each epoch. Must be in [0, 1).
    seed : int
        Random seed for reproducible thinning.
    min_keep : int
        Minimum number of observations to keep per epoch.

    Output variables
    ----------------
    A_out, SWD_out, Elev_out : lists
        Copies of the inputs with matching rows/elements removed.
    """
    if not (0.0 <= drop_frac < 1.0):
        raise ValueError("drop_frac must be in [0, 1).")
    if min_keep < 1:
        raise ValueError("min_keep must be >= 1.")
    rng = np.random.default_rng(seed)
    A_out, SWD_out, Elev_out = [], [], []
    for idx, (A, SWD, Elev) in enumerate(zip(A_list, SWD_list, Elev_list)):
        m = A.shape[0]
        if len(SWD) != m or len(Elev) != m:
            raise ValueError(f"Epoch {idx}: SWD/Elev lengths must match A rows ({m}).")
        if m <= min_keep:
            warnings.warn(f"Epoch {idx}: only {m} observations; keeping unchanged.")
            A_out.append(A); SWD_out.append(SWD); Elev_out.append(Elev)
            continue
        keep_m = max(min_keep, int(np.round(m * (1.0 - drop_frac))))
        keep_idx = np.sort(rng.choice(m, size=keep_m, replace=False))
        A_out.append(A[keep_idx, :])
        SWD_out.append(SWD[keep_idx])
        Elev_out.append(Elev[keep_idx])
    return A_out, SWD_out, Elev_out


def load_or_process_year(pathDATA, year, months, args):
    """Load cached yearly NPZ data or call gather_data.process_all_data."""
    cache_path = os.path.join(pathDATA, args.cache_template.format(year=year))
    if args.use_cache and os.path.exists(cache_path):
        print(f"[INFO] Loading precomputed data for {year}: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        required = ["A_scaled", "SWD_scaled", "Nref_scaled", "Napr_scaled"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Cache file {cache_path} is missing keys {missing}.")
        A_list = list(data["A_scaled"])
        SWD_list = list(data["SWD_scaled"])
        Nref_list = list(data["Nref_scaled"])
        Napr_list = list(data["Napr_scaled"])
        if "Elev" in data:
            Elev_list = list(data["Elev"])
        else:
            warnings.warn(f"Cache file {cache_path} has no 'Elev' key; using zeros as a fallback.")
            Elev_list = [np.zeros_like(np.asarray(swd)) for swd in SWD_list]
        return A_list, SWD_list, Nref_list, Napr_list, Elev_list

    print(f"[INFO] Processing raw data for {year}...")
    data_tuple = gd.process_all_data(
        pathDATA,
        years=[year],
        months=months,
        synthetic_filename=args.synthetic_filename,
        model_filename=args.model_filename,
        work_dirname=args.work_dirname,
        matrix_template=args.matrix_template,
        strict=not args.skip_bad_epochs,
    )
    if args.use_cache:
        np.savez_compressed(
            cache_path,
            A_scaled=np.array(data_tuple[0], dtype=object),
            SWD_scaled=np.array(data_tuple[1], dtype=object),
            Nref_scaled=np.array(data_tuple[2], dtype=object),
            Napr_scaled=np.array(data_tuple[3], dtype=object),
            Elev=np.array(data_tuple[4], dtype=object),
        )
        print(f"[OK] Saved {year} data to {cache_path}")
    return data_tuple


def extend_dataset(target_lists, source_tuple, swd_scale):
    A_t, SWD_t, Nref_t, Napr_t, Elev_t = target_lists
    A, SWD, Nref, Napr, Elev = source_tuple
    A_t.extend(list(A))
    SWD_t.extend([np.asarray(x, dtype=np.float32) * swd_scale for x in SWD])
    Nref_t.extend(list(Nref))
    Napr_t.extend(list(Napr))
    Elev_t.extend(list(Elev))


def validate_dataset(name, A, SWD, Nref, Napr, Elev):
    if not A:
        raise RuntimeError(f"{name} dataset is empty.")
    lengths = [len(A), len(SWD), len(Nref), len(Napr), len(Elev)]
    if len(set(lengths)) != 1:
        raise ValueError(f"{name} list lengths differ: A,SWD,Nref,Napr,Elev={lengths}")
    for i, (Ai, swd, nref, napr, elev) in enumerate(zip(A, SWD, Nref, Napr, Elev)):
        if Ai.shape[0] != len(swd) or Ai.shape[0] != len(elev):
            raise ValueError(f"{name} epoch {i}: A rows, SWD length, and Elev length must match.")
        if Ai.shape[1] != np.asarray(nref).size or Ai.shape[1] != np.asarray(napr).size:
            raise ValueError(f"{name} epoch {i}: A columns must match Nref and Napr sizes.")
        if not np.all(np.isfinite(np.asarray(swd))):
            raise ValueError(f"{name} epoch {i}: SWD contains NaN or Inf.")


def build_parser():
    parser = argparse.ArgumentParser(description="Run TomoNet v.1.1 MLP tomography training and validation.")
    parser.add_argument("--model", choices=["one_branch", "double_branch"], default="double_branch", help="TomoNet architecture to train.")
    parser.add_argument("--pathDATA", required=True, help="Root directory with tomography data and output files.")
    parser.add_argument("--train-years", type=parse_int_list, required=True, help="Training years, e.g. '2022,2023' or '2022:2024'.")
    parser.add_argument("--val-years", type=parse_int_list, required=True, help="Validation years, e.g. '2023'.")
    parser.add_argument("--months", type=parse_int_list, default=list(range(1, 13)), help="Months to process, e.g. '1:12' or '8'.")
    parser.add_argument("--results-name", default="validation_results.npz", help="Output validation results filename.")
    parser.add_argument("--plots-dir", default="validation_plots", help="Validation plot output directory name.")
    parser.add_argument("--synthetic-filename", default=gd.DEFAULT_SYNTHETIC_FILE, help="Monthly synthetic/reference MATLAB filename.")
    parser.add_argument("--model-filename", default=gd.DEFAULT_MODEL_FILE, help="Apriori MATLAB filename inside WORK.")
    parser.add_argument("--work-dirname", default=gd.DEFAULT_WORK_DIR, help="WORK subdirectory name.")
    parser.add_argument("--matrix-template", default="amtrix_{epoch}.mat", help="Epoch matrix filename template containing '{epoch}'.")
    parser.add_argument("--cache-template", default="{year}_data.npz", help="Yearly cache filename template.")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable yearly NPZ cache loading/saving.")
    parser.add_argument("--skip-bad-epochs", action="store_true", help="Warn and skip bad epochs/months instead of stopping.")
    parser.add_argument("--thin-drop-frac", type=float, default=0.0, help="Fraction of validation observations to remove before evaluation.")
    parser.add_argument("--thin-seed", type=int, default=20241215, help="Random seed for validation thinning.")
    parser.add_argument("--swd-scale", type=float, default=1000.0, help="Multiplicative scale applied to SD/SWD observations.")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor c passed to evaluate_model.")
    parser.add_argument("--no-structured-mean-prior", action="store_true", help="Do not replace training Napr with element-wise mean of Nref_train.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    pathDATA = os.path.abspath(args.pathDATA)
    if not os.path.isdir(pathDATA):
        raise NotADirectoryError(f"pathDATA does not exist: {pathDATA}")

    train_module_name = "train_one" if args.model == "one_branch" else "train_double"
    mt = importlib.import_module(train_module_name)
    print(f"[INFO] Selected model: {args.model} ({train_module_name}.py)")

    A_train, SWD_train, Nref_train, Napr_train, Elev_train = [], [], [], [], []
    A_val, SWD_val, Nref_val, Napr_val, Elev_val = [], [], [], [], []

    for year in args.train_years:
        extend_dataset((A_train, SWD_train, Nref_train, Napr_train, Elev_train), load_or_process_year(pathDATA, year, args.months, args), args.swd_scale)
    for year in args.val_years:
        extend_dataset((A_val, SWD_val, Nref_val, Napr_val, Elev_val), load_or_process_year(pathDATA, year, args.months, args), args.swd_scale)

    validate_dataset("training", A_train, SWD_train, Nref_train, Napr_train, Elev_train)
    validate_dataset("validation", A_val, SWD_val, Nref_val, Napr_val, Elev_val)

    print("\n[OK] Data loaded and split successfully.")
    print(f"[INFO] Training samples:   {len(A_train)}")
    print(f"[INFO] Validation samples: {len(A_val)}")

    if not args.no_structured_mean_prior:
        print("\n[INFO] Computing element-wise structured mean Napr from Nref_train fields...")
        flat_refs = [np.asarray(n, dtype=np.float32).ravel() for n in Nref_train]
        mean_field = np.mean(np.stack(flat_refs, axis=0), axis=0)
        Napr_train = [mean_field.copy() for _ in Nref_train]
        print(f"[OK] Training Napr replaced with structured mean prior. Shape={mean_field.shape}")
        print("[INFO] Validation Napr left unchanged.")

    gd.summarize_data(A_train, SWD_train, Nref_train, Napr_train)

    print("\n[INFO] Starting model training...")
    start = time.time()
    model = mt.model_train(A_train, SWD_train, Nref_train, Napr_train, Elev_train, A_val, SWD_val, Nref_val, Napr_val, Elev_val)
    print(f"[INFO] Training call finished after {(time.time() - start)/60:.2f} minutes.")

    eval_A, eval_SWD, eval_Elev = A_val, SWD_val, Elev_val
    if args.thin_drop_frac > 0:
        print(f"\n[INFO] Thinning validation observations: drop_frac={args.thin_drop_frac}")
        eval_A, eval_SWD, eval_Elev = thin_observations(A_val, SWD_val, Elev_val, drop_frac=args.thin_drop_frac, seed=args.thin_seed)

    print("\n[INFO] Evaluating predictions and computing statistics...")
    results_path = os.path.join(pathDATA, args.results_name)
    mt.evaluate_model(model, eval_A, eval_SWD, Nref_val, Napr_val, eval_Elev, c=args.scale_factor, save_results=True, results_path=results_path)

    output_dir = os.path.join(pathDATA, args.plots_dir)
    ms.analyze_validation_results(results_path, output_dir=output_dir)
    print("\n[OK] All steps completed successfully.")
    print(f"[INFO] Results: {results_path}")
    print(f"[INFO] Plots:   {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
