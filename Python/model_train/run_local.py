#!/usr/bin/env python3
# ============================================================
# TomoNet v1.1
#
# Neural Network GNSS Tomography Framework
#
# Developed by:
# Adam Cegła
# ETH Zurich — Chair of Space Geodesy
# 15.05.2026

# Distributed WITHOUT ANY WARRANTY.
# See GNU GPL license for details.
# ============================================================

import os
import importlib
import numpy as np

import gather_data as gd
import stats as ms

# ============================================================
# === USER SETTINGS
# ============================================================

# Select model:
# "one_branch"  -> train_one.py
# "double_branch" -> train_double.py
MODEL_TYPE = "double_branch"

# Main data directory
pathDATA = "/scratch/AWARE/TOMONNCAL"

# Training years
TRAIN_YEARS = [2023]

# Validation years
VAL_YEARS = [2023]

# Months to process
MONTHS = [8]

# MATLAB filenames
MODEL_FILE = "modelTOMO.mat"

# Validation output filename
RESULTS_NAME = "validation_results.npz"

# Output plot directory
PLOTS_DIR = "validation_plots"

# Observation thinning
USE_THINNING = False
THINNING_FRAC = 0.10

# ============================================================
# === IMPORT MODEL
# ============================================================

if MODEL_TYPE == "one_branch":
    train_module_name = "train_one"

elif MODEL_TYPE == "double_branch":
    train_module_name = "train_double"

else:
    raise ValueError(
        f"Invalid MODEL_TYPE='{MODEL_TYPE}'. "
        f"Expected 'one_branch' or 'double_branch'."
    )

print(f"[INFO] Using model: {MODEL_TYPE}")

mt = importlib.import_module(train_module_name)

# ============================================================
# === STORAGE
# ============================================================

A_train, SWD_train, Nref_train, Napr_train, Elev_train = [], [], [], [], []
A_val, SWD_val, Nref_val, Napr_val, Elev_val = [], [], [], [], []

# ============================================================
# === LOAD TRAINING DATA
# ============================================================

print("\n================================================")
print("Loading TRAINING data")
print("================================================")

for year in TRAIN_YEARS:

    print(f"\n[INFO] Processing training year {year}")

    try:

        (
            A_list,
            SWD_list,
            Nref_list,
            Napr_list,
            Elev_list,
        ) = gd.process_all_data(
            pathDATA=pathDATA,
            years=[year],
            months=MONTHS,
        )

        A_train.extend(A_list)
        SWD_train.extend([x * 1000 for x in SWD_list])
        Nref_train.extend(Nref_list)
        Napr_train.extend(Napr_list)
        Elev_train.extend(Elev_list)

    except Exception as e:
        print(f"[ERROR] Failed loading training year {year}")
        print(e)

# ============================================================
# === LOAD VALIDATION DATA
# ============================================================

print("\n================================================")
print("Loading VALIDATION data")
print("================================================")

for year in VAL_YEARS:

    print(f"\n[INFO] Processing validation year {year}")

    try:

        (
            A_list,
            SWD_list,
            Nref_list,
            Napr_list,
            Elev_list,
        ) = gd.process_all_data(
            pathDATA=pathDATA,
            years=[year],
            months=MONTHS,
        )

        A_val.extend(A_list)
        SWD_val.extend([x * 1000 for x in SWD_list])
        Nref_val.extend(Nref_list)
        Napr_val.extend(Napr_list)
        Elev_val.extend(Elev_list)

    except Exception as e:
        print(f"[ERROR] Failed loading validation year {year}")
        print(e)

# ============================================================
# === BASIC CHECKS
# ============================================================

if len(A_train) == 0:
    raise RuntimeError("No training data loaded.")

if len(A_val) == 0:
    raise RuntimeError("No validation data loaded.")

print("\n================================================")
print("Dataset summary")
print("================================================")

print(f"Training samples:   {len(A_train)}")
print(f"Validation samples: {len(A_val)}")

gd.summarize_data(A_train, SWD_train, Nref_train, Napr_train)

# ============================================================
# === OPTIONAL STRUCTURED PRIOR
# ============================================================

print("\n[INFO] Computing structured mean prior...")

flat_refs = [n.flatten() for n in Nref_train]
ref_stack = np.stack(flat_refs, axis=0)
mean_field = np.mean(ref_stack, axis=0)

Napr_train = [mean_field.copy() for _ in Nref_train]

print("[OK] Structured mean prior applied.")

# ============================================================
# === TRAIN MODEL
# ============================================================

print("\n================================================")
print("Starting training")
print("================================================")

model = mt.model_train(
    A_train,
    SWD_train,
    Nref_train,
    Napr_train,
    Elev_train,
    A_val,
    SWD_val,
    Nref_val,
    Napr_val,
    Elev_val
)

# ============================================================
# === VALIDATION
# ============================================================

print("\n================================================")
print("Evaluating model")
print("================================================")

results_path = os.path.join(pathDATA, RESULTS_NAME)

mt.evaluate_model(
    model,
    A_val,
    SWD_val,
    Nref_val,
    Napr_val,
    Elev_val,
    c=1.0,
    save_results=True,
    results_path=results_path
)

# ============================================================
# === STATISTICS & PLOTS
# ============================================================

print("\n================================================")
print("Generating statistics and plots")
print("================================================")

output_dir = os.path.join(pathDATA, PLOTS_DIR)

ms.analyze_validation_results(
    results_path,
    output_dir=output_dir
)

print("\n================================================")
print("TomoNet processing finished successfully.")
print("================================================")


