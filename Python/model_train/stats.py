#!/usr/bin/env python3
"""
TomoNet v.1.1
Designed by Adam Cegla at ETH Zurich, Chair of Space Geodesy.
15.05.2026

This software is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users should
refer to the GNU General Public License for more details.
"""

import numpy as np
import os
import matplotlib.pyplot as plt

def plot_histogram(data, title, save_path=None, bins=100):
    """Plot a histogram of a one-dimensional TomoNet validation variable.

    Inputs: data array, figure title, optional save_path, and number of bins.
    Output: a displayed or saved PNG figure.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color="steelblue", alpha=0.7)
    plt.title(title)
    plt.xlabel("Difference value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def plot_scatter(x, y, title, save_path=None):
    """Plot TomoNet reference/apriori values against predictions.

    Inputs: x and y arrays, figure title, and optional save_path.
    Output: a displayed or saved PNG figure.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=2, alpha=0.5)
    plt.title(title)
    plt.xlabel("True / Reference")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()

def analyze_validation_results(results_path, output_dir=None):
    """
    Analyze TomoNet validation outputs stored in an NPZ file.

    Input variables
    ---------------
    results_path : str
        Path to an NPZ file containing Npred/Npredict, Nref, Napr, and optionally resid_rms.
    output_dir : str or None
        Directory for PNG plots. If None, figures are displayed interactively.

    Output variables
    ----------------
    stats : dict
        Min, max, mean, standard deviation, and RMSE for Pred-Ref and Pred-Apr differences.
    """

    print(f"[INFO] Loading validation results from {results_path}...")
    data = np.load(results_path, allow_pickle=True)

    # --- Flexible key detection ---
    if "Npredict" in data:
        preds = np.concatenate(data["Npredict"])
    elif "Npred" in data:
        preds = np.concatenate(data["Npred"])
    else:
        raise KeyError("No Npredict or Npred key found in results file.")

    refs = np.concatenate(data["Nref"])
    aprs = np.concatenate(data["Napr"])

    # Optional residual RMS values
    resid_rms = data["resid_rms"] if "resid_rms" in data else None

    # --- Compute stats ---
    diff_pred_ref = preds - refs
    diff_pred_apr = preds - aprs

    stats = {
        "Pred - Ref": {
            "min": diff_pred_ref.min(),
            "max": diff_pred_ref.max(),
            "mean": diff_pred_ref.mean(),
            "std": diff_pred_ref.std(),
            "rmse": np.sqrt(np.mean(diff_pred_ref ** 2)),
        },
        "Pred - Apr": {
            "min": diff_pred_apr.min(),
            "max": diff_pred_apr.max(),
            "mean": diff_pred_apr.mean(),
            "std": diff_pred_apr.std(),
            "rmse": np.sqrt(np.mean(diff_pred_apr ** 2)),
        },
    }

    print("\n=== Validation Statistics ===")
    for key, vals in stats.items():
        print(f"[{key}] min={vals['min']:.4e}, max={vals['max']:.4e}, "
              f"mean={vals['mean']:.4e}, std={vals['std']:.4e}, rmse={vals['rmse']:.4e}")

    if resid_rms is not None:
        print(f"[Residual RMS] mean={np.mean(resid_rms):.4e}, std={np.std(resid_rms):.4e}")
    print("=============================\n")

    # --- Plots ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_histogram(diff_pred_ref, "Pred - Ref Differences",
                       os.path.join(output_dir, "hist_pred_ref.png"))
        plot_histogram(diff_pred_apr, "Pred - Apr Differences",
                       os.path.join(output_dir, "hist_pred_apr.png"))
        plot_scatter(refs, preds, "Reference vs Prediction",
                     save_path=os.path.join(output_dir, "scatter_pred_ref.png"))
        plot_scatter(aprs, preds, "Apriori vs Prediction",
                     save_path=os.path.join(output_dir, "scatter_pred_apr.png"))
        if resid_rms is not None:
            plot_histogram(resid_rms, "Residual RMS per sample",
                           os.path.join(output_dir, "hist_resid_rms.png"))

    return stats
