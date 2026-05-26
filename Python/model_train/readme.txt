# TomoNet v1.1 - training scripts

## Neural Network Tomography for Atmospheric Water Vapour Reconstruction

**TomoNet v1.1** is a machine-learning tomography framework designed for atmospheric refractivity and water vapour reconstruction from GNSS Slant Wet Delay (SWD) observations. The model_train section contains routines being a part and for a strict use within TomoNet framework. Support for other types of tomography software is not provided.

Developed by **Adam Cegła**
ETH Zurich — Chair of Space Geodesy

---

## Disclaimer

This software is distributed **without any warranty**.
It is provided for scientific and research purposes only.

See the GNU General Public License (GPL) for details.

---

# Features

* GNSS tomography using deep neural networks
* Supports:

  * **Single-branch model** (`train_one.py`)
  * **Dual-branch model** (`train_double.py`)
* Elevation-aware observation weighting
* Automatic validation and statistics generation
* Observation thinning support
* MATLAB `.mat` input compatibility
* Automatic consistency checks and error reporting

---

# Repository Structure

```text
TomoNet/
│
├── run.py                # Main execution script
├── run_local.py          # Main script for direct execution from IDE 
├── gather_data.py        # Data loading and preprocessing
├── train_one.py          # Single-branch TomoNet model
├── train_double.py       # Dual-branch TomoNet model
├── stats.py              # Validation analysis and plotting
│
├── README.md
└── LICENSE
```

---

# Requirements

## Python Version

Recommended:

```bash
Python >= 3.10
```

---

## Required Python Packages

Install dependencies using:

```bash
pip install numpy scipy matplotlib torch
```

Optional but recommended:

```bash
pip install tqdm
```

---

# Input Data Structure

The software expects tomography data organized by:

```text
pathDATA/
│
├── 2023/ ##YEAR
│   ├── 1/ ##MONTH
│   │   ├── tomo_output.mat
│   │   └── WORK/
│   │       ├── modelTOMO.mat
│   │       ├── amtrix_1.mat
│   │       ├── amtrix_2.mat
│   │       └── ...
```

---

# Required MATLAB Variables

## File: `tomo_output.mat`

Expected variable (output from LSQ-based tomography processing):

```matlab
output.Nw
```

---

## File: `modelTOMO.mat`

Expected variables (apriori and reference wet refractivity fields)::

```matlab
apriori.Nw_DETER #apriori tomography model
apriori.Nw_ERA5  #ray tracing model
```

---

## File: `amtrix_X.mat` #X is a number of the epoch

# Input Variables

## Main Inputs

| Variable | Description                                     |
| -------- | ----------------------------------------------- |
| `A`      | Geometry matrix relating voxels to observations |
| `SWD`    | Slant Wet Delay observations                    |
| `Nref`   | Reference refractivity field                    |
| `Napr`   | Apriori refractivity field                      |
| `Elev`   | Satellite elevation angles                      |

---

# Output Variables

## Validation Output (`validation_results.npz`)

| Variable             | Description                       |
| -------------------- | --------------------------------- |
| `Npred` / `Npredict` | Predicted refractivity field      |
| `Nref`               | Reference field                   |
| `Napr`               | Apriori field                     |
| `resid_rms`          | RMS residuals                     |
| `branch_choice`      | Selected branch (dual model only) |

---

# Running the Software

## Basic Usage

```bash
python run.py
```

---

# Command Line Options

## Select Model Type

### Single-branch model

```bash
python run.py --model one_branch
```

### Dual-branch model

```bash
python run.py --model double_branch
```

---

## Define Dataset Path

```bash
python run.py --pathDATA /path/to/dataset
```

---

## Define Training and Validation Years

```bash
python run.py \
    --train-years 2022 2023 \
    --val-years 2024
```

---

## Define Validation Output Filename

```bash
python run.py --results-name validation_results_2024.npz
```

---

## Define Input MATLAB File Name

```bash
python run.py \
    --synthetic-file tomo_output.mat
```
"/scratch/path"
---
"
# Observation Thinning

The software supports random observation thinning for robustness testing.

Example:

```python
thin_observations(
    A_list,
    SWD_list,
    Elev_list,
    drop_frac=0.10
)
```

This removes 10% of observations randomly.

---

# Validation and Statistics

Validation statistics and plots are generated automatically.

Output directory:

```text
validation_plots/
```

Generated plots include:

* Prediction vs Reference scatter plots
* Histograms of residuals
* RMS statistics
* Prediction vs Apriori comparison

---

# Model Description

## Single-Branch Model

The single-branch TomoNet predicts a direct refractivity field correction:

```text
Npred = f(A, SWD, Napr, Elev)
```

Advantages:

* Faster training
* Lower memory usage
* Stable inference

---

## Dual-Branch Model

The dual-branch TomoNet predicts:

* Increasing refractivity solution
* Decreasing refractivity solution

The branch with lower observation residual RMS is selected.

Advantages:

* Better ambiguity handling
* Improved robustness

---

# GPU Support

The software automatically detects CUDA-compatible GPUs.

If CUDA is available:

```text
[INFO] Training on cuda
```

Otherwise CPU mode is used.

---

# Safety Checks Included

TomoNet v1.1 includes:

* Missing variable detection
* Invalid matrix shape checks
* Sparse/dense matrix consistency handling
* Elevation data verification
* Observation count consistency checks
* Automatic fallback warnings

---

# Example Workflow

```bash
python run.py \
    --model double_branch \
    --pathDATA /scratch/AWARE/TOMONNCAL \
    --train-years 2023 \
    --val-years 2023 \
    --results-name validation_results.npz
```



# Citation

If you use this software in scientific work, please cite:

```

```

---

# License

GNU General Public License (GPL)

---

# Contact

Adam Cegła
ETH Zurich
Chair of Space Geodesy, Robert-Gnehm-Weg 15, 8093 Zurich, Switzerland
acegla@ethz.ch

GitHub Issues are preferred for bug reports and feature requests.




