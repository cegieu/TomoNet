# TomoNet
# TomoNet v1.1  Neural network - based GNSS tomography framework for atmospheric refractivity and water vapour reconstruction.
Designed by Adam Cegla  
ETH Zurich — Chair of Space Geodesy
acegla@ethz.ch

---
When using, please cite:

.....

---

# Overview

TomoNet v1.1 is a GNSS tomography framework designed for:
- atmospheric wet refractivity reconstruction
- water vapour tomography
- slant wet delay (SWD) processing
- ray tracing through numerical weather models
- neural-network-assisted tomography

The framework combines:
- LSQ tomography
- Kalman filtering tomography
- neural-network tomography

using:
- ERA5 meteorological fields
- GPT2/DETER apriori models
- GNSS SWD observations
- SP3 precise orbit products

---

# Main Folder Structure

```text
TomoNet/
│
├── CONF/
│   ├── stations_coordinates.txt
│   ├── undulation_file.npy
│   └── ...
│
├── DATA/
│   └── <PROJECT_NAME>/
│       ├── ATM/
│       ├── METEO/
│       ├── ORB/
│       ├── OUT/
│       ├── TOMO/
│       └── EXPORT/
│
├── Python/
│   │
│   ├── run.py
│   ├── tomo_conf.py
│   │
│   ├── modelprep/
│   ├── raytracer/
│   ├── tomo/
│   │   ├── tomography.py
│   │   └── nn_tomography.py
│   │
│   ├── read/
│   ├── others/
│   │
│   └── model_train/
│       ├── train_one.py
│       ├── train_double.py
│       ├── gather_data.py
│       ├── stats.py
│       ├── tomonet_model.pt
│       └── ...
│
├── README.md
└── LICENSE
```

---

# Main Components

## Operational Tomography Framework

Main scripts:
- Python/run.py
- Python/tomo_conf.py

Responsible for:
- reading GNSS observations
- loading ERA5/NWM data
- ray tracing
- tomography inversion
- operational NN tomography

Supports:
- LSQ
- KALMAN
- NN

---

## Neural Network Training Framework

Folder:
```text
Python/model_train/
```

Responsible for:
- training TomoNet neural networks
- validation statistics
- checkpoint generation
- NN model evaluation

Produces:
```text
tomonet_model.pt
```

which is later used operationally by:
```text
Python/tomo/nn_tomography.py
```

---

# Main Processing Modes

Defined in:
```python
switches["method"]
```

Options:

| Method | Description |
|---|---|
| LSQ | Least Squares tomography |
| KALMAN | Kalman filtering tomography |
| NN | Neural-network tomography using pretrained .pt model |

---

# Apriori Models

Defined in:
```python
switches["aprModel"]
```

Options:

| Model | Description |
|---|---|
| ERA5 | ERA5-derived refractivity |
| DETER | GPT2 deterministic refractivity |

---

# Neural Network Tomography

Operational NN tomography is executed through:
```text
Python/tomo/nn_tomography.py
```

The pretrained model:
```text
Python/model_train/tomonet_model.pt
```

is loaded automatically.

The NN receives:
- geometry matrix A
- SWD observations
- apriori refractivity
- elevation angles

and predicts:
- wet refractivity field Nw

---

# Neural Network Training

Training scripts:
```text
train_one.py
train_double.py
```

Supported architectures:
- single-branch model
- dual-branch model

The dual-branch model predicts:
- increasing refractivity solution
- decreasing refractivity solution

The branch with lower observation RMS is selected automatically.

---

# Input Data

## GNSS Stations

```text
CONF/stations_new.txt
```

## ERA5 Files

```text
DATA/<PROJECT>/METEO/
```

Example:
```text
ERA5_2020-7-7.nc
```

## GNSS Atmosphere Products

```text
DATA/<PROJECT>/ATM/
```

## SP3 Orbit Products

Automatically downloaded into:
```text
DATA/<PROJECT>/ORB/
```

---

# Main Output Files

## Operational Tomography

```text
DATA/<PROJECT>/OUT/out_REAL_LSQ_ERA5.npy
DATA/<PROJECT>/OUT/out_REAL_KALMAN_ERA5.npy
DATA/<PROJECT>/OUT/nn_output.npy
```

## Cached Intermediate Files

```text
modelNWM.npy
obs.npy
```

## Neural Network Checkpoint

```text
Python/model_train/tomonet_model.pt
```

---

# Main Configuration

All operational settings are controlled through:
```text
Python/tomo_conf.py
```
---

# Requirements

Recommended:
```bash
Python >= 3.10
```

Required packages:
```bash
pip install numpy scipy netCDF4 torch matplotlib
```

Optional:
```bash
pip install tqdm
```
---

# Running Operational Tomography

```bash
python run.py
```
---

# Running NN Training

```bash
python train_double.py
```
or:

```bash
python train_one.py
```

---

# Contact
Adam Cegla  
ETH Zurich  
Chair of Space Geodesy  
acegla@ethz.ch

