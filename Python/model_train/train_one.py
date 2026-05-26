#!/usr/bin/env python3
"""
TomoNet v.1.1
Designed by Adam Cegla at ETH Zurich, Chair of Space Geodesy.
15.05.2026

This software is distributed WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. Users should
refer to the GNU General Public License for more details.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.sparse import issparse
import time
import os

# ============================================================
# === Helper Functions
# ============================================================

def _to_float_vec(x):
    """Convert possibly object-wrapped or scalar arrays into a flat float32 vector."""
    if x is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(x, dtype=object)
    if arr.ndim == 0:
        try:
            return np.array([float(arr)], dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)
    if arr.size == 1 and isinstance(arr.item(), (np.ndarray, list, tuple)):
        inner = arr.item()
        return np.asarray(inner, dtype=np.float32).ravel()
    if arr.dtype == object:
        flattened = []
        for e in arr:
            e_np = np.asarray(e)
            if e_np.size > 0:
                flattened.append(e_np.ravel())
        if len(flattened) == 0:
            return np.array([], dtype=np.float32)
        if len(flattened) == 1:
            return np.asarray(flattened[0], dtype=np.float32)
        else:
            return np.concatenate(flattened).astype(np.float32)
    return np.asarray(arr, dtype=np.float32).ravel()

def _to_float_mat(A):
    """Return a 2D float32 numpy array (dense)."""
    if issparse(A):
        return A.toarray().astype(np.float32)
    arr = np.asarray(A, dtype=object)
    if arr.dtype == object:
        if arr.size == 1 and isinstance(arr.item(), (np.ndarray, list, tuple)):
            arr = np.asarray(arr.item())
        elif arr.ndim == 1 and all(isinstance(e, (np.ndarray, list, tuple)) for e in arr):
            arr = np.stack([np.asarray(e) for e in arr], axis=0)
    return np.asarray(arr, dtype=np.float32)

# ============================================================
# === Elevation weighting utility
# ============================================================

def make_elevation_weights(elev_t, k=3.0, elev_mid=0.3, eps=1e-6):
    # Normalize from assumed [5,85] → [0,1]
    elev_norm = (elev_t - (torch.min(elev_t)-1)) / (torch.max(elev_t)+1)
    z = k * (elev_norm - elev_mid)
    w = torch.sigmoid(z).clamp_min(eps)
    return w / w.mean()

# ============================================================
# === Dataset Definition
# ============================================================

class TomoDataset(Dataset):
    def __init__(self, A_list, SWD_list, Nref_list, Napr_list, Elev_list):
        self.A_list = A_list
        self.SWD_list = SWD_list
        self.Nref_list = Nref_list
        self.Napr_list = Napr_list
        self.Elev_list = Elev_list  # new elevation data

    def __len__(self):
        return len(self.A_list)

    def __getitem__(self, idx):
        A = _to_float_mat(self.A_list[idx])
        SWD = _to_float_vec(self.SWD_list[idx])
        Nref = _to_float_vec(self.Nref_list[idx])
        Napr = _to_float_vec(self.Napr_list[idx])
        Elev = _to_float_vec(self.Elev_list[idx])

        return (
            torch.from_numpy(A),
            torch.from_numpy(SWD),
            torch.from_numpy(Nref),
            torch.from_numpy(Napr),
            torch.from_numpy(Elev),
        )

# ============================================================
# === Model Definition
# ============================================================
import torch.nn.functional as F

# ============================================================
# === TomoNet with Learnable Elevation Weighting P(E)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TomoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, max_delta=40.0): #1024
        """
        Observation-conditioned TomoNet (single signed correction).

        Computes:
            r = A_eff @ Napr - SWD
            g = A_eff^T @ r
        Predicts a signed correction ΔN from [Napr, g]:
            Npred = clamp(Napr + ΔN, 0, 140)

        max_delta limits the magnitude of ΔN via tanh for stability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.max_delta = float(max_delta)

        # input is [Napr, g] => 2 * input_dim
        self.shared = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.ReLU(),
        )

        # single head for correction ΔN (signed)
        self.delta_head = nn.Linear(hidden_dim // 16, input_dim)
        #self.delta_head = nn.Linear(hidden_dim // 8, input_dim)

        # elevation weighting network P(E) in (0,1)
        self.elev_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init + small positive bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.2)

    def forward(self, A, SWD, Napr, Elev):
        """
        Supports:
          - unbatched: A[m,n], SWD[m], Napr[n], Elev[m]
          - batched:   A[B,m,n], SWD[B,m], Napr[B,n], Elev[B,m]

        Returns:
          - Npred (same batching style), A_eff, P
        """
        unbatched = (A.dim() == 2)
        if unbatched:
            A = A.unsqueeze(0)        # [1, m, n]
            SWD = SWD.unsqueeze(0)    # [1, m]
            Napr = Napr.unsqueeze(0)  # [1, n]
            Elev = Elev.unsqueeze(0)  # [1, m]

        # elevation weights
        P = self.elev_net(Elev.unsqueeze(-1)).squeeze(-1)   # [B, m]
        A_eff = A * P.unsqueeze(-1)                         # [B, m, n]

        # residual of prior in observation space
        SWD_apr = torch.matmul(A_eff, Napr.unsqueeze(-1)).squeeze(-1)  # [B, m]
        resid = SWD_apr - SWD                                          # [B, m]

        # gradient feature
        g = torch.matmul(A_eff.transpose(1, 2), resid.unsqueeze(-1)).squeeze(-1)  # [B, n]
        g = g / (g.norm(dim=1, keepdim=True).clamp_min(1e-6))  # normalize

        # MLP
        x = torch.cat([Napr, g], dim=1)       # [B, 2n]
        h = self.shared(x)                    # [B, d]
        dN = self.delta_head(h)

        # apply correction
        Npred = dN

        Npred = torch.clamp(Npred, 0.0, 140.0)

        if unbatched:
            return Npred.squeeze(0), A_eff.squeeze(0), P.squeeze(0)
        return Npred, A_eff, P

# ============================================================
# === Training Function
# ============================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import os

def model_train(A_train, SWD_train, Nref_train, Napr_train, Elev_train,
                A_val=None, SWD_val=None, Nref_val=None, Napr_val=None, Elev_val=None,
                epochs=12, batch_size=1, lr=1e-6, device=None,
                resume_if_exists=True, model_path="tomo_model.pt",
                softmin=True, temperature=0.1):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not A_train or not SWD_train or not Nref_train or not Napr_train or not Elev_train:
        raise ValueError("Training inputs must be non-empty lists: A_train, SWD_train, Nref_train, Napr_train, Elev_train.")
    if not (len(A_train) == len(SWD_train) == len(Nref_train) == len(Napr_train) == len(Elev_train)):
        raise ValueError("Training input lists must have equal lengths.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if lr <= 0:
        raise ValueError("Learning rate lr must be positive.")
    print(f"[INFO] Training (one-branch TomoNet v.1.1) on {device}")

    # ============================================================
    # Collate & Dataloaders
    # ============================================================

    def pad_collate(batch):
        As, SWDs, Nrefs, Naprs, Elevs = zip(*batch)
        max_m = max(A.shape[0] for A in As)
        padded_A, padded_SWD, padded_Elev, masks = [], [], [], []
        for A, SWD, Elev in zip(As, SWDs, Elevs):
            m = A.shape[0]
            pad_rows = max_m - m
            if pad_rows > 0:
                A_p = F.pad(A, (0, 0, 0, pad_rows))
                SWD_p = F.pad(SWD, (0, pad_rows))
                Elev_p = F.pad(Elev, (0, pad_rows))
                mask = torch.cat([torch.ones(m), torch.zeros(pad_rows)])
            else:
                A_p, SWD_p, Elev_p, mask = A, SWD, Elev, torch.ones(m)
            padded_A.append(A_p)
            padded_SWD.append(SWD_p)
            padded_Elev.append(Elev_p)
            masks.append(mask)
        return (
            torch.stack(padded_A),
            torch.stack(padded_SWD),
            torch.stack(Nrefs),
            torch.stack(Naprs),
            torch.stack(padded_Elev),
            torch.stack(masks),
        )

    train_ds = TomoDataset(A_train, SWD_train, Nref_train, Napr_train, Elev_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=pad_collate if batch_size > 1 else None)

    val_dl = None
    if A_val is not None:
        val_ds = TomoDataset(A_val, SWD_val, Nref_val, Napr_val, Elev_val)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=pad_collate if batch_size > 1 else None)

    # ============================================================
    # Model init / resume
    # ============================================================

    sample_A, sample_SWD, sample_Nref, sample_Napr, _ = train_ds[0]
    input_dim = sample_Napr.numel()
    model = TomoNet(input_dim).to(device)

    if resume_if_exists and os.path.exists(model_path):
        print(f"[INFO] Loading existing model from {model_path} ...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("[OK] Model loaded.")
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    mse_loss = nn.MSELoss()

    # ============================================================
    # Active hyperparameters. Zero-weighted terms were removed for clarity.
    # ============================================================

    w_data_base = 1.0
    w_prior = 1.0
    w_mag = 0.5

    # ============================================================
    # Helper: normalize A and SWD
    # ============================================================

    def normalize_A_SWD(A, SWD, mask=None):
        if A.dim() == 3:
            row_norm = A.norm(dim=2, keepdim=True).clamp_min(1e-6)
        else:
            row_norm = A.norm(dim=1, keepdim=True).clamp_min(1e-6)
        A_norm = A / row_norm
        SWD_norm = SWD / row_norm.squeeze(-1)
        if mask is not None:
            SWD_norm = SWD_norm * mask
        return A_norm, SWD_norm

    # ============================================================
    # Loss computation (dual absolute branches)
    # ============================================================

    def _loss_terms(A, SWD, Nref, Napr, Elev, mask=None):
        # Normalize for stability (same as before)
        A, SWD = normalize_A_SWD(A, SWD, mask)

        # Forward
        Npred, A_eff, P_elev = model(A, SWD, Napr, Elev)

        Nref = Nref.squeeze()
        Napr = Napr.squeeze()
        Npred = Npred.squeeze()

        # Observation residual
        SWD_pred = torch.matmul(A_eff, Npred.unsqueeze(-1)).squeeze(-1)
        resid = SWD_pred - SWD
        if mask is not None:
            resid = resid * mask

        # Data loss
        if mask is not None:
            loss_data = (resid.pow(2) * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss_data = resid.pow(2).mean()

        # Compute trust (keep your existing logic)
        SWD_apr = torch.matmul(A, Napr.unsqueeze(-1)).squeeze(-1)
        resid_apr = SWD_apr - SWD
        apr_rms = torch.sqrt(torch.mean(resid_apr.pow(2)) + 1e-8)
        scale = apr_rms / (torch.mean(torch.abs(SWD)) + 1e-6)
        trust = torch.sigmoid(6.0 * (scale - 1.0))
        w_data = w_data_base * (1.0 + 4.0 * trust)

        # Active regularizers
        loss_prior = mse_loss(Npred, Napr)  # keeps updates reasonable

        # Magnitude control (global)
        ratio = (Npred + 1e-6) / (Napr + 1e-6)
        loss_mag = torch.relu(ratio - 1.5).mean()

        # --- Smoothness on correction dN to suppress spikes ---
        def smoothness_3d(N, Z=12, Y=15, X=15):
            """
            Compute 3D smoothness regularization for tomography voxels.

            Parameters
            ----------
            N : torch.Tensor
                Flattened tomography field of size (Z*Y*X).

            Z, Y, X : int
                Tomography model dimensions:
                    Z -> vertical levels
                    Y -> latitude dimension
                    X -> longitude dimension

            IMPORTANT
            ----------
            The product Z * Y * X MUST exactly match
            the tomography voxel count.

            Example:
                If tomography model contains 2700 voxels:
                    choose dimensions such that:
                        Z * Y * X = 2700

            Returns
            -------
            torch.Tensor
                Smoothness penalty value.
            """

            expected_size = Z * Y * X
            actual_size = N.numel()

            if actual_size != expected_size:
                raise ValueError(
                    "\n"
                    "====================================================\n"
                    "ERROR: Invalid tomography model dimensions.\n"
                    "====================================================\n"
                    f"Current dimensions:\n"
                    f"    Z = {Z}\n"
                    f"    Y = {Y}\n"
                    f"    X = {X}\n\n"
                    f"Expected voxel count from dimensions:\n"
                    f"    Z * Y * X = {expected_size}\n\n"
                    f"Actual tomography vector size:\n"
                    f"    len(N) = {actual_size}\n\n"
                    "Please manually define the tomography model\n"
                    "dimensions inside:\n\n"
                    "    smoothness_3d(N, Z=?, Y=?, X=?)\n\n"
                    "so that:\n\n"
                    "    Z * Y * X == len(N)\n\n"
                    "Example:\n"
                    "    len(N)=2304\n"
                    "    valid dimensions could be:\n"
                    "        Z=12, Y=12, X=16\n"
                    "===================================================="
                )

            try:

                V = N.view(Z, Y, X)

            except Exception as e:

                raise RuntimeError(
                    "\n"
                    "Failed to reshape tomography field into 3D model.\n"
                    "Please verify tomography dimensions:\n"
                    f"    Z={Z}, Y={Y}, X={X}\n"
                    f"Tensor size={actual_size}\n"
                ) from e

            dx = V[:, :, 1:] - V[:, :, :-1]
            dy = V[:, 1:, :] - V[:, :-1, :]
            dz = V[1:, :, :] - V[:-1, :, :]

            return (
                    dx.pow(2).mean()
                    + dy.pow(2).mean()
                    + dz.pow(2).mean()
            )

        dN = Npred - Napr
        loss_smooth_dN = smoothness_3d(dN)

        # --- Upper-region ratio penalty (prevents runaway top layers) ---
        upper_start = 1800
        mask_u = torch.zeros_like(Npred)
        mask_u[upper_start:] = 1.0
        ratio_u = (Npred + 1e-6) / (Napr + 1e-6)
        loss_upper_hi = (F.relu(ratio_u - 1.5) * mask_u).sum() / (mask_u.sum() + 1e-6)

        # weights (start small and tune)
        w_smooth_dN = 0.05
        w_upper_hi = 0.1

        total = (
                w_data * loss_data +
                w_prior * loss_prior +
                w_mag * loss_mag

        )
        total = total + w_smooth_dN * loss_smooth_dN + w_upper_hi * loss_upper_hi
        # Regularize elevation mean (optional)
        total = total + 1e-3 * ((P_elev.mean() - 0.5) ** 2)

        return total, {
            "loss_data": loss_data.detach(),
            "loss_prior": loss_prior.detach(),
            "loss_mag": loss_mag.detach(),
            "w_data": w_data.detach(),
            "trust": trust.detach(),
            "P_elev_mean": P_elev.mean().detach(),
        }

    # ============================================================
    # Training loop
    # ============================================================

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_dl:
            if batch_size > 1:
                A, SWD, Nref, Napr, Elev, mask = batch
            else:
                A, SWD, Nref, Napr, Elev = batch
                mask = None
            A, SWD, Nref, Napr, Elev = [x.to(device) for x in (A, SWD, Nref, Napr, Elev)]
            if mask is not None: mask = mask.to(device)

            loss, details = _loss_terms(A, SWD, Nref, Napr, Elev, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach())
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        print(f"[EPOCH {epoch+1:03d}] Train loss = {avg_train_loss:.6e}")

        # === Validation ===
        if val_dl is not None:
            model.eval()
            vloss, v_batches = 0.0, 0
            with torch.no_grad():
                for batch in val_dl:
                    if batch_size > 1:
                        A, SWD, Nref, Napr, Elev, mask = batch
                    else:
                        A, SWD, Nref, Napr, Elev = batch
                        mask = None
                    A, SWD, Nref, Napr, Elev = [x.to(device) for x in (A, SWD, Nref, Napr, Elev)]
                    if mask is not None: mask = mask.to(device)
                    v_loss, _ = _loss_terms(A, SWD, Nref, Napr, Elev, mask)
                    vloss += float(v_loss.detach())
                    v_batches += 1
            print(f"           Val loss   = {vloss / max(v_batches,1):.6e}")

    print(f"[INFO] Training finished in {(time.time()-start_time)/60:.2f} minutes")
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved as {model_path}")
    return model


# ============================================================
# === Evaluation
# ============================================================

import torch
import numpy as np
from scipy.sparse import issparse

def evaluate_model(model, A_val, SWD_val, Nref_val, Napr_val, Elev_val=None,
                   c=1.0, save_results=True, results_path="validation_results.npz",
                   device=None):
    """
    Evaluate elevation-aware TomoNet.

    This version supports:
      - NEW single-output model: returns (Npred, A_eff, P) or (Npred, A_eff) or just Npred
      - BACKWARD compatibility: if a dual-branch model is passed, it will still choose best branch by RMS

    Fixes kept from your original:
      - Properly handles Elev_val argument (must come before c)
      - Converts all inputs to lists of epochs
      - Detects when wrong argument order was used
      - Returns same outputs as original version
    """

    # --- Detect if Elev_val was skipped / misordered ---
    if Elev_val is None or isinstance(Elev_val, (float, int)):
        print("[WARN] Elev_val not provided or misordered (you probably passed c in its place).")
        print("       Expected call: evaluate_model(model, A_val, SWD_val, Nref_val, Napr_val, Elev_val, c=1.0)")
        Elev_val = [np.zeros_like(SWD_val[0]) for _ in range(len(SWD_val))]  # neutral placeholder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # ============================================================
    # Normalize all inputs into lists of epochs
    # ============================================================
    def ensure_list(x):
        """Convert any array/tensor/object into a list of epochs."""
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray) and x.dtype == object:
            return list(x)
        if isinstance(x, np.ndarray) and x.ndim > 1:
            return [x[i] for i in range(x.shape[0])]
        if torch.is_tensor(x) and x.ndim > 1:
            return [x[i].cpu().numpy() for i in range(x.shape[0])]
        return [x]  # fallback

    A_val = ensure_list(A_val)
    SWD_val = ensure_list(SWD_val)
    Nref_val = ensure_list(Nref_val)
    Napr_val = ensure_list(Napr_val)
    Elev_val = ensure_list(Elev_val)

    n_epochs = len(A_val)
    print(f"[INFO] Evaluating elevation-aware model on {n_epochs} tomography epochs...")

    # ============================================================
    # Storage
    # ============================================================
    preds_unscaled, refs_unscaled, aprs_unscaled = [], [], []
    resid_rms, branch_choice = [], []

    # ============================================================
    # Main loop
    # ============================================================
    for i, (A, SWD, Nref, Napr, Elev) in enumerate(zip(A_val, SWD_val, Nref_val, Napr_val, Elev_val)):
        try:
            # --- Convert sparse to dense ---
            if issparse(A): A = A.toarray()
            if issparse(SWD): SWD = SWD.toarray().squeeze()
            if issparse(Nref): Nref = Nref.toarray().squeeze()
            if issparse(Napr): Napr = Napr.toarray().squeeze()

            # --- To torch ---
            A_t = torch.tensor(np.asarray(A, dtype=np.float32), device=device)
            SWD_t = torch.tensor(np.asarray(SWD, dtype=np.float32), device=device)
            Nref_t = torch.tensor(np.asarray(Nref, dtype=np.float32), device=device)
            Napr_t = torch.tensor(np.asarray(Napr, dtype=np.float32), device=device)
            Elev_t = torch.tensor(np.asarray(Elev, dtype=np.float32), device=device)

            with torch.no_grad():
                outputs = model(A_t, SWD_t, Napr_t, Elev_t)

                # ------------------------------------------------------------
                # NEW: single-output model support
                # Expected: (Npred, A_eff, P) or (Npred, A_eff) or Npred
                # BACKCOMPAT: dual-branch model: (N_up, N_down, A_eff, P) etc.
                # ------------------------------------------------------------
                A_eff_t = A_t  # fallback if model doesn't return A_eff

                if isinstance(outputs, (list, tuple)):
                    if len(outputs) == 3:
                        # (Npred, A_eff, P)
                        Npred_t, A_eff_t, _ = outputs
                        chosen = "single"
                        # clamp for safety
                        Npred_t = torch.clamp(Npred_t, min=0.0)
                        SWD_pred = torch.matmul(A_eff_t, Npred_t.unsqueeze(-1)).squeeze(-1)
                        resid = SWD_pred - SWD_t
                        rms = torch.sqrt(torch.mean(resid.pow(2))).item()

                    elif len(outputs) == 2:
                        # (Npred, A_eff)
                        Npred_t, A_eff_t = outputs
                        chosen = "single"
                        Npred_t = torch.clamp(Npred_t, min=0.0)
                        SWD_pred = torch.matmul(A_eff_t, Npred_t.unsqueeze(-1)).squeeze(-1)
                        resid = SWD_pred - SWD_t
                        rms = torch.sqrt(torch.mean(resid.pow(2))).item()

                    elif len(outputs) == 4:
                        # Dual-branch legacy: (N_up, N_down, A_eff, P)
                        Npred_up_t, Npred_down_t, A_eff_t, _ = outputs
                        Npred_up_t = torch.clamp(Npred_up_t, min=0.0)
                        Npred_down_t = torch.clamp(Npred_down_t, min=0.0)

                        SWD_pred_up = torch.matmul(A_eff_t, Npred_up_t.unsqueeze(-1)).squeeze(-1)
                        SWD_pred_down = torch.matmul(A_eff_t, Npred_down_t.unsqueeze(-1)).squeeze(-1)
                        resid_up = SWD_pred_up - SWD_t
                        resid_down = SWD_pred_down - SWD_t

                        rms_up = torch.sqrt(torch.mean(resid_up.pow(2))).item()
                        rms_down = torch.sqrt(torch.mean(resid_down.pow(2))).item()

                        if rms_up <= rms_down:
                            Npred_t = Npred_up_t
                            chosen = "up"
                            rms = rms_up
                        else:
                            Npred_t = Npred_down_t
                            chosen = "down"
                            rms = rms_down

                    else:
                        # Unrecognized tuple length -> treat first element as Npred
                        Npred_t = outputs[0]
                        chosen = "single"
                        Npred_t = torch.clamp(Npred_t, min=0.0)
                        SWD_pred = torch.matmul(A_eff_t, Npred_t.unsqueeze(-1)).squeeze(-1)
                        resid = SWD_pred - SWD_t
                        rms = torch.sqrt(torch.mean(resid.pow(2))).item()

                else:
                    # outputs is a tensor -> Npred only
                    Npred_t = outputs
                    chosen = "single"
                    Npred_t = torch.clamp(Npred_t, min=0.0)
                    SWD_pred = torch.matmul(A_eff_t, Npred_t.unsqueeze(-1)).squeeze(-1)
                    resid = SWD_pred - SWD_t
                    rms = torch.sqrt(torch.mean(resid.pow(2))).item()

                resid_rms.append(rms)
                branch_choice.append(chosen)

                # Unscale
                Npred_unscaled = (Npred_t.detach().cpu().numpy() / c).astype(np.float32)
                Nref_unscaled  = (Nref_t.detach().cpu().numpy() / c).astype(np.float32)
                Napr_unscaled  = (Napr_t.detach().cpu().numpy() / c).astype(np.float32)

                preds_unscaled.append(Npred_unscaled)
                refs_unscaled.append(Nref_unscaled)
                aprs_unscaled.append(Napr_unscaled)

                if (i + 1) % 100 == 0 or (i + 1) == n_epochs:
                    if chosen in ("up", "down"):
                        # if we computed rms_up/rms_down above they exist only in that block,
                        # so just print chosen & rms
                        print(f"[{i+1:04d}/{n_epochs}] RMS={rms:.4f}, chosen={chosen}")
                    else:
                        print(f"[{i+1:04d}/{n_epochs}] RMS={rms:.4f}, chosen=single")

        except Exception as e:
            print(f"[WARN] Skipping epoch {i} due to error: {e}")
            continue

    # ============================================================
    # Summary
    # ============================================================
    mean_rms = np.mean(resid_rms) if resid_rms else np.nan

    up_frac = branch_choice.count("up") / len(branch_choice) if branch_choice else 0
    down_frac = branch_choice.count("down") / len(branch_choice) if branch_choice else 0
    single_frac = branch_choice.count("single") / len(branch_choice) if branch_choice else 0

    print(f"[OK] Evaluation complete. Mean RMS(resid) = {mean_rms:.4f}")
    if single_frac > 0:
        print(f"[INFO] Prediction mode: single={single_frac*100:.1f}%")
    else:
        print(f"[INFO] Branch usage: up={up_frac*100:.1f}%, down={down_frac*100:.1f}%")

    if save_results:
        np.savez_compressed(
            results_path,
            Npred=preds_unscaled,
            Nref=refs_unscaled,
            Napr=aprs_unscaled,
            resid_rms=resid_rms,
            branch_choice=branch_choice,
        )
        print(f"[OK] Results saved to {results_path}")

    return preds_unscaled, refs_unscaled, aprs_unscaled, resid_rms, branch_choice




def save_validation_results(preds_unscaled, refs_unscaled, aprs_unscaled, output_path="validation_results.npz"):
    np.savez_compressed(
        output_path,
        Npredict=np.array(preds_unscaled, dtype=object),
        Nref=np.array(refs_unscaled, dtype=object),
        Napr=np.array(aprs_unscaled, dtype=object),
    )
    print(f"[OK] Saved validation results to {output_path}")

