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
import time
import kalman_func as kf


def _as_switch(value):
    """Handle switches stored either as ['REAL'] or 'REAL'."""
    if isinstance(value, list):
        return value[0]
    return value


def _extract_stdstd_from_obs(obs_epoch):
    """
    Extract M_STD from obs[epoch]["h"][station]["satellite"][sat].
    Returns standard deviations in mm.
    """
    vals = []

    for st in range(len(obs_epoch["h"])):
        sats = obs_epoch["h"][st].get("satellite", [])
        for sat in sats:
            if "M_STD" in sat:
                vals.append(sat["M_STD"])

    return np.asarray(vals, dtype=float).ravel() * 1000.0


def tomography(raytracing, apriori, obs, model, paths, switches):
    """
    Tomography processing for LSQ / REAL or SYNTHETIC.

    Main change compared with the previous Python version:
    - does NOT build dense P matrix
    - does NOT build dense covariance matrix
    - does NOT append prior identity rows to A
    - uses prior as diagonal regularization in the normal equations
    """

    solution = _as_switch(switches["solution"])
    method = _as_switch(switches["method"])
    set_weights = _as_switch(switches.get("setwieghts", "YES"))

    n = (
        int(model["num_lat_TOMO"])
        * int(model["num_lon_TOMO"])
        * int(model["num_levels_TOMO"])
    )

    print(f"[INFO] tomography.py started. Expected voxels: {n}")

    output = {"Nw": [], "Nwerr": []}
    Nw = np.zeros((len(obs), n), dtype=float)
    Nwerr = np.zeros((len(obs), n), dtype=float)

    Pplus = None
    N_vec = None

    for epoch in range(len(obs)):
        start = time.time()

        A = np.asarray(raytracing[epoch]["A"], dtype=float)
        SD = np.asarray(raytracing[epoch]["SD"], dtype=float).ravel() * 1000.0
        elev = np.asarray(raytracing[epoch]["elev"], dtype=float).ravel()

        apr_model = _as_switch(switches.get("aprModel", "ERA5"))
        if apr_model == "ERA5":
            Napr = np.asarray(apriori["Nw_ERA5"][epoch], dtype=float).ravel()
            Nall = apriori["Nw_ERA5"]

        elif apr_model == "DETER":
            Napr = np.asarray(apriori["Nw_DETER"][epoch], dtype=float).ravel()
            Nall = apriori["Nw_DETER"]

        if A.size == 0 or SD.size == 0:
            print(f"[WARN] Empty A or SD at epoch {epoch}. Using apriori.")
            Nw[epoch, :] = Napr
            Nwerr[epoch, :] = Napr
            continue

        if A.shape[1] != n:
            raise ValueError(
                f"[ERROR] A has wrong number of columns at epoch {epoch}. "
                f"Expected {n}, got {A.shape[1]}."
            )

        if A.shape[0] != len(SD):
            raise ValueError(
                f"[ERROR] Observation mismatch at epoch {epoch}. "
                f"A rows={A.shape[0]}, SD length={len(SD)}."
            )

        if len(Napr) != n:
            raise ValueError(
                f"[ERROR] Apriori vector mismatch at epoch {epoch}. "
                f"Expected {n}, got {len(Napr)}."
            )

        if len(elev) != A.shape[0]:
            print(
                f"[WARN] Elevation count mismatch at epoch {epoch}: "
                f"elev={len(elev)}, observations={A.shape[0]}"
            )

        if A.shape[0] < 2:
            print(f"[WARN] Too few observations at epoch {epoch}. Using apriori.")
            Nw[epoch, :] = Napr
            Nwerr[epoch, :] = Napr
            continue

        var0 = 1.0

        if set_weights == "YES":
            if solution == "REAL":
                stdSTD = _extract_stdstd_from_obs(obs[epoch])

                if len(stdSTD) != A.shape[0]:
                    raise ValueError(
                        f"[ERROR] M_STD count mismatch at epoch {epoch}. "
                        f"Found {len(stdSTD)}, expected {A.shape[0]}."
                    )
            else:
                stdSTD = np.ones(A.shape[0], dtype=float)

            elev_safe = np.asarray(elev, dtype=float).ravel()
            elev_safe = elev_safe[:A.shape[0]]

            sin_el = np.sin(np.radians(elev_safe))
            sin_el[np.abs(sin_el) < 1e-6] = 1e-6

            # Observation variances, vector only
            obs_var = (stdSTD ** 2) / (sin_el ** 2)

            # Prior standard deviation
            stdNapr = 0.1 * np.abs(Napr)
            stdNapr[stdNapr < 1.0] = 1.0

            prior_var = stdNapr ** 2

            if method == "LSQ":
                obs_weight = var0 / obs_var
                prior_weight = var0 / prior_var
            else:
                # Kalman code may expect covariance-like values
                obs_weight = obs_var / var0
                prior_weight = prior_var / var0

        else:
            obs_weight = np.ones(A.shape[0], dtype=float)
            prior_weight = np.ones(n, dtype=float)

        if method == "LSQ":
            N_vec, Nwerr_epoch = LSQ(
                A=A,
                SD=SD,
                obs_weight=obs_weight,
                prior_weight=prior_weight,
                Napr=Napr,
                epoch=epoch,
            )

        elif method == "KALMAN":
            # Compatibility path.
            # This still builds a full covariance/weight vector only here.
            # If Kalman also becomes memory-heavy, it needs the same sparse/vector treatment.
            P_diag = np.concatenate((obs_weight, prior_weight))

            A_aug = np.vstack((A, np.eye(n)))
            SD_aug = np.concatenate((SD, Napr))

            P = np.diag(P_diag)

            N_vec, Nwerr_epoch, Pplus = Kalman(
                model, Nall, A_aug, P, SD_aug, epoch, N_vec, Pplus, switches
            )

        else:
            raise ValueError(f"[ERROR] Unknown method: {method}")

        if len(N_vec) != n:
            raise ValueError(
                f"[ERROR] Inversion output size mismatch at epoch {epoch}. "
                f"Expected {n}, got {len(N_vec)}."
            )

        Nw[epoch, :] = N_vec
        Nwerr[epoch, :] = Nwerr_epoch

        print(
            f"[INFO] Epoch {epoch + 1}/{len(obs)} processed "
            f"in {time.time() - start:.2f} sec"
        )

    output["Nw"] = Nw
    output["Nwerr"] = Nwerr
    return output


def LSQ(A, SD, obs_weight, prior_weight, Napr, epoch):
    """
    Weighted LSQ with apriori regularization.

    Solves:

        (A.T W A + W0) x = A.T W SD + W0 Napr

    without constructing full W, P, cov, or apriori identity rows.
    """

    A = np.asarray(A, dtype=float)
    SD = np.asarray(SD, dtype=float).ravel()
    obs_weight = np.asarray(obs_weight, dtype=float).ravel()
    prior_weight = np.asarray(prior_weight, dtype=float).ravel()
    Napr = np.asarray(Napr, dtype=float).ravel()

    n_obs, n = A.shape

    if len(SD) != n_obs:
        raise ValueError(
            f"[ERROR] LSQ epoch {epoch}: SD length {len(SD)} != A rows {n_obs}"
        )

    if len(obs_weight) != n_obs:
        raise ValueError(
            f"[ERROR] LSQ epoch {epoch}: obs_weight length {len(obs_weight)} "
            f"!= A rows {n_obs}"
        )

    if len(prior_weight) != n:
        raise ValueError(
            f"[ERROR] LSQ epoch {epoch}: prior_weight length {len(prior_weight)} "
            f"!= A columns {n}"
        )

    if len(Napr) != n:
        raise ValueError(
            f"[ERROR] LSQ epoch {epoch}: Napr length {len(Napr)} != A columns {n}"
        )

    obs_weight = np.nan_to_num(obs_weight, nan=0.0, posinf=0.0, neginf=0.0)
    prior_weight = np.nan_to_num(prior_weight, nan=0.0, posinf=0.0, neginf=0.0)

    obs_weight[obs_weight < 0] = 0.0
    prior_weight[prior_weight < 0] = 0.0

    # Normal matrix: A.T @ W @ A, without forming W
    Nmat = A.T @ (A * obs_weight[:, None])

    # Apriori contribution: W0, diagonal only
    diag_idx = np.arange(n)
    Nmat[diag_idx, diag_idx] += prior_weight

    # RHS: A.T @ W @ SD + W0 @ Napr
    rhs = A.T @ (SD * obs_weight) + prior_weight * Napr

    try:
        N_vec = np.linalg.solve(Nmat, rhs)
    except np.linalg.LinAlgError:
        print(f"[WARN] Singular normal matrix at epoch {epoch}; using pinv.")
        N_vec = np.linalg.pinv(Nmat) @ rhs

    res = A @ N_vec - SD
    weighted_rms = np.sqrt(np.sum(obs_weight * res ** 2) / max(len(res), 1))

    try:
        C = np.linalg.inv(Nmat)
    except np.linalg.LinAlgError:
        C = np.linalg.pinv(Nmat)

    Nerr = np.sqrt(np.maximum(np.diag(C), 0.0))

    print(f"[INFO] LSQ epoch {epoch}: weighted RMS = {weighted_rms:.4f}")

    return N_vec, Nerr


def Kalman(model, apriori, A, R_SWD, SWD_t, epoch, xPplus, Pplus, switches):
    """
    Original Kalman logic kept mostly unchanged.
    This may still be memory-heavy for large voxel counts.
    """

    Nall = apriori

    if epoch == 0:
        xPminus = np.asarray(Nall[epoch], dtype=float).ravel()

        Phi = np.eye(A.shape[1])

        Q = kf.QApr(
            Nall,
            A,
            model["num_lat_TOMO"],
            model["num_lon_TOMO"],
            model["num_levels_TOMO"],
            model["BLh_pudel_rad"],
            switches,
        )
        Q = np.diag(np.asarray(Q).ravel())

        Pzero = kf.PzeroApr(
            Nall,
            A,
            model["num_lat_TOMO"],
            model["num_lon_TOMO"],
            model["num_levels_TOMO"],
            model["BLh_pudel_rad"],
        )

        Pminus = Phi @ Pzero @ Phi + Q

        K, _, _, _, _ = kf.getGain(Pminus, A, R_SWD)

        xPplus = xPminus + K @ (SWD_t - A @ xPminus)
        Pplus = Pminus - K @ A @ Pminus

        mxP = np.sqrt(np.diag(Pplus))

    else:
        Phi = np.asarray(Nall[epoch], dtype=float).ravel() / np.asarray(
            Nall[epoch - 1], dtype=float
        ).ravel()

        Phi[np.isinf(Phi)] = 1.0
        Phi[np.isnan(Phi)] = 1.0
        Phi = np.diag(Phi)

        xPminus = Phi @ xPplus

        Q = kf.QApr(
            Nall,
            A,
            model["num_lat_TOMO"],
            model["num_lon_TOMO"],
            model["num_levels_TOMO"],
            model["BLh_pudel_rad"],
            switches,
        )
        Q = np.diag(np.asarray(Q).ravel())

        Pminus = Phi @ Pplus @ Phi.T + Q

        K, _, _ = kf.getGainR(Pminus, A, R_SWD, SWD_t, xPminus)

        xPminusit = xPminus.copy()
        for _ in range(1):
            xPminusit = xPminusit + K @ (SWD_t - A @ xPminusit)
            try:
                K, _ = kf.getGainR(Pminus, A, R_SWD, SWD_t, xPminusit)
            except Exception:
                print("[WARN] Kalman inversion failed in robust filtering")

        xPplus = xPminus + K @ (SWD_t - A @ xPminus)
        Pplus = Pminus - K @ A @ Pminus

        mxP = np.sqrt(np.diag(Pplus))

    print(f"[INFO] Kalman epoch {epoch} done")

    return xPplus, mxP, Pplus