import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh


SEED = 42
N_HEALTHY = 100
N_DAMAGED = 100
N_STRUCTURES = N_HEALTHY + N_DAMAGED
MIN_DOF = 4
MAX_DOF = 8
MAX_SAVED_MODES = 6

HEALTHY_STIFFNESS_MIN = 1.0e6
HEALTHY_STIFFNESS_MAX = 1.4e6
DAMAGED_STIFFNESS_MIN = 0.45e6
DAMAGED_STIFFNESS_MAX = 0.75e6

MASS_MIN = 1500.0
MASS_MAX = 2500.0

HEIGHT_MIN = 3.0
HEIGHT_MAX = 5.0

FREQ_NOISE_DB = -24.0
MODE_SHAPE_NOISE_MAX_FRAC = 0.02  # max 2% additive noise after normalization


def shear_stiffness_matrix(k: np.ndarray) -> np.ndarray:
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i > 0:
            K[i - 1, i - 1] += k[i]
            K[i, i - 1] -= k[i]
            K[i - 1, i] -= k[i]
    return K


def normalize_mode_columns(vecs: np.ndarray) -> np.ndarray:
    out = vecs.copy()
    for m in range(out.shape[1]):
        col = out[:, m]
        # fix sign for consistency: largest-magnitude entry positive
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            col = -col
        max_abs = np.max(np.abs(col))
        if max_abs > 0:
            col = col / max_abs
        out[:, m] = col
    return out


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rng = np.random.default_rng(SEED)

    all_ids = np.arange(N_STRUCTURES)
    damaged_ids = set(rng.choice(all_ids, size=N_DAMAGED, replace=False).tolist())

    damage_severity_trace = rng.uniform(0.15, 0.40, N_STRUCTURES)
    damage_storey = rng.integers(0, MAX_DOF, N_STRUCTURES)

    structures = []
    labels = []
    diagnostics = []

    freq_noise_scale_factor = 10 ** (FREQ_NOISE_DB / 20.0)

    for i in range(N_STRUCTURES):
        ndof = int(rng.integers(MIN_DOF, MAX_DOF + 1))

        masses = rng.uniform(MASS_MIN, MASS_MAX, ndof)
        stiffs_healthy = rng.uniform(HEALTHY_STIFFNESS_MIN, HEALTHY_STIFFNESS_MAX, ndof)
        heights = rng.uniform(HEIGHT_MIN, HEIGHT_MAX, ndof)

        is_damaged = int(i in damaged_ids)
        dmg_loc = int(damage_storey[i]) % ndof

        stiffs_physical = stiffs_healthy.copy()
        actual_damage_severity = 0.0
        if is_damaged:
            damaged_value = float(rng.uniform(DAMAGED_STIFFNESS_MIN, DAMAGED_STIFFNESS_MAX))
            actual_damage_severity = 1.0 - damaged_value / stiffs_healthy[dmg_loc]
            stiffs_physical[dmg_loc] = damaged_value

        M = np.diag(masses)
        K = shear_stiffness_matrix(stiffs_physical)

        vals, vecs = eigh(K, M)
        vals = np.abs(vals)
        nat_freqs_hz = np.sqrt(vals) / (2.0 * np.pi)

        nat_freqs_noisy = nat_freqs_hz + rng.normal(0.0, freq_noise_scale_factor * nat_freqs_hz, ndof)

        # normalize exact modes first to max component 1
        vecs_norm = normalize_mode_columns(vecs)

        # add bounded small noise (max ±2% of normalized scale) entrywise
        noise = rng.uniform(-MODE_SHAPE_NOISE_MAX_FRAC, MODE_SHAPE_NOISE_MAX_FRAC, size=vecs_norm.shape)
        vecs_noisy = vecs_norm + noise

        # renormalize again so each noisy mode shape has max component exactly 1
        vecs_noisy = normalize_mode_columns(vecs_noisy)

        n_saved_modes = min(ndof, MAX_SAVED_MODES)
        saved_mode_indices = list(range(n_saved_modes))

        edges = [[j, j + 1] for j in range(ndof - 1)]

        structures.append({
            "structure_id": int(i),
            "n_storeys": int(ndof),
            "edges": edges,
            "geometry": {
                "storey_heights_m": [float(x) for x in heights.tolist()],
                "cumulative_heights_m": [float(x) for x in np.cumsum(heights).tolist()],
            },
            "modal_data": {
                "n_modes_available": int(ndof),
                "n_modes_saved": int(n_saved_modes),
                "mode_indices": [int(m + 1) for m in saved_mode_indices],
                "frequencies_Hz": [float(x) for x in nat_freqs_noisy[saved_mode_indices].tolist()],
                "mode_shapes_rows_storeys_cols_modes": [
                    [float(v) for v in row] for row in vecs_noisy[:, saved_mode_indices].tolist()
                ],
            },
        })

        labels.append({
            "structure_id": int(i),
            "damaged": int(is_damaged),
            "damage_storey": int(dmg_loc) if is_damaged else -1,
        })

        diagnostics.append({
            "structure_id": int(i),
            "n_storeys": int(ndof),
            "damaged": int(is_damaged),
            "damage_storey": int(dmg_loc) if is_damaged else -1,
            "assigned_damage_severity_from_old_scheme": float(damage_severity_trace[i]) if is_damaged else 0.0,
            "actual_damage_severity_vs_local_healthy": float(actual_damage_severity) if is_damaged else 0.0,
            "min_healthy_stiffness": float(np.min(stiffs_healthy)),
            "max_healthy_stiffness": float(np.max(stiffs_healthy)),
            "damaged_stiffness_value": float(stiffs_physical[dmg_loc]) if is_damaged else np.nan,
            "min_mass": float(np.min(masses)),
            "max_mass": float(np.max(masses)),
            "n_modes_saved": int(n_saved_modes),
            "mode_shape_noise_max_frac": MODE_SHAPE_NOISE_MAX_FRAC,
        })

    with open(out_dir / "structures_measurements.json", "w", encoding="utf-8") as f:
        json.dump(structures, f, indent=2)

    pd.DataFrame(labels).to_csv(out_dir / "structure_labels.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out_dir / "latent_generation_diagnostics.csv", index=False)

    metadata = {
        "seed": SEED,
        "n_structures": N_STRUCTURES,
        "n_healthy": N_HEALTHY,
        "n_damaged": N_DAMAGED,
        "representation": "per_structure_modal_data_limited_to_first_modes",
        "notes": [
            "Modal quantities are stored per structure and per mode.",
            "Only the first up to 6 modes are saved for each structure.",
            "If a structure has fewer than 6 DOFs, all available modes are saved.",
            "frequencies_Hz contains one noisy natural frequency per saved mode.",
            "mode_shapes_rows_storeys_cols_modes contains the noisy mode shape matrix with rows=storeys and columns=saved modes.",
            "Exact mode shapes are first normalized so each mode has max component 1.",
            "Then bounded additive noise with magnitude at most 2% is added entrywise, followed by renormalization.",
            "Storey heights are stored separately under geometry.",
            "Mass varies by floor and by structure in this version.",
            "Healthy and damaged stiffness bands do not overlap in the latent generator."
        ],
        "healthy_stiffness_range_N_per_m": [HEALTHY_STIFFNESS_MIN, HEALTHY_STIFFNESS_MAX],
        "damaged_stiffness_range_N_per_m": [DAMAGED_STIFFNESS_MIN, DAMAGED_STIFFNESS_MAX],
        "mass_range_kg": [MASS_MIN, MASS_MAX],
        "height_range_m": [HEIGHT_MIN, HEIGHT_MAX],
        "max_saved_modes": MAX_SAVED_MODES,
        "mode_shape_noise_max_frac": MODE_SHAPE_NOISE_MAX_FRAC,
        "mode_shape_storage": {"orientation": "rows_storeys_cols_modes"},
    }
    with open(out_dir / "population_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Wrote dataset with normalized mode shapes and max 2% noise.")


if __name__ == "__main__":
    main()
