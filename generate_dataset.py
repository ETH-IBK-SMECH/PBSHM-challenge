"""
generate_dataset_no_overlap.py
==============================

PBSHM toy dataset variant with guaranteed separation between healthy and
post-damage local inter-storey stiffness values.

Key change vs original generator
--------------------------------
The original code samples healthy stiffnesses in [0.8e6, 1.4e6] N/m and then
applies a 15-40% local reduction for damaged structures. That creates overlap:
some damaged storeys can still lie inside the healthy stiffness range.

Here we enforce *no overlap* by:
- sampling healthy inter-storey stiffnesses from [1.0e6, 1.4e6] N/m
- assigning the damaged storey an absolute post-damage stiffness sampled from
  [0.45e6, 0.75e6] N/m

Thus, for every structure:
    max(damaged local stiffness) = 0.75e6 < 1.0e6 = min(healthy stiffness)

The exported candidate-facing files are unchanged in format.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SEED = 42
N = 50
MIN_DOF = 4
MAX_DOF = 8
K_POP = 5
DAMAGE_FRAC = 0.30
OUT_DIR = "pbshm_dataset_no_overlap"

# Healthy and damaged local stiffness bands chosen to guarantee no overlap.
HEALTHY_STIFF_MIN = 1.0e6
HEALTHY_STIFF_MAX = 1.4e6
DAMAGED_STIFF_MIN = 0.45e6
DAMAGED_STIFF_MAX = 0.75e6

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def shear_stiffness_matrix(k):
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i > 0:
            K[i - 1, i - 1] += k[i]
            K[i, i - 1] -= k[i]
            K[i - 1, i] -= k[i]
    return K


def geometry_summary(heights, ndof):
    return np.array([
        ndof,
        np.mean(heights),
        np.std(heights),
        np.sum(heights),
    ])


damage_idx = set(rng.choice(N, size=int(N * DAMAGE_FRAC), replace=False).tolist())
damage_storey = rng.integers(0, MAX_DOF, N)

structures = []
labels = []
pop_summaries = []
latent_rows = []

for i in range(N):
    ndof = int(rng.integers(MIN_DOF, MAX_DOF + 1))

    masses = rng.uniform(1500, 2500, ndof)
    stiffs = rng.uniform(HEALTHY_STIFF_MIN, HEALTHY_STIFF_MAX, ndof)
    heights = rng.uniform(3.0, 5.0, ndof)

    is_damaged = int(i in damage_idx)
    stiffs_phys = stiffs.copy()
    dmg_loc = int(damage_storey[i]) % ndof

    damage_severity = None
    post_damage_local_stiffness = None
    if is_damaged:
        post_damage_local_stiffness = rng.uniform(DAMAGED_STIFF_MIN, DAMAGED_STIFF_MAX)
        undamaged_local_stiffness = stiffs[dmg_loc]
        stiffs_phys[dmg_loc] = post_damage_local_stiffness
        damage_severity = 1.0 - post_damage_local_stiffness / undamaged_local_stiffness
    else:
        undamaged_local_stiffness = stiffs[dmg_loc]

    M = np.diag(masses)
    K = shear_stiffness_matrix(stiffs_phys)
    vals, vecs = eigh(K, M)
    nat_freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)

    freq_noise_scale = 10 ** (-24 / 20)
    nat_freqs_noisy = nat_freqs + rng.normal(0, freq_noise_scale * nat_freqs, ndof)

    shape_noise_scale = 0.03
    vecs_noisy = vecs + rng.normal(0, shape_noise_scale, vecs.shape)
    dominant_mode = np.argmax(np.abs(vecs_noisy), axis=1)

    node_features = []
    for j in range(ndof):
        m_j = int(dominant_mode[j])
        node_features.append({
            "storey": j,
            "height_m": round(float(heights[j]), 3),
            "dominant_modal_frequency_Hz": round(float(nat_freqs_noisy[m_j]), 5),
        })

    edges = [[j, j + 1] for j in range(ndof - 1)]

    structures.append({
        "structure_id": i,
        "n_storeys": ndof,
        "edges": edges,
        "feature_names": [
            "height_m",
            "dominant_modal_frequency_Hz",
        ],
        "node_features": node_features,
    })

    labels.append({
        "structure_id": i,
        "damaged": is_damaged,
        "damage_storey": dmg_loc if is_damaged else None,
    })

    latent_rows.append({
        "structure_id": i,
        "damaged": is_damaged,
        "damage_storey": dmg_loc if is_damaged else None,
        "undamaged_local_stiffness_N_per_m": float(undamaged_local_stiffness),
        "post_damage_local_stiffness_N_per_m": (
            float(post_damage_local_stiffness) if post_damage_local_stiffness is not None else np.nan
        ),
        "damage_severity": float(damage_severity) if damage_severity is not None else np.nan,
        "min_healthy_stiffness_band_N_per_m": HEALTHY_STIFF_MIN,
        "max_damaged_stiffness_band_N_per_m": DAMAGED_STIFF_MAX,
        "min_storey_mass_kg": float(np.min(masses)),
        "max_storey_mass_kg": float(np.max(masses)),
    })

    pop_summaries.append(geometry_summary(heights, ndof))

S = normalize(np.array(pop_summaries), axis=0)
nbrs = NearestNeighbors(n_neighbors=K_POP + 1, metric="cosine").fit(S)
distances, indices = nbrs.kneighbors(S)

src_list, dst_list, w_list = [], [], []
for i in range(N):
    for j, d in zip(indices[i, 1:], distances[i, 1:]):
        src_list.append(i)
        dst_list.append(int(j))
        w_list.append(round(float(1 - d), 6))

with open(f"{OUT_DIR}/structures_measurements.json", "w", encoding="utf-8") as f:
    json.dump(structures, f, indent=2)

labels_df = pd.DataFrame(labels)
labels_df["damage_storey"] = pd.array(labels_df["damage_storey"], dtype="Int64")
labels_df.to_csv(f"{OUT_DIR}/structure_labels.csv", index=False)

pd.DataFrame({
    "source": src_list,
    "target": dst_list,
}).to_csv(f"{OUT_DIR}/population_edges_geometry.csv", index=False)

pd.DataFrame({
    "source": src_list,
    "target": dst_list,
    "cosine_similarity": w_list,
}).to_csv(f"{OUT_DIR}/population_edge_weights_geometry.csv", index=False)

# Optional latent diagnostic file for your own verification; not intended for candidates.
pd.DataFrame(latent_rows).to_csv(f"{OUT_DIR}/latent_generation_diagnostics.csv", index=False)

meta = {
    "description": (
        "Population of 50 shear-frame structures with 4-8 storeys. "
        "This variant enforces non-overlapping local stiffness bands between "
        "healthy and post-damage storeys. Candidate-facing exports remain the same."
    ),
    "n_structures": N,
    "n_damaged": int(sum(row["damaged"] for row in labels)),
    "n_healthy": int(N - sum(row["damaged"] for row in labels)),
    "storey_range": [MIN_DOF, MAX_DOF],
    "population_graph_k": K_POP,
    "random_seed": SEED,
    "healthy_stiffness_band_N_per_m": [HEALTHY_STIFF_MIN, HEALTHY_STIFF_MAX],
    "damaged_local_stiffness_band_N_per_m": [DAMAGED_STIFF_MIN, DAMAGED_STIFF_MAX],
    "guaranteed_separation_condition": DAMAGED_STIFF_MAX < HEALTHY_STIFF_MIN,
    "candidate_input_files": {
        "structures_measurements.json": "Topology and node-level measurement-like features.",
        "structure_labels.csv": "Structure-level label and true damage location for evaluation.",
        "population_edges_geometry.csv": "Starter population graph based on geometry-only summaries.",
        "population_edge_weights_geometry.csv": "Weighted version of the starter graph.",
    },
    "node_feature_descriptions": {
        "height_m": "Storey height [m].",
        "dominant_modal_frequency_Hz": (
            "Noisy dominant modal frequency assigned to the storey via the "
            "largest noisy mode-shape amplitude."
        ),
    },
    "hidden_generation_notes": [
        "Masses still vary independently by storey and by structure, sampled from U(1500, 2500) kg.",
        "Healthy inter-storey stiffnesses are sampled from U(1.0e6, 1.4e6) N/m.",
        "For damaged structures, one storey is reassigned an absolute post-damage stiffness in U(0.45e6, 0.75e6) N/m.",
        "Therefore the damaged local stiffness never overlaps with the healthy stiffness band.",
    ],
}

with open(f"{OUT_DIR}/population_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Dataset with non-overlapping damaged/healthy stiffness bands generated.")
