"""
generate_dataset_revised.py
===========================

Revised PBSHM toy dataset with variable-size structures and measurement-like exports.

Design goals:
- keep the simulation model transparent
- avoid exporting direct shortcut features such as post-damage stiffness or MAC
- support simple baselines first, with graph methods as a natural extension

Simulation model
----------------
Each structure is an N-DOF lumped-mass shear frame with 4-8 storeys.
Damage is simulated as a localized reduction in one inter-storey stiffness.
Modal quantities are computed from the generalized eigenproblem and then
perturbed with light noise to emulate imperfect measurements.

Candidate-facing exports
------------------------
The exported measurement file contains:
- graph topology
- storey height
- a noisy dominant modal frequency assigned per storey

Latent physical parameters such as the true damaged stiffness values are used
internally to generate the synthetic data but are not exported as candidate
inputs for inference.
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
OUT_DIR = "pbshm_dataset_revised"

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
damage_severity = rng.uniform(0.15, 0.40, N)
damage_storey = rng.integers(0, MAX_DOF, N)

structures = []
labels = []
pop_summaries = []

for i in range(N):
    ndof = int(rng.integers(MIN_DOF, MAX_DOF + 1))

    masses = rng.uniform(1500, 2500, ndof)
    stiffs = rng.uniform(0.8e6, 1.4e6, ndof)
    heights = rng.uniform(3.0, 5.0, ndof)

    is_damaged = int(i in damage_idx)
    stiffs_phys = stiffs.copy()
    dmg_loc = int(damage_storey[i]) % ndof
    if is_damaged:
        stiffs_phys[dmg_loc] *= (1.0 - damage_severity[i])

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

meta = {
    "description": (
        "Population of 50 shear-frame structures with 4-8 storeys. "
        "The candidate-facing data exposes graph topology and lightweight "
        "measurement-like modal features, while latent simulator parameters "
        "remain hidden."
    ),
    "n_structures": N,
    "n_damaged": int(sum(row["damaged"] for row in labels)),
    "n_healthy": int(N - sum(row["damaged"] for row in labels)),
    "storey_range": [MIN_DOF, MAX_DOF],
    "population_graph_k": K_POP,
    "random_seed": SEED,
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
        "Each structure is generated from random masses and inter-storey stiffnesses.",
        "Damage is simulated as a local stiffness reduction in one storey.",
        "True damaged stiffness values are not exported as inference inputs.",
        "Mode shapes are perturbed before assigning a dominant modal frequency to each storey.",
    ],
}

with open(f"{OUT_DIR}/population_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("Revised dataset generated.")
