"""
generate_dataset_v2.py
======================
Population-Based SHM dataset — variable-geometry structures.

Each of the 50 structures is a shear-frame with a RANDOM number of storeys (4–8).
Structures are represented as small graphs: nodes = storeys, edges = inter-storey connections.
This means feature matrices vary in size across structures — you cannot simply stack them.

Physical model: N-DOF lumped-mass shear frame.
  - Each storey has mass m_i and inter-storey stiffness k_i
  - Damage = localised stiffness reduction (15–40%) in one storey
  - Modal features extracted via eigenvalue analysis

Output files (./pbshm_dataset_v2/):
  structures.json          — one entry per structure: geometry, node features, edge list, label
  population_edges.csv     — population-level similarity graph (which structures are similar)
  population_edge_weights.csv
  population_metadata.json — full description of every field

Dependencies: numpy, scipy, scikit-learn, pandas
"""

import numpy as np
import json
import os
import pandas as pd
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SEED = 42
N = 50
MIN_DOF = 4
MAX_DOF = 8
K_POP = 5          # population graph k-NN
DAMAGE_FRAC = 0.30
OUT_DIR = "pbshm_dataset_v2"

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── damage assignments ────────────────────────────────────────────────────────
damage_idx      = set(rng.choice(N, size=int(N * DAMAGE_FRAC), replace=False).tolist())
damage_severity = rng.uniform(0.15, 0.40, N)   # stiffness reduction fraction
damage_storey   = rng.integers(0, MAX_DOF, N)  # which storey (clamped to ndof later)

# ── helper: build stiffness matrix for N-DOF shear frame ─────────────────────
def shear_stiffness_matrix(k):
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i > 0:
            K[i-1, i-1] += k[i]
            K[i,   i-1] -= k[i]
            K[i-1, i  ] -= k[i]
    return K

# ── helper: fixed-length population-level summary (for k-NN graph) ───────────
# We use statistics over the node features so structures of different sizes
# can still be compared at population level.
def pop_summary(masses, stiffs, nat_freqs, ndof):
    return np.array([
        ndof,
        np.mean(masses), np.std(masses),
        np.mean(stiffs), np.std(stiffs),
        nat_freqs[0],                         # fundamental frequency
        np.mean(np.diff(nat_freqs)),          # mean freq spacing
        nat_freqs[-1] - nat_freqs[0],         # freq range
    ])

# ── generate structures ───────────────────────────────────────────────────────
structures   = []
pop_summaries = []

for i in range(N):
    ndof = int(rng.integers(MIN_DOF, MAX_DOF + 1))

    # Physical parameters — scatter around base values
    masses = rng.uniform(1500, 2500, ndof)          # kg per storey
    stiffs = rng.uniform(0.8e6, 1.4e6, ndof)        # N/m inter-storey
    height = rng.uniform(3.0, 5.0, ndof)            # storey height [m]

    # Damage
    is_damaged = int(i in damage_idx)
    stiffs_phys = stiffs.copy()
    dmg_loc = int(damage_storey[i]) % ndof          # clamp to actual ndof
    if is_damaged:
        stiffs_phys[dmg_loc] *= (1.0 - damage_severity[i])

    # Modal analysis
    M = np.diag(masses)
    K = shear_stiffness_matrix(stiffs_phys)
    vals, vecs = eigh(K, M)
    nat_freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)   # Hz, ascending

    # Add sensor noise (SNR ~ 30 dB)
    noise = 10 ** (-30 / 20)
    nat_freqs_noisy = nat_freqs + rng.normal(0, noise * nat_freqs, ndof)

    # MAC values vs. undamaged reference
    K_ref = shear_stiffness_matrix(stiffs)          # undamaged
    _, vecs_ref = eigh(K_ref, M)
    mac_vals = np.array([
        (np.dot(vecs[:, j], vecs_ref[:, j])**2 /
         (np.dot(vecs[:, j], vecs[:, j]) * np.dot(vecs_ref[:, j], vecs_ref[:, j]) + 1e-12))
        for j in range(ndof)
    ])

    # ── Node features (one row per storey) ───────────────────────────────────
    # Each node: [mass, stiffness_above, storey_height, nat_freq_j, mac_j]
    # nat_freq_j and mac_j are the mode most associated with storey j
    # (using mode shape amplitude as proxy for participation)
    dominant_mode = np.argmax(np.abs(vecs), axis=1)   # shape (ndof,)
    node_features = []
    for j in range(ndof):
        m_j = int(dominant_mode[j])
        node_features.append({
            "storey":           j,
            "mass_kg":          round(float(masses[j]), 2),
            "stiffness_Nm":     round(float(stiffs_phys[j]), 2),
            "height_m":         round(float(height[j]), 3),
            "nat_freq_Hz":      round(float(nat_freqs_noisy[m_j]), 5),
            "mac":              round(float(np.clip(mac_vals[m_j], 0, 1)), 5),
        })

    # ── Edges (chain: storey j — storey j+1) ─────────────────────────────────
    edges = [[j, j+1] for j in range(ndof - 1)]

    # ── Structure record ──────────────────────────────────────────────────────
    structures.append({
        "structure_id":   i,
        "n_storeys":      ndof,
        "damaged":        is_damaged,
        "damage_storey":  dmg_loc if is_damaged else None,
        "node_features":  node_features,
        "edges":          edges,
        "feature_names":  ["mass_kg", "stiffness_Nm", "height_m", "nat_freq_Hz", "mac"],
    })

    pop_summaries.append(pop_summary(masses, stiffs_phys, nat_freqs_noisy, ndof))

# ── Population graph (k-NN on summary vectors) ───────────────────────────────
S = normalize(np.array(pop_summaries), axis=0)
nbrs = NearestNeighbors(n_neighbors=K_POP + 1, metric="cosine").fit(S)
distances, indices = nbrs.kneighbors(S)

src_list, dst_list, w_list = [], [], []
for i in range(N):
    for j, d in zip(indices[i, 1:], distances[i, 1:]):
        src_list.append(i)
        dst_list.append(int(j))
        w_list.append(round(float(1 - d), 6))

# ── Save ──────────────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/structures.json", "w") as f:
    json.dump(structures, f, indent=2)

pd.DataFrame({"source": src_list, "target": dst_list}).to_csv(
    f"{OUT_DIR}/population_edges.csv", index=False)
pd.DataFrame({"source": src_list, "target": dst_list, "cosine_similarity": w_list}).to_csv(
    f"{OUT_DIR}/population_edge_weights.csv", index=False)

# ── Metadata ──────────────────────────────────────────────────────────────────
damaged_count = sum(1 for s in structures if s["damaged"])
storey_counts = [s["n_storeys"] for s in structures]

meta = {
    "description": (
        "Population of 50 shear-frame structures with VARIABLE number of storeys (4–8). "
        "Each structure is a small graph: nodes = storeys, edges = inter-storey connections. "
        "Structures vary in geometry — feature matrices differ in size across structures. "
        "The population graph connects structurally similar structures."
    ),
    "n_structures":       N,
    "n_damaged":          damaged_count,
    "n_healthy":          N - damaged_count,
    "storey_range":       [MIN_DOF, MAX_DOF],
    "storey_distribution": {str(k): storey_counts.count(k) for k in range(MIN_DOF, MAX_DOF+1)},
    "population_graph_k": K_POP,
    "random_seed":        SEED,
    "files": {
        "structures.json": (
            "List of 50 structure objects. Each has: structure_id, n_storeys, damaged (0/1), "
            "damage_storey (null if healthy), node_features (list of dicts, one per storey), "
            "edges (list of [src, dst] pairs), feature_names."
        ),
        "population_edges.csv":        "Population-level similarity graph. Columns: source, target.",
        "population_edge_weights.csv": "Same with cosine_similarity weight column.",
    },
    "node_feature_descriptions": {
        "mass_kg":       "Storey mass [kg]",
        "stiffness_Nm":  "Inter-storey stiffness [N/m] — reduced if this storey is damaged",
        "height_m":      "Storey height [m]",
        "nat_freq_Hz":   "Natural frequency of the dominant mode for this storey [Hz] (noisy)",
        "mac":           "MAC value of that mode vs. undamaged reference [-]",
    },
    "population_summary_features_used_for_edges": [
        "n_storeys",
        "mean(mass)", "std(mass)",
        "mean(stiffness)", "std(stiffness)",
        "fundamental_frequency",
        "mean_frequency_spacing",
        "frequency_range",
    ],
    "loading_examples": {
        "python": "import json; data = json.load(open('structures.json'))",
        "matlab": "data = jsondecode(fileread('structures.json'));",
        "r":      "library(jsonlite); data <- fromJSON('structures.json')",
    },
    "design_notes": [
        "Structures have 4–8 storeys — feature matrices are NOT the same size across structures.",
        "This is intentional: candidates must handle variable-size graphs.",
        "The population graph edges are based on summary statistics, not raw features.",
        "Damage signal is in stiffness_Nm and mac at the damaged storey.",
        "Candidates may construct their own population edges instead of using the provided ones.",
        "Task is graph-level binary classification: is this structure damaged?",
    ]
}

with open(f"{OUT_DIR}/population_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Dataset generated.")
print(f"  Structures : {N}  |  Damaged: {damaged_count}  |  Healthy: {N - damaged_count}")
print(f"  Storey counts: { {k: storey_counts.count(k) for k in range(MIN_DOF, MAX_DOF+1)} }")
print(f"  Population edges: {len(src_list)}")
print(f"  Saved to ./{OUT_DIR}/")
