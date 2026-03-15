"""
generate_dataset.py
====================
Generates a synthetic Population-Based SHM dataset simulating 50 structures
(e.g. wind turbine towers) with varying properties.

Outputs (all in ./pbshm_dataset/):
  - node_features.csv        : structural + vibration features, one row per structure
  - labels.csv               : binary damage label per structure
  - edges.csv                : similarity-based population graph edges (k-NN, k=5)
  - edge_weights.csv         : cosine similarity weights for each edge
  - population_metadata.json : column descriptions, graph stats, dataset notes

No external SHM library required. Dependencies: numpy, scipy, pandas, scikit-learn.
Run:  python generate_dataset.py
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SEED = 42
N = 50          # number of structures
K = 5           # k-NN edges
DAMAGE_FRAC = 0.30
OUT_DIR = "pbshm_dataset"

rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Structural parameters ──────────────────────────────────────────────────
# Each structure: 3-DOF shear frame with variable mass/stiffness/damping
# Base values with population scatter (~±20%)
m_base = np.array([2000., 1800., 1500.])   # kg
k_base = np.array([1.2e6, 1.0e6, 0.8e6])  # N/m
z_base = np.array([0.03, 0.03, 0.03])      # damping ratios

scatter = 0.20
masses    = rng.uniform(1-scatter, 1+scatter, (N, 3)) * m_base
stiffs    = rng.uniform(1-scatter, 1+scatter, (N, 3)) * k_base
dampings  = rng.uniform(0.01, 0.06, (N, 3))

# Additional scalar geometric properties
heights   = rng.uniform(20., 50., N)        # m
aspect    = rng.uniform(10., 25., N)        # H/D
foundation_stiff = rng.uniform(0.8, 1.2, N) * 5e6  # N/m

# ── 2. Damage flags & effect ──────────────────────────────────────────────────
damage_idx = rng.choice(N, size=int(N * DAMAGE_FRAC), replace=False)
damage_labels = np.zeros(N, dtype=int)
damage_labels[damage_idx] = 1

# Damage reduces stiffness in one storey by 10–35%
damage_severity = rng.uniform(0.10, 0.35, N)
damaged_storey  = rng.integers(0, 3, N)

stiffs_damaged = stiffs.copy()
for i in damage_idx:
    stiffs_damaged[i, damaged_storey[i]] *= (1 - damage_severity[i])

# ── 3. Modal analysis (eigenvalue problem) ───────────────────────────────────
def compute_modes(m, k):
    """3-DOF shear frame: stiffness assembly -> eigenvalues -> natural freqs."""
    M = np.diag(m)
    K = np.array([
        [ k[0]+k[1], -k[1],      0     ],
        [-k[1],       k[1]+k[2], -k[2]  ],
        [ 0,         -k[2],       k[2]  ]
    ])
    vals, vecs = eigh(K, M)
    freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)   # Hz
    return freqs, vecs

nat_freqs   = np.zeros((N, 3))
mode_shapes = np.zeros((N, 3, 3))

for i in range(N):
    f, v = compute_modes(masses[i], stiffs_damaged[i])
    nat_freqs[i]   = f
    mode_shapes[i] = v

# ── 4. MAC values (mode shape correlation vs. undamaged reference) ────────────
ref_f, ref_v = compute_modes(m_base, k_base)

def mac(phi1, phi2):
    num  = np.dot(phi1, phi2)**2
    den  = np.dot(phi1, phi1) * np.dot(phi2, phi2)
    return num / (den + 1e-12)

mac_vals = np.zeros((N, 3))
for i in range(N):
    for j in range(3):
        mac_vals[i, j] = mac(mode_shapes[i, :, j], ref_v[:, j])

# ── 5. Add sensor noise ───────────────────────────────────────────────────────
snr_db = 30
noise_scale = 10 ** (-snr_db / 20)
nat_freqs += rng.normal(0, noise_scale * nat_freqs.std(axis=0), nat_freqs.shape)
mac_vals   = np.clip(mac_vals + rng.normal(0, 0.01, mac_vals.shape), 0, 1)

# ── 6. Assemble node feature matrix ──────────────────────────────────────────
# Structural params (mean per structure)
mean_mass    = masses.mean(axis=1)
mean_stiff   = stiffs.mean(axis=1)
mean_damp    = dampings.mean(axis=1)

feature_cols = (
    ["mass_mean_kg", "stiffness_mean_Nm", "damping_ratio_mean",
     "height_m", "aspect_ratio", "foundation_stiffness_Nm"] +
    [f"nat_freq_{i+1}_Hz" for i in range(3)] +
    [f"mac_mode_{i+1}"    for i in range(3)]
)

X = np.column_stack([
    mean_mass, mean_stiff, mean_damp,
    heights, aspect, foundation_stiff,
    nat_freqs,
    mac_vals
])

df_features = pd.DataFrame(X, columns=feature_cols)
df_features.insert(0, "structure_id", np.arange(N))
df_features.to_csv(f"{OUT_DIR}/node_features.csv", index=False)

# ── 7. Labels ─────────────────────────────────────────────────────────────────
df_labels = pd.DataFrame({
    "structure_id": np.arange(N),
    "damaged":      damage_labels
})
df_labels.to_csv(f"{OUT_DIR}/labels.csv", index=False)

# ── 8. Population graph (k-NN on normalised features) ─────────────────────────
X_norm = normalize(X, axis=0)
nbrs = NearestNeighbors(n_neighbors=K+1, metric="cosine").fit(X_norm)
distances, indices = nbrs.kneighbors(X_norm)

src_list, dst_list, w_list = [], [], []
for i in range(N):
    for j, d in zip(indices[i, 1:], distances[i, 1:]):
        src_list.append(i)
        dst_list.append(int(j))
        w_list.append(float(1 - d))   # cosine similarity

df_edges = pd.DataFrame({"source": src_list, "target": dst_list})
df_edges.to_csv(f"{OUT_DIR}/edges.csv", index=False)

df_weights = pd.DataFrame({"source": src_list, "target": dst_list, "cosine_similarity": w_list})
df_weights.to_csv(f"{OUT_DIR}/edge_weights.csv", index=False)

# ── 9. Metadata JSON ──────────────────────────────────────────────────────────
meta = {
    "description": (
        "Synthetic PBSHM dataset: 50 simulated 3-DOF shear-frame structures "
        "with variable mass, stiffness, damping, and geometry. "
        "Modal features extracted via eigenvalue analysis. "
        "Edges connect structurally similar structures (cosine k-NN, k=5)."
    ),
    "n_structures": N,
    "n_damaged":    int(damage_labels.sum()),
    "n_healthy":    int((1 - damage_labels).sum()),
    "n_edges":      len(src_list),
    "knn_k":        K,
    "random_seed":  SEED,
    "files": {
        "node_features.csv":  "Shape (50, 13). First column is structure_id.",
        "labels.csv":         "Shape (50, 2). Columns: structure_id, damaged (0/1).",
        "edges.csv":          "Edge list. Columns: source, target (structure_id indices).",
        "edge_weights.csv":   "Edge list with cosine similarity weight.",
    },
    "feature_descriptions": {
        "mass_mean_kg":              "Mean storey mass across 3 DOF [kg]",
        "stiffness_mean_Nm":         "Mean inter-storey stiffness [N/m]",
        "damping_ratio_mean":        "Mean viscous damping ratio [-]",
        "height_m":                  "Total structure height [m]",
        "aspect_ratio":              "Height-to-diameter ratio [-]",
        "foundation_stiffness_Nm":   "Foundation rotational stiffness [N/m]",
        "nat_freq_1_Hz":             "1st natural frequency [Hz] (noisy measurement)",
        "nat_freq_2_Hz":             "2nd natural frequency [Hz]",
        "nat_freq_3_Hz":             "3rd natural frequency [Hz]",
        "mac_mode_1":                "MAC value, mode 1 vs undamaged reference [-]",
        "mac_mode_2":                "MAC value, mode 2 vs undamaged reference [-]",
        "mac_mode_3":                "MAC value, mode 3 vs undamaged reference [-]",
    },
    "notes": [
        "Damage is simulated as a stiffness reduction (10-35%) in one storey.",
        "The undamaged reference is the population-mean structure.",
        "nat_freq and MAC features carry damage signal; structural params encode population heterogeneity.",
        "Candidates are free to construct their own edges instead of using the provided ones.",
        "Tool-agnostic: load with pandas (Python), readtable (MATLAB), read.csv (R), etc.",
    ]
}

with open(f"{OUT_DIR}/population_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Dataset generated successfully.")
print(f"  Structures : {N}  |  Damaged: {damage_labels.sum()}  |  Healthy: {N - damage_labels.sum()}")
print(f"  Edges      : {len(src_list)}")
print(f"  Files saved to ./{OUT_DIR}/")
