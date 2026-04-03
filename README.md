# PBSHM Coding Challenge

**Chair of Structural Mechanics and Monitoring | ETH Zurich**

## Overview

This repository contains a small take-home coding exercise on **Population-Based Structural Health Monitoring (PBSHM)**.

You will work with a simulated population of 50 shear-frame structures with a **variable number of storeys (4-8)**. The goal is to detect whether a structure is damaged and, if time permits, explore whether graph-based learning can support **damage localization** through node- or edge-level damage indicators.

## What Is Simulated vs. What You Should Use

The dataset was generated from an `N`-DOF lumped-mass shear-frame model with localized stiffness reduction used to simulate damage. We provide the generation script so the physical assumptions are transparent.

However, for the purposes of this exercise, assume that in deployment you **do not directly know the true damaged stiffness values or other hidden simulator state**. Your method should be built from the provided measurement-like quantities and labels. You should construct the within-structure graph representation yourself from the storey-level information.

In other words:

- the simulation model is disclosed for physical clarity
- the inference task should use the provided candidate-facing files
- do not treat hidden physical parameters as measured inputs at rollout time

## Dataset

The dataset files are located in this repository.

### Files

| File | Description |
|---|---|
| `structures_measurements.json` | Candidate-facing inputs: one entry per structure with storey-level measurement-like node features |
| `structure_labels.csv` | Structure-level detection label and true damaged storey for evaluation / optional localization analysis |
| `population_metadata.json` | Dataset summary and field descriptions |
| `generate_dataset_revised.py` | Reference generator showing how the synthetic data was created |

### Structure of `structures_measurements.json`

Each structure contains:

- `structure_id`
- `n_storeys`
- `node_features`
- `feature_names`

Each node currently includes:

- `storey`
- `height_m`
- `modal_data (frequencies_Hz + mode_shapes_rows_storeys_cols_modes)`

These are intended as lightweight, measurement-like features for a toy exercise. You may derive additional node, edge, or graph features from them if helpful.

### Labels

`structure_labels.csv` contains:

- `structure_id`
- `damaged` as a binary structure-level label
- `damage_storey` as the true damaged location for optional localization analysis

The structure-level label is the main supervision target. The storey-level target is included so that stronger candidates can explore approximate localization or node/edge damage indicators.

## Tasks

### Task 1 - Explore the population

Characterize the dataset and explain the variation across the population.

- Visualize the distribution of structure sizes and geometry
- Inspect the provided measurement-like node features
- Propose which raw or derived features might be damage-sensitive

We are looking for physical intuition and clear coding, not just plots.

### Task 2 - Simple structure-level baseline

Build a simple baseline for **damage detection** using fixed-length summaries of each structure.

Examples include:

- logistic regression
- random forest
- support vector machine
- a small MLP

Because structures have different numbers of nodes, you will need to design a sensible summary representation.

Report appropriate metrics such as accuracy, F1, and ROC-AUC using cross-validation.

### Task 3 - Unsupervised or anomaly-based baseline

Implement at least one simpler exploratory method that does not rely on a graph neural network.

Examples include:

- clustering on structure-level summaries
- PCA or other embedding plus visual separation
- nearest-neighbor anomaly scoring
- isolation forest or another anomaly detector

Discuss whether damaged structures appear separable and what the limitations of these simpler methods are.

### Task 4 - Graph-based extension

Construct a graph representation for each structure using the provided storey-level information, then implement a graph-based model and compare it to your simpler baselines.

You may:

- perform graph-level damage detection
- estimate node- or edge-level damage indicators
- or do both

If you choose a GNN, a sensible pattern is:

1. encode each structure graph
2. pool node information into a structure representation for detection
3. inspect node embeddings or scores for approximate localization

The emphasis is on whether the graph formulation is well-motivated and interpretable.

## Deliverables

1. Runnable code in your preferred language
2. A short `README` explaining how to run the work and key design decisions
3. A short slide deck for a 10-minute presentation

## What We Are Assessing

We care about:

- clear and reproducible code
- sensible handling of variable-size structures and their graph representations
- ability to build and justify simple baselines
- whether graph methods are used thoughtfully rather than by default
- physical interpretation of the results

Clarity of reasoning matters more than headline accuracy.

## Suggested Marking Rubric

| Criterion | Points |
|---|---|
| Code correctness and reproducibility | 20 |
| Exploratory analysis and feature reasoning | 20 |
| Quality of simple baselines | 20 |
| Graph-based modeling and interpretation | 25 |
| Communication and presentation clarity | 15 |
| **Total** | **100** |

## Rules

- Any open-source library or toolbox is permitted
- Internet access for documentation is permitted
- You may use AI coding assistants, but be prepared to explain every line you submit
- Collaboration with other candidates is not permitted

## Notes

- Please construct the graph representation for each individual structure yourself from the provided storey-level information.
- The generation script is included for transparency, but your inference pipeline should rely on the provided candidate-facing inputs.
- This is a toy exercise. A clean and well-argued small solution is better than an overly ambitious one.



## Updated dataset specification

This version of the dataset contains **200 structures total**:
- **100 healthy**
- **100 damaged**

For each structure, the dataset stores modal quantities **per mode** rather than node-level dominant frequencies.

Each entry in `structures_measurements.json` contains:
- `geometry.storey_heights_m`
- `geometry.cumulative_heights_m`
- `modal_data.n_modes_available`
- `modal_data.n_modes_saved`
- `modal_data.mode_indices`
- `modal_data.frequencies_Hz`
- `modal_data.mode_shapes_rows_storeys_cols_modes`

### Modal storage rule
For each structure, the dataset saves **the first up to 6 modes**:
- if the structure has **6 or more DOFs**, only the **first 6 frequencies and corresponding mode shapes** are stored
- if the structure has **fewer than 6 DOFs**, all available modes are stored

### Orientation
`mode_shapes_rows_storeys_cols_modes` is stored as:
- **rows = storeys**
- **columns = saved modes**

Frequencies are global modal quantities, while mode shapes are vectors over the storeys for each mode.

### Current generator assumptions
- random mass per floor and per structure
- healthy and damaged stiffness bands do not overlap
- no precomputed population graph is provided
