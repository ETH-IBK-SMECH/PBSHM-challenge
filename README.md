# PBSHM Coding Challenge

**Chair of Structural Mechanics and Monitoring | ETH Zurich**

## Overview

This repository contains a small take-home coding exercise on **Population-Based Structural Health Monitoring (PBSHM)**.

You will work with a simulated population of 50 shear-frame structures with a **variable number of storeys (4-8)**. The goal is to detect whether a structure is damaged and, if time permits, explore whether graph-based learning can support **damage localization** through node- or edge-level damage indicators.

## What Is Simulated vs. What You Should Use

The dataset was generated from an `N`-DOF lumped-mass shear-frame model with localized stiffness reduction used to simulate damage. We provide the generation script so the physical assumptions are transparent.

However, for the purposes of this exercise, assume that in deployment you **do not directly know the true damaged stiffness values or other hidden simulator state**. Your method should be built from the provided measurement-like quantities, graph structure, and labels.

In other words:

- the simulation model is disclosed for physical clarity
- the inference task should use the provided candidate-facing files
- do not treat hidden physical parameters as measured inputs at rollout time

## Dataset

The dataset files are located in this repository.

### Files

| File | Description |
|---|---|
| `structures_measurements.json` | Candidate-facing inputs: one entry per structure with graph topology and measurement-like node features |
| `structure_labels.csv` | Structure-level detection label and true damaged storey for evaluation / optional localization analysis |
| `population_edges_geometry.csv` | Starter population graph built from geometry-only summaries |
| `population_edge_weights_geometry.csv` | Same edges with cosine similarity weights |
| `population_metadata.json` | Dataset summary and field descriptions |
| `generate_dataset_revised.py` | Reference generator showing how the synthetic data was created |

### Structure of `structures_measurements.json`

Each structure contains:

- `structure_id`
- `n_storeys`
- `edges`
- `node_features`
- `feature_names`

Each node currently includes:

- `storey`
- `height_m`
- `dominant_modal_frequency_Hz`

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
- Explore the starter population graph
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

Implement a graph-based model that uses the **within-structure graph** and compare it to your simpler baselines.

You may:

- perform graph-level damage detection
- estimate node- or edge-level damage indicators
- or do both

If you choose a GNN, a sensible pattern is:

1. encode each structure graph
2. pool node information into a structure representation for detection
3. inspect node embeddings or scores for approximate localization

The emphasis is on whether the graph formulation is well-motivated and interpretable.

### Task 5 - Population-level graph extension 

Use the population graph to explore transfer across structures.

- Build your own graphs
- Test whether population information helps with detection on unseen structures
- Reflect on when population message passing helps and when it may mislead

You do not need to complete this task fully to produce a strong submission.

## Deliverables

1. Runnable code in your preferred language
2. A short `README` explaining how to run the work and key design decisions
3. A short slide deck for a 10-minute presentation

## What We Are Assessing

We care about:

- clear and reproducible code
- sensible handling of variable-size graphs
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

- The provided population graph is a **starter graph** based on geometry-only summaries. You are free to modify or replace it.
- The generation script is included for transparency, but your inference pipeline should rely on the provided candidate-facing inputs.
- This is a toy exercise. A clean and well-argued small solution is better than an overly ambitious one.
