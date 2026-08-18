# Code for "Self-organizing physical and biochemical interactions explain diverse behaviours in *Physarum polycephalum*"

This repository contains simulation code and analysis for the preprint:

**Self-organizing physical and biochemical interactions explain diverse behaviours in *Physarum polycephalum***

Linnéa Gyllingberg, Abid Haque, Subash K. Ray, Gregory Weber, Jason M. Graham, and Simon Garnier

bioRxiv, 2026
https://doi.org/10.64898/2026.05.07.723662

The repository is organised into separate subdirectories corresponding to the different modelling approaches used in the paper.

## Repository structure

```text
Slime_mould_project_code/
├── 1D_Physarum/
├── PhaseField_Physarum/
└── README.md
```

## `1D_Physarum/`

Julia code and notebooks for the one-dimensional model of *Physarum* dynamics.

This folder contains a notebook reproducing the numerical experiments for the one-dimensional model presented in the paper.

### Running the simulations

Open the notebook and run the cells in order from top to bottom, or use **Run All**.

## `PhaseField_Physarum/`

Julia code and Jupyter notebooks for the two-dimensional phase-field model of *Physarum* morphology and dynamics.

This folder contains notebooks for implementing, simulating, and analysing the phase-field model presented in the paper.

### Running the simulations

Open the relevant notebook and run the cells in order from top to bottom, or use **Run All**.

The phase-field simulations are computationally intensive and use GPU acceleration. A CUDA-compatible NVIDIA GPU is required to reproduce the GPU-accelerated simulations.

## Software requirements

The code in this repository uses:

* **Julia**, for both the one-dimensional and two-dimensional models and simulations
* **Jupyter notebooks**, for running simulations, analysis, and visualisation
* **CUDA**, for GPU acceleration of the two-dimensional phase-field simulations

Each subdirectory may have its own environment and package requirements.

## Reproducibility notes

* Julia dependencies are specified using `Project.toml` files in the relevant subdirectories.
* Notebooks are intended to be run from within their respective folders.
* Cells should generally be executed in order from top to bottom.
* Some parts of the analysis notebooks may assume that simulation data has already been generated.
* Large simulation outputs are not tracked in this repository.

## Citation

If you use this code, please cite:

> Gyllingberg, L., Haque, A., Ray, S. K., Weber, G., Graham, J. M., & Garnier, S. (2026). *Self-organizing physical and biochemical interactions explain diverse behaviours in Physarum polycephalum*. bioRxiv. https://doi.org/10.64898/2026.05.07.723662

## Contact

For questions about the code or manuscript, please contact:

**Linnéa Gyllingberg**
https://linneagyllingberg.github.io/
