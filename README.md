# Titanic Data Journey — D100 Exercise 4

## Project Overview

This project refactors a Jupyter Notebook based on the Kaggle Titanic dataset into a more structured and maintainable Python repository.

The Jupyter Notebook remains the main entry point for presenting the analysis, while reusable functions are organised into separate Python modules.

## Repository Structure

- `Titanic_Data_Journey.ipynb` — Main notebook containing the analysis workflow.
- `functions/` — Reusable Python modules and functions.
- `data/` — Input datasets used in the analysis.
- `configs/` — Project configuration files.
- `outputs/` — Directory for generated outputs.
- `environment.yml` — Conda environment specification.
- `.github/` — GitHub-related project configuration.

## Getting Started

1. Clone the repository to your local computer.
2. Set up a compatible Python environment using the dependencies specified in `environment.yml`. The environment file contains a machine-specific `prefix` path, which may need to be removed or adjusted for your system.
3. Open `Titanic_Data_Journey.ipynb` in VS Code or Jupyter.
4. Select the appropriate Python kernel.
5. Check the dataset paths and run the notebook cells in order.

## Development and Collaboration

The project follows a modular development workflow. Reusable code is organised into Python modules, while the notebook is retained for presenting the analysis.

Contributors should create separate Git branches for their changes, open Pull Requests for review, and merge approved changes into the main branch.