# Readme
This repository contains all of the code used in my MRN 412/422 research project.
This project was completed towards obtaining a degree in Mechanical Engineering
at the University of Pretoria.

## Project Information
**Topic:** Comparing the Performance of Principal Component Analysis~(PCA) and Independent Component Analysis~(ICA) as Unsupervised Latent Variable Models for Anomaly Detection in Time-Series Signals

**Name**: Ian de Villiers (18030123)

**Supervisor**: Prof. D.N. Wilke

## Setup
It is advised that a virtual environment (conda, virtualenv, or the like) be set up within which the project requirements can be installed without interfering with existing package installations on the target computer.

**Python version used:** 3.9.12

**NOTE:** This project requires `pathlib`, which added to the standard library in Python 3.4. The code will not run on Python < 3.4.

1. Open a Terminal window or Command Prompt/PowerShell.
2. Navigate to the root of this repository.
3. Install the required modules using `pip`:
```
python -m install -r requirements.txt
```
4. Install ```gm-utils-v2```:
```
python -m install -e gm-utils-v2
```

## Project Structure
<!-- | `research`           | Python code for generating all the plots shown in the report | -->
| Folder | Description |
|--------|-------------|
| `sampling` | Arduino code used to record low-resolution data |
| `research/inv-1-param-shifts` | Python code for generating all the plots for the numerical investigations |
| `research/inv-2-published` | Python code for generating all the plots for the investigations using published data |
| `research/inv-3-experiments` | Python code for generating all the plots for the experimental data |
