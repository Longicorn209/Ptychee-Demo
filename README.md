# Ptychee-Demo

Authors:
- Zeyu Wang
- Xiaoyan Wu

A lightweight demonstration toolbox for 4D-STEM data analysis and reconstruction, developed in Python and PyTorch.

This repository contains a collection of reconstruction and visualization routines for 4D-STEM datasets, including ptychographic reconstruction and other diffraction-based imaging methods.

The code is provided as a compact demonstration package together with a sample dataset and four Jupyter notebook tutorials.

## Features

Current demonstration modules include:

- 4D-STEM data preprocessing
- Diffraction pattern rotation estimation
- Center-of-Mass (CoM) and integrated CoM (iCoM) analysis
- Virtual detector imaging
- Simple parallax reconstruction (tcBF)
- Ptychographic reconstruction:<br>
    analytical WDD and SSB methods<br>
    iterative LSQML and ePIE methods (mixed-state and multi-slice implementations)

## Installation
```bash
git clone https://github.com/Longicorn209/Ptychee-Demo

cd Ptychee-Demo

pip install -r requirements.txt
```

## Tested Environment
- Python 3.11
- CUDA 12.4
- PyTorch 2.1.1
- NumPy 1.26.4
- SciPy 1.16.0
- Matplotlib 3.10.3

## Quick Start

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open and execute all cells in each notebook:

- run_ptyChee_01_xxx.ipynb  
- run_ptyChee_02_xxx.ipynb  
- run_ptyChee_03_xxx.ipynb  
- run_ptyChee_04_xxx.ipynb

## Dataset

A small example dataset is provided in the `data/` directory for demonstration and testing.

The dataset is stored in compressed `.zip` format due to high sparsity.

Before running the demo, please unzip the dataset.

## Citation

If you use this repository in academic work, please cite the associated publication:

DOI: https://doi.org/10.1038/s41586-025-09693-6

## License

GPL 3.0 License.
