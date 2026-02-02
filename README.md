# Explainable Machine Learning Insights into Molecular Clusters Nuclearity via Pair Distribution Function

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Neural network-based classification of pair distribution function (PDF) data for predicting nuclearity of metal-oxide clusters.

## Overview

This repository contains the code and trained models for classifying PDF patterns of:
- **Model ThxOy clusters** - Thorium-oxide polynuclear clusters with varying nuclearities
- **Model CexOy clusters** - Cerium-oxide polynuclear clusters
- **CSD crystal structures** - Crystalline compounds from the Cambridge Structural Database

The workflow enables prediction of cluster nuclearity from PDF data.

## Repository Structure

```
PDF-NN/
├── config.py                 # Path configuration (MODIFY THIS FIRST)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── complete_workflow/        # Main workflow notebooks (START HERE)
│   ├── 1-create-clusters.ipynb     # Create model clusters from parent structure
│   ├── 2-prepare-PDF.ipynb         # Calculate PDFs from structures
│   ├── 3-train-model-th.ipynb      # Train model on Th cluster PDFs
│   ├── 3b-train-model-ce.ipynb     # Train model on Ce cluster PDFs
│   ├── 4-train-model-CSD.ipynb     # Train model on CSD structure PDFs
│   ├── 5-predict.ipynb             # Predict from experimental PDFs
│   ├── descriptors-*.ipynb         # PDF/RDF descriptor analysis
│   └── architecture_comparison/    # Model architecture experiments
│
└── utils/                    # Utility notebooks
    ├── cif-preparePDF.ipynb  # CIF file PDF preparation
    └── plot-models.ipynb     # Model visualization

```

## Data

The data associated with this publication is available on Zenodo:

**DOI:** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

### Data Structure

After downloading, extract the data to your preferred location and update `config.py`:

```
pdf-nn-data/
├── th_clusters/
│   ├── model_clusters/       # .xyz files of ThxOy clusters
│   └── calculated_pdfs/      # .dat files with calculated PDFs
│
├── ce_clusters/
│   ├── model_clusters/       # .xyz files of CexOy clusters
│   └── calculated_pdfs/      # .dat files with calculated PDFs
│
├── csd_structures/
│   ├── cifs/                 # .cif files from CSD
│   └── calculated_pdfs/      # .dat files with calculated PDFs
│
├── experimental_pdfs/
│   ├── raw/                  # Original .gr files
│   └── processed/            # Interpolated _processed.gr files
│
└── models/
    ├── th_clusters/          # Trained models for Th clusters
    └── csd/                  # Trained models for CSD structures
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PDF-NN.git
cd PDF-NN
```

### 2. Create a conda environment

```bash
conda create -n pdfnn python=3.9
conda activate pdfnn
```

### 3. Install dependencies

```bash
# Install DiffPy-CMI (required for PDF calculations)
conda install -c diffpy diffpy-cmi

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download and configure data

1. Download data from Zenodo: [DOI_LINK]
2. Extract to a location of your choice
3. Edit `config.py` and set `DATA_ROOT` to your data location:

```python
DATA_ROOT = "/path/to/your/pdf-nn-data"
```

Or set via environment variable:
```bash
export PDF_NN_DATA_ROOT="/path/to/your/pdf-nn-data"
```

### 5. Verify installation

```bash
python config.py
```

This will check that all data paths are accessible.

## Quick Start

### Using the complete workflow

The recommended way to use this code is through the notebooks in `complete_workflow/`:

1. **Create model clusters** (`1-create-clusters.ipynb`):
   - Generates ThxOy or CexOy clusters of varying sizes from a parent structure

2. **Calculate PDFs** (`2-prepare-PDF.ipynb`):
   - Computes PDF patterns for model clusters and CSD structures
   - Prepares experimental PDFs for analysis

3. **Train the model** (`3-train-model-th.ipynb`, `3b-train-model-ce.ipynb`, or `4-train-model-CSD.ipynb`):
   - Hyperparameter tuning with Keras Tuner
   - Cross-validation training
   - Model evaluation

4. **Make predictions** (`5-predict.ipynb`):
   - Load trained model
   - Predict nuclearity from experimental PDFs

### Using pre-trained models

To use the pre-trained models for prediction:

```python
from config import get_path
import keras
from keras.utils import custom_object_scope
from keras_self_attention import SeqSelfAttention

# Load pre-trained model
model_path = get_path('th_models') / 'best_model.hdf5'
with custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
    model = keras.models.load_model(model_path)

# Your prediction code here...
```

## Model Architecture

The neural network uses a 1D CNN with self-attention:

- **Input**: Normalized PDF pattern (1000 points, r = 2-12 Å)
- **Conv1D** layers for feature extraction
- **Self-Attention** layer for capturing long-range correlations
- **Dense** layers with dropout for classification
- **Output**: Softmax classification (nuclearity classes)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{YOUR_PAPER,
  title={Machine Learning Classification of PDF Data for Metal-Oxide Clusters},
  author={Your Name et al.},
  journal={Journal Name},
  year={2026},
  doi={YOUR_DOI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PDF calculations performed using [DiffPy-CMI](https://www.diffpy.org/)
- Crystal structures obtained from [Cambridge Structural Database](https://www.ccdc.cam.ac.uk/)
