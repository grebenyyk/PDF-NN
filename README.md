# Explainable Machine Learning Insights into Molecular Clusters Nuclearity via Pair Distribution Function

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Neural network-based classification of pair distribution function (PDF) data for predicting nuclearity of lanthanide compounds.

## Overview

This repository contains the code and trained models for classifying PDF patterns of:

- **Model CexOy clusters** - Cerium-oxide polynuclear clusters with varying nuclearities (CeO2 and Ce40 parent structures)

- **CSD crystal structures** - Crystalline compounds from the Cambridge Structural Database

The workflow enables prediction of nuclearity from PDF data.

## Repository Structure

```structure
PDF-NN/
├── config.py                 # Path configuration (MODIFY THIS FIRST)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
│
├── complete workflow/        # Main workflow notebooks (START HERE)
│   ├── 1-create-clusters.ipynb     # Create model clusters from parent structure
│   ├── 2-prepare-PDF.ipynb         # Calculate PDFs from structures
│   ├── 3A-train-model-ceo2.ipynb   # Train model on CeO2 cluster PDFs
│   ├── 3B-train-model-ce40.ipynb   # Train model on Ce40 cluster PDFs
│   ├── 3C-train-model-CSD.ipynb    # Train model on CSD structure PDFs
│   ├── 4-predict.ipynb             # Predict from experimental PDFs
│   ├── descriptors/                # PDF/RDF descriptor analysis
│   │   ├── A-descriptors-CeO2.ipynb
│   │   ├── B-descriptors-Ce40.ipynb
│   │   └── C-descriptors-CSD.ipynb
│   └── architecture_comparison/    # Model architecture experiments
│       ├── 3C-train-model-CSD-minimal.ipynb
│       ├── 3C-train-model-CSD-with-attention.ipynb
│       └── README.md
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

```structure
pdf-nn-data/
├── ceo2_clusters/
│   ├── model_clusters/       # .xyz files of CeO2-based clusters
│   └── calculated_pdfs/      # .dat files with calculated PDFs
│
├── ce_clusters/
│   ├── model_clusters/       # .xyz files of Ce40-based clusters
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
    ├── ce_clusters/          # Trained models for Ce clusters
    └── csd/                  # Trained models for CSD structures
```

## Installation

Tested on:
- **OS:** Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Python:** 3.7.16
- **TensorFlow:** 2.11.0

### 1. Clone the repository

```bash
git clone https://github.com/grebenyyk/PDF-NN.git
cd PDF-NN
```

### 2. Create a conda environment

```bash
conda create -n pdfnn python=3.7
conda activate pdfnn
```

### 3. Install dependencies

```bash
# Install DiffPy-CMI
conda install -c diffpy diffpy-cmi

# Install tensorflow
pip install tensorflow

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
   - Generates CexOy clusters of varying sizes from a parent structure (CeO2 or Ce40)

2. **Calculate PDFs** (`2-prepare-PDF.ipynb`):
   - Computes PDF patterns for model clusters and CSD structures
   - Prepares experimental PDFs for analysis

3. **Train the model** (`3A-train-model-ceo2.ipynb`, `3B-train-model-ce40.ipynb`, or `3C-train-model-CSD.ipynb`):
   - Hyperparameter tuning with Keras Tuner
   - Cross-validation training
   - Model evaluation

4. **Make predictions** (`4-predict.ipynb`):
   - Load trained model
   - Predict nuclearity from experimental PDFs

### Using pre-trained models

To use the pre-trained models for prediction:

```python
from config import get_path
import tensorflow.keras as keras
from keras.utils import custom_object_scope
from keras_self_attention import SeqSelfAttention

# Load pre-trained model
if use_attention:
    with custom_object_scope({'SeqSelfAttention': SeqSelfAttention}):
        load_model = keras.models.load_model(best_model_path)
else:
    load_model = keras.models.load_model(best_model_path)

# Your prediction code here...
```

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{PDF-NN,
  title={Explainable Machine Learning Insights into Molecular Clusters Nuclearity via Pair Distribution Function},
  author={Grebenyuk et al.},
  journal={},
  year={2026},
  doi={}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PDF calculations performed using [DiffPy-CMI](https://www.diffpy.org/)
- Crystal structures obtained from [Cambridge Structural Database](https://www.ccdc.cam.ac.uk/)
