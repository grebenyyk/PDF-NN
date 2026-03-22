# Explainable Machine Learning Insights into Molecular Clusters Nuclearity via Pair Distribution Function

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Neural network-based classification of pair distribution function (PDF) data for predicting nuclearity of lanthanide compounds.

## Quick Start

The quickest way to use the neural network trained within this work for nuclearity prediction from your experimental PDF data is the `quick-predict.py` script:

```bash
# Using the default model (default_model.h5 in the project root)
python quick-predict.py /path/to/your/gr-files

# Using a specific model
python quick-predict.py /path/to/your/gr-files --model /path/to/model.h5

# Custom output path (default: predictions.csv in the input directory)
python quick-predict.py /path/to/your/gr-files --output /path/to/results.csv
```

The script handles preprocessing (header detection, interpolation to the model grid, normalization) and prediction. It recursively searches the input directory for `.gr` files, so all subdirectories are included. Input files must cover the 2–12 Å r-range; files that don't will be skipped with a warning. The default model is `csd-3.h5` which is trained in the `3C-train-model-CSD.ipynb` notebook on the calculated PDF data from the CSD crystal structures. The model `csd-3-minimal.h5` was trained on the same dataset but has a significantly more simple architecture (30 times less trainable parameters). While demonstrating similar test accuracy as the large model on a CSD dataset of calculated PDF data, its performance on real data is not on par with a `csd-3.h5`, so the latter is recommended for real-life usage. Refer to the text of the paper for details.

## Overview

The repository contains the code for developing the machine learning models do predict the nuclearity from PDF data of:

- **Model CexOy clusters** - Cerium-oxide polynuclear clusters with varying nuclearities (CeO2 and Ce40 parent structures)

- **CSD crystal structures** - Crystalline compounds from the Cambridge Structural Database

## Repository Structure

```structure
PDF-NN/
├── config.py                 # Path configuration (needs to be modified before running)
├── quick-predict.py          # Use trained networks for nuclearity prediction
├── *.h5                      # CSD-trained models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
│
├── complete workflow/              # Main workflow notebooks for all the training
│   ├── 1-create-clusters.ipynb     # Create model clusters from parent structure
│   ├── 2-prepare-PDF.ipynb         # Calculate PDFs from structures
│   ├── 3A-train-model-ceo2.ipynb   # Train model on CeO2-based clusters PDFs
│   ├── 3B-train-model-ce40.ipynb   # Train model on Ce40-based clusters PDFs
│   ├── 3C-train-model-CSD.ipynb    # Train model on CSD structure PDFs
│   ├── descriptors/                # Structure-related PDF descriptors analysis
│   │   ├── A-descriptors-CeO2.ipynb
│   │   ├── B-descriptors-Ce40.ipynb
│   │   └── C-descriptors-CSD.ipynb
│   └── architecture_comparison/    # Model architecture experiments
│       ├── 3C-train-model-CSD-minimal.ipynb
│       └── 3C-train-model-CSD-with-attention.ipynb
│
├── experimental data/        # Experimental PDF data used in the paper
│
└── utils/                    # Utility notebooks
    ├── cif-preparePDF.ipynb  # PDF preparation from .cif files
    └── plot-models.ipynb     # Model visualization

```

## Data

The trained models associated with this publication, calculated PDF data and clean notebooks for results reproduction are available on Zenodo:

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
│   └── calculated_pdfs/      # .dat files with calculated PDFs from CSD .cif files
│
└── models/
    ├── ce_clusters/          # Trained models for Ce40-based clusters
    ├── ceo2_clusters/        # Trained models for CeO2-based clusters
    ├── csd_minimal/          # Simple CNN models for CSD structures
    ├── csd_no_attention/     # CNN without attention layer for CSD structures
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

### Use the complete workflow

1. **Create model clusters** (`1-create-clusters.ipynb`):
   - Generates CexOy clusters of varying sizes from a parent structure (CeO2 or Ce40)

2. **Calculate PDFs** (`2-prepare-PDF.ipynb`):
   - Computes PDF patterns for model clusters and CSD structures
   - Prepares experimental PDFs for analysis

3. **Train the model** (`3A-train-model-ceo2.ipynb`, `3B-train-model-ce40.ipynb`, or `3C-train-model-CSD.ipynb`):
   - Hyperparameter tuning with Keras Tuner
   - Cross-validation training
   - Model evaluation

4. **Make predictions** (`quick-predict.ipynb`):
   - Predict nuclearity from experimental PDFs

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
