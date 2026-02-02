# PDF-NN: Publication Preparation Guide

This document provides guidance on organizing the repository and data for publication.

## Repository Organization

### Files to INCLUDE in the GitHub repository

The following files should be included in the published repository:

```
PDF-NN/
├── README.md                           # Project documentation
├── LICENSE                             # MIT License (already exists)
├── requirements.txt                    # Python dependencies
├── config.py                           # Path configuration
├── .gitignore                          # Git ignore rules
│
├── complete_workflow/                  # MAIN WORKFLOW (cleaned up)
│   ├── 1-create-clusters.ipynb
│   ├── 2-prepare-PDF.ipynb
│   ├── 3-train-model-th.ipynb
│   ├── 3b-train-model-ce.ipynb
│   ├── 4-train-model-CSD.ipynb
│   ├── 5-predict.ipynb
│   ├── descriptors-*.ipynb             # PDF/RDF descriptor analysis
│   └── architecture_comparison/        # Model architecture experiments
│
└── utils/                              # Utility notebooks
    ├── cif-preparePDF.ipynb
    └── plot-models.ipynb
```

### Files to EXCLUDE from the GitHub repository

The following are work-in-progress files and should NOT be in the final repo:

| File | Reason |
|------|--------|
| `train_real_clusters_all_data.ipynb` | Superseded by complete_workflow |
| `train_calc_clusters_calc_data.ipynb` | Superseded by complete_workflow |
| `train_sliced_calc_clusters_calc_data.ipynb` | Superseded by complete_workflow |
| `train_onehot.ipynb` | Experimental, not used in paper |
| `cross_train_*.ipynb` | Intermediate experiments |
| `hyperparameter_*.ipynb` | Superseded by complete_workflow |
| `GOOD_*.ipynb` | Intermediate versions → merge into complete_workflow |
| `csd_ simpler_model.ipynb` | Experimental |
| `prepareCe.ipynb`, `prepareTh.ipynb`, `preparePDF.ipynb`, `preparePDFexp.ipynb` | Superseded by `2_prepare_PDF.ipynb` |
| `cif_preparePDF.ipynb` | Merge into `2_prepare_PDF.ipynb` |
| `predict.ipynb` | Superseded by `5_predict.ipynb` |
| `plot_model.ipynb` | Can keep if adds value, or extract to utils |
| `shap.ipynb` | Keep if SHAP analysis is in paper |

### Recommended actions

1. **Move work-in-progress files** to a separate branch or archive folder (not tracked)
2. **Consolidate** the best code from GOOD_* notebooks into the complete_workflow
3. **Clean** complete_workflow notebooks: clear outputs, remove debugging cells

---

## Data Organization for Zenodo

### What to upload to Zenodo

Create a ZIP archive with this structure:

```
pdf-nn-data-v1.0.zip
│
├── README_DATA.txt              # Data description
│
├── th_clusters/
│   ├── model_clusters/          # ~10,000 .xyz files (ThxOy clusters)
│   │   ├── 1_0001.xyz
│   │   ├── 2_0001.xyz
│   │   └── ...
│   ├── calculated_pdfs/         # Corresponding .dat files
│   │   ├── 1_0001.dat
│   │   └── ...
│   └── structure_catalogue.txt  # Cluster generation catalogue
│
├── ce_clusters/
│   ├── model_clusters/          # CexOy cluster .xyz files
│   ├── calculated_pdfs/         # Corresponding .dat files
│   └── structure_catalogue.txt
│
├── csd_structures/
│   ├── cifs/                    # CIF files from CSD
│   │   ├── REFCODE1.cif
│   │   └── ...
│   └── calculated_pdfs/         # Calculated .dat files
│
├── experimental_pdfs/
│   ├── raw/                     # Original experimental .gr files
│   └── processed/               # Interpolated _processed.gr files
│
├── models/                      # Pre-trained models
│   ├── th_clusters/
│   │   ├── best_model.hdf5      # Best performing model
│   │   └── fold_*_model.hdf5    # Cross-validation models
│   └── csd/
│       ├── best_model.hdf5
│       └── fold_*_model.hdf5
│
└── labels/                      # Label files
    ├── th_labels.txt
    ├── ce_labels.txt
    └── csd_labels.txt
```

### Zenodo metadata

```yaml
Title: "Data and Models for PDF-NN: Machine Learning Classification of PDF Data"

Description: |
  This dataset contains:
  - Model ThxOy and CexOy polynuclear cluster structures (.xyz)
  - Calculated PDF patterns from cluster structures (.dat)
  - CIF files from Cambridge Structural Database
  - Experimental PDF data
  - Pre-trained neural network models (.hdf5)
  
  Associated paper: [PAPER TITLE]
  Code repository: https://github.com/YOUR_USERNAME/PDF-NN

Keywords:
  - Pair Distribution Function
  - Machine Learning
  - Neural Networks
  - Thorium oxide clusters
  - Cerium oxide clusters

License: CC-BY-4.0

Related identifiers:
  - GitHub repository (isSupplementedBy)
  - Paper DOI (isSupplementTo)
```

---

## Updating Notebooks to Use config.py

### Before (hardcoded paths):
```python
os.chdir('/Users/dimitrygrebenyuk/Yandex.Disk.localized/Working/PDF/.../th_clusters')
```

### After (using config.py):
```python
from config import setup_workdir, get_path

# Option 1: Change working directory
setup_workdir('th_calculated_pdfs')

# Option 2: Get path without changing directory
data_path = get_path('th_calculated_pdfs')
files = list(data_path.glob('*.dat'))

# Option 3: For model loading
model_path = get_path('th_models') / 'best_model.hdf5'
model = keras.models.load_model(model_path)
```

---

## Checklist for Publication

### Repository preparation
- [ ] Remove all work-in-progress notebooks
- [ ] Update complete_workflow notebooks to use config.py
- [ ] Clear all notebook outputs (optional, but recommended)
- [ ] Test that notebooks run from a fresh environment
- [ ] Update README with correct DOI links
- [ ] Add proper citation information
- [ ] Verify LICENSE file is correct

### Data preparation
- [ ] Organize data into the structure above
- [ ] Remove any personal/sensitive file paths from data files
- [ ] Create README_DATA.txt with data description
- [ ] Upload to Zenodo and get DOI
- [ ] Update config.py with Zenodo DOI

### Final testing
- [ ] Clone repo to a new location
- [ ] Download data from Zenodo
- [ ] Configure DATA_ROOT
- [ ] Run all notebooks end-to-end
- [ ] Verify model predictions match paper results

---

## Version Control Strategy

### Branch structure
```
main              # Clean, publication-ready code
├── develop       # Active development (optional)
└── archive       # Work-in-progress files (not published)
```

### Tagging
```bash
# After paper acceptance
git tag -a v1.0 -m "Version accompanying publication DOI:XXX"
git push origin v1.0
```

This tag can be linked to both the paper and Zenodo deposit.
