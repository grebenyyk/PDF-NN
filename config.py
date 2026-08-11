"""
Configuration file for PDF-NN project.
This file manages all paths and settings for the project.

Users should modify the DATA_ROOT path to point to their local data directory.
The data can be downloaded from Zenodo at: [DOI_LINK_HERE]
"""

import os
from pathlib import Path

# =============================================================================
# USER CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# Get the directory where this config file is located
_CONFIG_DIR = Path(__file__).parent.resolve()

# Root directory for all data files
# Download data from Zenodo and extract to this location
# Example: DATA_ROOT = "/path/to/your/pdf-nn-data"
DATA_ROOT = os.environ.get("PDF_NN_DATA_ROOT", None)
if DATA_ROOT is None:
    #DATA_ROOT = Path("/workspace/home/pdf-nn-data")
    DATA_ROOT = Path('/Volumes/Extreme SSD/WORK/PDF/PDF-NN paper/PDF-NN/pdf-nn-data')
else:
    DATA_ROOT = Path(DATA_ROOT)

# =============================================================================
# DERIVED PATHS - DO NOT MODIFY UNLESS CHANGING DATA STRUCTURE
# =============================================================================

# Model clusters (CexOy)
PATHS = {
    # Cerium oxide clusters (Ce40 parent)
    "ce_clusters": DATA_ROOT / "ce_clusters",
    "ce_model_clusters": DATA_ROOT / "ce_clusters" / "model_clusters",
    "ce_calculated_pdfs": DATA_ROOT / "ce_clusters" / "calculated_pdfs",
    
    # Cerium oxide clusters (CeO2 parent)
    "ceo2_clusters": DATA_ROOT / "ceo2_clusters",
    "ceo2_model_clusters": DATA_ROOT / "ceo2_clusters" / "model_clusters",
    "ceo2_calculated_pdfs": DATA_ROOT / "ceo2_clusters" / "calculated_pdfs",
    
    # CSD crystal structures
    "csd_structures": DATA_ROOT / "csd_structures",
    "csd_cifs": DATA_ROOT / "csd_structures" / "cifs",
    "csd_calculated_pdfs": DATA_ROOT / "csd_structures" / "calculated_pdfs",
    
    # Experimental PDFs
    "experimental_pdfs": DATA_ROOT / "experimental_pdfs",
    "experimental_processed": DATA_ROOT / "experimental_pdfs" / "processed",
    
    # Trained models
    "models": DATA_ROOT / "models",
    "ce_models": DATA_ROOT / "models" / "ce_clusters",
    "csd_models": DATA_ROOT / "models" / "csd",
    
    # Results and outputs
    "results": DATA_ROOT / "results",
    "figures": DATA_ROOT / "results" / "figures",
    
    # Labels
    "labels": DATA_ROOT / "labels",
}


# =============================================================================
# QRANGE STUDY (additive) - explore alternative (qmin, qmax) PDF-calculation
# values without overwriting the baseline (qmin=0.3, qmax=11) published work.
#
# The existing PATHS dict above is unchanged: the baseline .dat files in the
# bare calculated_pdfs/ directories are never touched by the qrange-study
# notebooks. Each (qmin, qmax) pair writes to its own tagged, namespaced
# directory (calculated_pdfs_qX-Y/, models/qX-Y/, ...) so multiple qranges
# coexist on disk.
# =============================================================================

# (qmin, qmax) used in the published work - implicit in the existing PATHS
BASELINE_QRANGE = (0.3, 11.0)

# (qmin, qmax) pairs explored on the qrange-study branch
QRANGE_PAIRS = [
    (0.0, 11.0),
    (1.0, 11.0),
    (0.3, 25.0),
    (0.3, 20.0),
    (0.0, 25.0),
]


def qrange_tag(qmin, qmax):
    """Filesystem-safe tag for a (qmin, qmax) pair, e.g. (0.3, 11) -> 'q0.3-11'."""
    return f"q{qmin:g}-{qmax:g}"


def qrange_paths(qmin, qmax):
    """
    Namespaced data/model/result paths for a given (qmin, qmax) pair.

    Mirrors the keys used by get_path() so qrange-study notebooks can swap
    get_path('key') -> qrange_paths(qmin, qmax)['key']. Every output path is
    tagged so different qranges coexist on disk and never overwrite the
    baseline. Read-only inputs (model_clusters, cifs) are shared, not tagged.
    """
    tag = qrange_tag(qmin, qmax)
    return {
        # Shared, read-only inputs (not tagged)
        "ceo2_model_clusters": DATA_ROOT / "ceo2_clusters" / "model_clusters",
        "ce_model_clusters": DATA_ROOT / "ce_clusters" / "model_clusters",
        "csd_cifs": DATA_ROOT / "csd_structures" / "cifs",
        "ceo2_clusters": DATA_ROOT / "ceo2_clusters",
        "ce_clusters": DATA_ROOT / "ce_clusters",
        "csd_structures": DATA_ROOT / "csd_structures",
        # Tagged, per-qrange outputs
        "ceo2_calculated_pdfs": DATA_ROOT / "ceo2_clusters" / f"calculated_pdfs_{tag}",
        "ce_calculated_pdfs": DATA_ROOT / "ce_clusters" / f"calculated_pdfs_{tag}",
        "csd_calculated_pdfs": DATA_ROOT / "csd_structures" / f"calculated_pdfs_{tag}",
        "models": DATA_ROOT / "models" / tag,
        "ce_models": DATA_ROOT / "models" / tag / "ce_clusters",
        "csd_models": DATA_ROOT / "models" / tag / "csd",
        "labels": DATA_ROOT / "labels" / tag,
        "results": DATA_ROOT / "results" / tag,
        "figures": DATA_ROOT / "results" / tag / "figures",
        # Shared experimental data (qrange-independent)
        "experimental_pdfs": DATA_ROOT / "experimental_pdfs",
        "experimental_processed": DATA_ROOT / "experimental_pdfs" / "processed",
    }


def get_path(key: str) -> Path:
    """
    Get a path by key name.
    
    Parameters
    ----------
    key : str
        The path key (e.g., 'ce_clusters', 'experimental_pdfs')
    
    Returns
    -------
    Path
        The absolute path to the requested directory
    
    Raises
    ------
    KeyError
        If the path key is not found
    """
    if key not in PATHS:
        raise KeyError(f"Path key '{key}' not found. Available keys: {list(PATHS.keys())}")
    return PATHS[key].resolve()


def ensure_paths_exist():
    """Create all data directories if they don't exist."""
    for key, path in PATHS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {key}: {path}")


def validate_paths():
    """
    Validate that required data paths exist.
    
    Returns
    -------
    dict
        Dictionary with path keys and their existence status
    """
    status = {}
    all_exist = True
    
    print("Checking data paths...")
    print("-" * 60)
    
    for key, path in PATHS.items():
        exists = path.exists()
        status[key] = exists
        symbol = "✓" if exists else "✗"
        print(f"{symbol} {key}: {path}")
        if not exists:
            all_exist = False
    
    print("-" * 60)
    if all_exist:
        print("All paths exist!")
    else:
        print("Some paths are missing. Run ensure_paths_exist() to create them,")
        print("or download data from Zenodo and update DATA_ROOT in config.py")
    
    return status


# Convenience function to set up working directory for a notebook
def setup_workdir(path_key: str):
    """
    Change to the specified working directory.
    
    Parameters
    ----------
    path_key : str
        The path key to change to
    
    Example
    -------
    >>> from config import setup_workdir
    >>> setup_workdir('ce_clusters')
    """
    import os
    target = get_path(path_key)
    os.chdir(target)
    print(f"Working directory: {target}")


if __name__ == "__main__":
    print("PDF-NN Configuration")
    print("=" * 60)
    print(f"DATA_ROOT: {DATA_ROOT.resolve()}")
    print()
    validate_paths()
