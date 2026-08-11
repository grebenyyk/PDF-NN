# qrange-study

Exploration of alternative `(qmin, qmax)` PDF-calculation values across the full
pipeline (everything except `architecture_comparison/`), as an **additive** layer
on top of the published baseline work. Nothing here overwrites the baseline.

## Pairs explored

Defined in `config.py` → `QRANGE_PAIRS`:

| qmin | qmax | tag       |
|------|------|-----------|
| 0.0  | 11.0 | `q0-11`   |
| 1.0  | 11.0 | `q1-11`   |
| 0.3  | 25.0 | `q0.3-25` |
| 0.3  | 20.0 | `q0.3-20` |
| 0.0  | 25.0 | `q0-25`   |

The published baseline `(qmin=0.3, qmax=11)` lives in the untagged
`calculated_pdfs/` directories and is never written to by these notebooks.

## How outputs are isolated

`config.qrange_paths(qmin, qmax)` returns namespaced, tagged directories for every
pair, e.g. for `(0.3, 25)` → `q0.3-25`:

- `calculated_pdfs_q0.3-25/`  (per dataset)
- `models/q0.3-25/...`
- `labels/q0.3-25/...`
- `results/q0.3-25/...`

So multiple pairs coexist on disk. The shared, read-only inputs (`model_clusters/`,
`csd_structures/cifs/`) are **not** tagged.

## Notebooks

| Notebook | Role |
|----------|------|
| `2-prepare-PDF_qrange.ipynb` | Recalculates CeO2 + Ce40 cluster PDFs for **all** pairs in `QRANGE_PAIRS` (loops). |
| `cif-preparePDF_qrange.ipynb` | Recalculates CSD PDFs for **one** pair (`PAIR_INDEX`). Run once per pair. |
| `3A-train-model-ceo2_qrange.ipynb` | Train CeO2 model for one pair (`PAIR_INDEX`). |
| `3B-train-model-ce40_qrange.ipynb` | Train Ce40 model for one pair (`PAIR_INDEX`). |
| `3C-train-model-CSD_qrange.ipynb` | Train CSD (no-attention) model for one pair (`PAIR_INDEX`). |
| `A/B/C-descriptors-*_qrange.ipynb` | Descriptor analysis for one pair (`PAIR_INDEX`). |

### Running a pair

1. Run `2-prepare-PDF_qrange.ipynb` once — it writes cluster PDFs for every pair.
2. For each pair you want to study, set `PAIR_INDEX` in `cif-preparePDF_qrange.ipynb`
   and run it (writes that pair's CSD PDFs + exclusion list).
3. For each pair, set `PAIR_INDEX` in `3A/3B/3C_qrange.ipynb` and run end-to-end.
4. Optionally run the matching descriptors notebook with the same `PAIR_INDEX`.

`PAIR_INDEX` indexes into `config.QRANGE_PAIRS`; the tag is derived automatically
and used for every output filename/dir, so there is no risk of cross-pair collisions.

## Notes

- `rmax` stays at 20 for all pairs, so the 2–12 Å slice used by the training
  notebooks (`skiprows=201, skipfooter=800` → 1000 points) remains valid.
  `qmin/qmax` affect PDF peak shape (termination ripples), not the r-grid sampling.
- Experimental PDF preparation is qrange-independent and is unchanged from the
  original `2-prepare-PDF.ipynb`.
- `1-create-clusters.ipynb` is reused as-is (cluster geometry does not depend on
  qrange).
- The original notebooks in `complete workflow/`, `utils/`, and
  `architecture_comparison/` are untouched on this branch.
