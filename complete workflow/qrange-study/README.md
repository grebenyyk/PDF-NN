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

## Quick start: sweep driver (recommended)

The **sweep driver** (`qrange_sweep.py`) automates the per-pair loop so you don't
need to manually change `PAIR_INDEX` in every notebook. Run from the project root:

```bash
# 1. Recalculate all PDFs for all pairs (idempotent — skips completed pairs)
python "complete workflow/qrange-study/qrange_sweep.py" prepare

# 2. Dry run first to check disk estimates
python "complete workflow/qrange-study/qrange_sweep.py" prepare --dry-run

# 3. Train all datasets across all pairs
python "complete workflow/qrange-study/qrange_sweep.py" train --dataset all

# 4. Aggregate metrics + comparison figures across pairs
python "complete workflow/qrange-study/qrange_sweep.py" compare

# Or do everything in sequence:
python "complete workflow/qrange-study/qrange_sweep.py" all
```

### Running a subset of pairs

To add one new pair without re-running everything:

1. Add the pair to `QRANGE_PAIRS` in `config.py`.
2. `python qrange_sweep.py prepare --pairs 5`  (only the new index)
3. `python qrange_sweep.py train --pairs 5`
4. `python qrange_sweep.py compare`  (aggregates all completed pairs)

### Aggregated comparison

`qrange_compare.py` produces a summary table (CSV + LaTeX) and multi-panel
figures in `results/qrange_comparison/`:

- `qrange_metrics.csv` / `.tex` — accuracy and F1 per dataset per pair.
- `qrange_comparison.png` — bar chart comparing metrics across pairs.
- `qrange_pdf_overlay.png` — PDF shape overlay showing how representative
  PDFs change with Q-range.

These are the figure/table the manuscript needs to answer the referee's question
on Q-dependence of calculated PDFs.

## Notebooks (manual per-pair workflow)

The notebooks remain available for interactive single-pair exploration. For the
full sweep, prefer the driver above.

| Notebook | Role |
|----------|------|
| `2-prepare-PDF_qrange.ipynb` | Recalculates CeO2 + Ce40 cluster PDFs for **all** pairs in `QRANGE_PAIRS` (loops). |
| `cif-preparePDF_qrange.ipynb` | Recalculates CSD PDFs for **one** pair (`PAIR_INDEX`). Run once per pair. |
| `3A-train-model-ceo2_qrange.ipynb` | Train CeO2 model for one pair (`PAIR_INDEX`). |
| `3B-train-model-ce40_qrange.ipynb` | Train Ce40 model for one pair (`PAIR_INDEX`). |
| `3C-train-model-CSD_qrange.ipynb` | Train CSD (no-attention) model for one pair (`PAIR_INDEX`). |
| `A/B/C-descriptors-*_qrange.ipynb` | Descriptor analysis for one pair (`PAIR_INDEX`). |

### Running a pair manually

1. Run `2-prepare-PDF_qrange.ipynb` once — it writes cluster PDFs for every pair.
2. For each pair you want to study, set `PAIR_INDEX` in `cif-preparePDF_qrange.ipynb`
   and run it (writes that pair's CSD PDFs + exclusion list).
3. For each pair, set `PAIR_INDEX` in `3A/3B/3C_qrange.ipynb` and run end-to-end.
4. Optionally run the matching descriptors notebook with the same `PAIR_INDEX`.

`PAIR_INDEX` indexes into `config.QRANGE_PAIRS`; the tag is derived automatically
and used for every output filename/dir, so there is no risk of cross-pair collisions.

### Automated training with papermill

For headless training without manual `PAIR_INDEX` editing:

```bash
pip install papermill
for i in $(seq 0 4); do
    papermill 3A-train-model-ceo2_qrange.ipynb output_ceo2_pair$i.ipynb -p PAIR_INDEX $i
    papermill 3B-train-model-ce40_qrange.ipynb output_ce40_pair$i.ipynb -p PAIR_INDEX $i
    papermill 3C-train-model-CSD_qrange.ipynb  output_csd_pair$i.ipynb  -p PAIR_INDEX $i
done
```

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
- The sweep driver and comparison module write to the **same** tagged output
  directories as the notebooks — they are fully interchangeable.
