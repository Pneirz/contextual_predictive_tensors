# Contextual Predictive Tensors

Experiment code and generated artifacts for the contextual predictive tensor
experiments.

This repository contains only reproducibility materials: Python scripts,
processed outputs, raw public data used by the loaders, and generated
supplemental figures. Manuscript sources are intentionally not included.

## Layout

- `experiments/`: Python scripts used to generate the experiment tables and records.
- `data/raw/`: raw source data needed by the WDBC loader.
- `data/processed/`: generated CSVs that are small enough for GitHub.
- `data/large_archives/`: compressed CSV outputs that were larger than GitHub's 100 MB single-file limit when uncompressed.
- `figures/`: generated supplemental figures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Large data archives

Two raw order-3 CSVs exceed GitHub's single-file limit uncompressed:

- `null_swap_order3_v2.csv`
- `null_swap_order3_logreg.csv`

They are stored as ZIP archives under `data/large_archives/`. Most scripts that
only need to read `null_swap_order3_logreg.csv` can read the ZIP directly via
Pandas. To restore the uncompressed CSVs into `data/processed/`, run:

```bash
python experiments/unpack_large_data.py
```

## Reproducing artifacts

```bash
python experiments/null_swap_orders_monks1.py
python experiments/null_swap_order3_v2.py
python experiments/null_swap_order3_logreg.py
python experiments/null_swap_orders_wdbc.py
python experiments/exp5_sota_comparison.py
python experiments/exp6_mixed_dgp.py
python experiments/exp7_xor3_benchmark.py
python experiments/exp8_logloss_consistency.py
python experiments/exp9_scm_causal_control.py
```

The order-3 and WDBC scripts are the expensive runs. The current generated CSVs
are included so the experiment outputs can be inspected without rerunning
everything.
