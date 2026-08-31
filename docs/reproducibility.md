# ⚙️ Reproducibility Guide & Computational Environment Setup

This repository is designed to guarantee complete computational reproducibility for all experimental findings, tables, and figures reported in the research paper.

---

## 1. System & Environment Requirements

- **Operating System**: Windows / Linux / macOS
- **Python Version**: Python 3.10+ (Tested on Python 3.14)
- **Random Seed**: Fixed globally to `42` across all models and sampling procedures.

---

## 2. Environment Setup Instructions

### Option A: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate supply_chain_research
```

### Option B: Using Pip
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Step-by-Step Experiment Execution

To reproduce all experiments, generate CSV metric tables, and export publication figures:

```powershell
python main.py
```

Or run modular scripts individually:

```powershell
# 1. Run Data Preprocessing & Scoping Audit
python scripts/run_full_pipeline.py
```

All generated tables will be saved in `results/metrics/` and `results/tables/`, and figures in `figures/` and `visuals/`.
