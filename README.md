# Privacy-Preserving Industrial Electric-Carbon Synthetic Data Generation

This repository packages the code, and the dataset used for a study on privacy-preserving synthetic data generation for industrial electric-carbon coupled data. The project focuses on generating realistic tabular data that preserves electric-carbon coupling structure, supports conditional synthesis for specific operating regimes, and evaluates privacy risk with distance-based audits.

## Overview

The repository includes:
- a two-stage synthetic data generation pipeline for industrial electric-carbon data
- comparative experiments with TVAE and CTGAN baselines
- conditional generation experiments for specific `Load_Type` regimes
- privacy audits based on DCR, NNDR, and distance-based membership-inference-style analysis
- the final IEEE-style paper source and figures
- a packaged dataset under `data/raw/`

## Repository Structure

```text
.
├── code/
│   ├── experiment_two_stage_v3.py
│   ├── run_comprehensive_experiment.py
│   ├── run_privacy_audit.py
│   ├── conditional_generation.py
│   └── generate_paper_figures.py
├── data/
│   └── raw/
│       └── Steel_industry_data.csv
├── docs/
│   └── RESEARCH_BRIEF.md
├── paper/
│   ├── paper_ieee.tex
│   ├── paper_ieee.pdf
│   ├── IEEEtran.cls
│   ├── reference/
│   │   └── references.bib
│   └── figures/papers/
├── results/
│   ├── comprehensive_results.json
│   ├── conditional_results.json
│   ├── two_stage_results_v3.json
│   ├── synthetic_tvae.csv
│   ├── synthetic_ctgan.csv
│   └── synthetic_conditional.csv
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Dataset

The packaged scripts now read the dataset from:

```text
data/raw/Steel_industry_data.csv
```

## Environment Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Scripts

### 1. Two-stage generation experiment

```bash
python code/experiment_two_stage_v3.py
```

This script runs the refined two-stage generation pipeline with validation-derived physics bounds and filtered synthetic outputs.

### 2. Comprehensive experiment

```bash
python code/run_comprehensive_experiment.py
```

This script runs the broader evaluation pipeline, including fidelity, utility, and privacy-related metrics.

### 3. Privacy audit

```bash
python code/run_privacy_audit.py
```

This script performs the dedicated privacy audit using DCR, NNDR, and distance-threshold analysis.

### 4. Conditional generation

```bash
python code/conditional_generation.py
```

This script generates targeted synthetic samples for specific `Load_Type` conditions and evaluates conditional generation utility.

### 5. Regenerate paper figures

```bash
python code/generate_paper_figures.py
```

This script regenerates paper figures from the packaged dataset and selected result artifacts.

## Notes on Paths

Run the packaged scripts from the repository root. They expect:
- dataset under `data/raw/Steel_industry_data.csv`
- generated outputs and cached artifacts under `results/`
- paper assets under `paper/`

## Paper

The final paper assets are under `paper/`:
- LaTeX source: `paper/paper_ieee.tex`
- Compiled PDF: `paper/paper_ieee.pdf`
- Bibliography: `paper/reference/references.bib`
- Figures: `paper/figures/papers/`

If your LaTeX environment is configured, compile from the `paper/` directory.

## Citation

If you use this repository, please cite the accompanying paper once the bibliographic metadata is finalized.

## License

This packaged repository currently uses an MIT license placeholder for the code and documentation in this release package. Verify dataset and paper redistribution requirements before public release.
