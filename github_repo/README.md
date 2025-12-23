# An Adaptive Governance Framework for Mitigating Fairness Drift in Healthcare AI

This repository contains the open-source code and resources for the dissertation "An Adaptive Governance Framework for Mitigating Fairness Drift in Healthcare AI" by Zemen Ghelaw.

## Abstract

Artificial intelligence (AI) models in healthcare are susceptible to performance degradation and fairness drift over time. This project proposes and validates a comprehensive, adaptive governance framework (AGF) to mitigate these risks through continuous monitoring, fairness auditing, and stakeholder engagement. The AGF integrates five core components: real-time drift detection, multi-metric fairness auditing, theorem-guided constrained retraining, deliberative stakeholder forums, and public transparency reporting. Our simulations and real-world validation on the MIMIC-IV dataset demonstrate that the AGF significantly outperforms baseline approaches in both maintaining fairness and improving overall model accuracy.

## Features

- **Real-Time Drift Detection**: Algorithms to continuously monitor model performance and fairness metrics.
- **Fairness Auditing Dashboard**: A modular dashboard for visualizing disparities and triggering alerts.
- **Constrained Retraining**: A novel retraining procedure guided by the Fairness Preservation Boundary Theorem.
- **Stakeholder Engagement Module**: Tools and templates for conducting deliberative stakeholder forums.
- **Public Transparency Portal**: A web-based portal for public reporting of model performance.

## Getting Started

### Prerequisites

- Python 3.9+
- Pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zemen-ghelaw/adaptive-governance-framework-dissertation.git
   cd adaptive-governance-framework-dissertation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the main simulation, execute the following command:

```bash
python src/run_simulation.py --config configs/simulation_config.yaml
```

To generate the visualizations from the dissertation, you can run the individual scripts in the `src/visualizations` directory. For example, to generate Figure 4:

```bash
python src/visualizations/generate_figure_4.py
```

## Repository Structure

```
├── configs/              # Configuration files for simulations
├── data/                 # Placeholder for datasets (MIMIC-IV should be downloaded separately)
├── notebooks/            # Jupyter notebooks for exploratory analysis
├── src/                  # Source code
│   ├── components/       # Core components of the AGF
│   ├── visualizations/   # Scripts to generate figures
│   └── run_simulation.py # Main simulation script
├── LICENSE               # MIT License
└── README.md             # This file
```

## Citation

If you use this code or framework in your research, please cite the dissertation:

```
Ghelaw, Z. (2025). An Adaptive Governance Framework for Mitigating Fairness Drift in Healthcare AI. Capitol Technology University.
```

## Contact

Zemen Ghelaw - [zemen.ghelaw@captechu.edu](mailto:zemen.ghelaw@captechu.edu)

Project Link: [https://github.com/zemen-ghelaw/adaptive-governance-framework-dissertation](https://github.com/zemen-ghelaw/adaptive-governance-framework-dissertation)
