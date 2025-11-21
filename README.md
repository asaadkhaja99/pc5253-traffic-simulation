# Agent-Based modeling of Traffic Flow

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Mesa](https://img.shields.io/badge/Mesa-3.0+-green.svg)](https://mesa.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Final project for PC 5253 at the National University of Singapore.

Team Members:
- Li Xiaoyue
- Davendra Shayna Hassan
- Shaik Asaaduddin Khwaja
- Jam Lambert Ubay Catenza
- Prerthan Munireternam

---

## Introduction
Traffic congestion is a widespread problem that requires modeling to understand. Agent-based
models represent individual vehicles/drivers and their decision rules, allowing them to be used for modelling a variety of scenarios in urban planning and evacuation settings.

### Key Objectives
The goal of this project is to survey the following questions:
1. Emergence detection:  Under what critical density and disturbance conditions does free flow
transition to congestion?
2. Sensitivity analysis: How does varying agent parameters (reaction time, desired speed) affect
flow stability?
3. Scenario testing: How do different road layouts and interventions affect the performance of a
traffic network?

### Main Findings
1. Using the OVM, IDM, Nagel-Schreckenberg and MOBIL models, and bridging them with geospacial data from the Singapore traffic network, we demonstrated that traffic congestion is an emergent phase transition
2. We established the Cogestion Order Parameter $\chi$ as a robust metric for representing traffic states across different roads, and quantified the influence of points of interest (POIs) on traffic flow
3. Using the Nagel-Schreckenberg and IDM model, we demonstrated the utility of agent-based models in modelling traffic under various evacuation and urban planning scenarios

---

## Repository Structure

```
.
├── README.md                   # This file
├── pyproject.toml             # Python dependencies (managed with uv)
│
├── presentation/              # Presentation materials
│   ├── Group 1 Traffic.pdf    # Final presentation slides
│   ├── hysteresis/            # Hysteresis phenomenon demonstrations
│   │   ├── critical_points/   # Critical density analysis
│   │   ├── idm_model/         # IDM hysteresis plots
│   │   ├── nasch_model/       # NaSch hysteresis plots
│   │   └── ovm_model/         # OVM hysteresis plots
│   ├── models/                # Model demonstrations
│   │   ├── presentation_nasch.py           # NaSch implementation
│   │   ├── presentation_ovm.ipynb          # OVM notebook
│   │   └── traffic_combined_figure.png    # Combined model comparison
│   └── singapore_traffic_network/         # Singapore network analysis
│       ├── actual_traffic.ipynb           # Empirical data analysis
│       ├── sg_traffic_bando_NEW[2].ipynb # Bando OVM on Singapore network
│       ├── sg_traffic_sim_interactive.ipynb  # Interactive simulation
│       └── traffic_critical_comparison.ipynb # Critical point comparison
│
└── report/                    # Report-related code and analysis
    ├── empirical_network_characterisation/
    │   └── bando_traffic_mesageo_new.ipynb  # Singapore network characterization
    │
    ├── model_demonstration/   # Model validation figures
    │   ├── Fig1_nasch.ipynb   # NaSch fundamental diagram
    │   └── Fig2_OVM.ipynb     # OVM fundamental diagram
    │
    ├── scenario_testing/      # Application scenarios
    │   ├── evacuation_model/  # Evacuation scenario implementation
    │   │   ├── core/          # Core evacuation model components
    │   │   │   ├── evacuation_base.py      # Base evacuation model
    │   │   │   └── plot_utils.py           # Visualization utilities
    │   │   ├── scenarios/     # Different evacuation scenarios
    │   │   │   ├── contraflow_intervention.py  # Contraflow strategy
    │   │   │   ├── scenario_simultaneous.py    # Simultaneous evacuation
    │   │   │   ├── scenario_staged.py          # Staged evacuation
    │   │   │   └── scenario_staged_contraflow.py  # Staged + contraflow
    │   │   ├── analysis/      # Analysis and metrics
    │   │   │   ├── analyze_evacuation.py       # Evacuation analysis
    │   │   │   └── calculate_tet_metrics.py    # Total Evacuation Time
    │   │   ├── runners/       # Execution scripts
    │   │   │   ├── run_evacuation_study.py     # Run evacuation experiments
    │   │   │   └── run_spatial_visualization.py  # Spatial visualizations
    │   │   ├── visualization/ # Plotting and visualization
    │   │   │   ├── create_comparison_plots.py  # Compare scenarios
    │   │   │   ├── create_evacuation_zone_map.py  # Zone maps
    │   │   │   ├── create_intervention_effectiveness.py  # Intervention plots
    │   │   │   ├── visualize_evacuation.py     # Main visualization
    │   │   │   └── visualize_spatial.py        # Spatial plots
    │   │   └── README.md                       # Documentation
    │   │
    │   └── urban_planning/    # Urban planning scenario (Paya Lebar incident)
    │       ├── urban_base.py              # Base urban planning model
    │       ├── urban_road_idm.py          # IDM road agent with Gaussian bottleneck
    │       ├── scenario_paya_lebar.py     # Paya Lebar localized incident
    │       ├── create_scenario_maps.py    # Generate scenario maps
    │       ├── visualize_urban.py         # Urban planning visualizations
    │       ├── plot_utils.py              # Shared plotting utilities
    │       ├── run_urban_study.py         # Execute urban planning study
    │       ├── output/                    # Simulation outputs
    │       └── README.md                  # Urban planning documentation
    │
    └── sensitivity_analysis/  # Parameter sensitivity studies
        ├── idm_timegap_sensitivity_results/  # IDM time gap sensitivity
        └── nasch_sensitivity_results/        # NaSch parameter sensitivity
```

---

## Setup Instructions

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

### Prerequisites

Install uv using one of the following methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**With pip:**
```bash
pip install uv
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pc5253-traffic-simulation.git
cd pc5253-traffic-simulation
```

2. Sync dependencies and create virtual environment:
```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies specified in the project.

### Setting Up Jupyter Kernel

To use this project's virtual environment in Jupyter notebooks:

1. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install ipykernel if not already installed:
```bash
uv pip install ipykernel
```

3. Create a Jupyter kernel for this project:
```bash
python -m ipykernel install --user --name=pc5253-traffic --display-name "Python (pc5253-traffic)"
```

4. Now you can select the "Python (pc5253-traffic)" kernel in Jupyter notebooks or JupyterLab.

To remove the kernel later:
```bash
jupyter kernelspec uninstall pc5253-traffic
```
---

## Mapping to Paper/Presentation

This section maps repository directories to specific sections in our paper and presentation slides.

### Paper Sections

| Paper Section/Figure | Repository Location | Description |
|---------------|---------------------|-------------|
| Section 2.2/Figure 1 | `report/model_demonstration/Fig1_nasch.ipynb` | Fundamental diagram for NaSch |
| Section 2.3/Figure 2 | `report/model_demonstration/Fig2_OVM.ipynb` | OVM Simulation on ring road |
| Section 3.3.1/Figure 3, Section 4.4/Figure 16 | `report/scenario_testing/evacuation_model` | Evacuation Scenario Testing Setup |
| Section 3.3.2/Figure 4, Section 4.4/Figure 17 | `report/scenario_testing/urban_planning` | Urban Planning Scenario Testing Setup |
| Section 4.1/Figure 5,6,7,8 | `presentation/hysteresis` | Hysteresis plots on Fundamental Diagrams |

### Presentation Slides

* Model Plots: `presentation/models`
* Critical Point Analysis Plots: `presentation/hysteresis`
* Map visualisations of actual data and IDM, OVM and Nagel-Schreckenberg models

---

## Technology Stack

- **Agent-Based Modeling**: [Mesa](https://mesa.readthedocs.io/) 3.0+
- **Geospatial Analysis**: [Mesa-Geo](https://mesa-geo.readthedocs.io/), [OSMnx](https://osmnx.readthedocs.io/)
- **Network Analysis**: [NetworkX](https://networkx.org/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Contextily](https://contextily.readthedocs.io/)
- **Package Management**: [uv](https://docs.astral.sh/uv/)

---

## Ethics, Limits and Reuse
No personal data is processed. TSB coverage focuses on expressways/arterials; local streets are under-observed. We gratefully acknowledge the use of generative AI tools, including ChatGPT (OpenAI), Gemini 2.5/3 Pro (Google) and Claude (Antrophic) which were utilized for tasks such as literature search, preliminary drafting, code troubleshooting, and proofreading during the preparation of this report. All final content was critically reviewed and edited to ensure accuracy and originality.

---

## Acknowledgments

We acknowledge LTA DataMall and OpenStreetMap contributors. Code builds on the Mesa, Mesa-Geo, and OSMnx ecosystems.
