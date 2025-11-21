# Urban Planning Module

Agent-based simulation using the Intelligent Driver Model (IDM) with Gaussian bottleneck on Singapore's road network.

---

## Overview

This module implements an urban planning scenario for Singapore's Paya Lebar area, testing network performance under localized disruptions.

---

## Module Structure

```
urban_planning/
├── __init__.py                  # Module initialization
├── urban_base.py                # Base urban planning model with disruption support
├── urban_road_idm.py            # IDM road agent with Gaussian bottleneck implementation
├── scenario_paya_lebar.py       # Paya Lebar localized incident scenario
├── create_scenario_maps.py      # Generate scenario visualization maps
├── visualize_urban.py           # Plotting and analysis visualizations
├── plot_utils.py                # Shared plotting utilities
├── run_urban_study.py           # Main execution script
└── README.md                    # This file
```

---

## File Descriptions

### Core Components

#### `urban_base.py`
Base framework for urban planning simulations containing:
- **`UrbanPlanningConfig`**: Configuration dataclass with network bounds and simulation parameters
- **`DisruptionConfig`**: Configuration for network disruptions (Gaussian bottleneck parameters)
- **`UrbanRoadAgent`**: Road segment agent with disruption support
- **`UrbanVehicleAgent`**: Vehicle agent with origin-destination routing
- **`UrbanPlanningModel`**: Main Mesa model class

**Key Classes:**
```python
@dataclass
class UrbanPlanningConfig:
    # Network bounds, vehicle counts, IDM parameters

@dataclass
class DisruptionConfig:
    # Gaussian bottleneck parameters (ε, σ, position)

class UrbanRoadAgent(EvacuationRoadAgent):
    # Road segment with IDM and disruption support

class UrbanPlanningModel(EvacuationModel):
    # Main simulation model
```

#### `urban_road_idm.py`
IDM implementation with Gaussian bottleneck factor:
- **`gaussian_bottleneck_factor()`**: Computes B(x) = 1 - ε·exp[-(x-x_incident)²/(2σ²)]
- **`IDMRoadAgentWithBottleneck`**: Road agent with IDM dynamics and bottleneck
- **`idm_acceleration()`**: Intelligent Driver Model acceleration calculation

**Gaussian Bottleneck Equation (Eq. 17):**
```
B(x) = 1 - ε·exp[-(x-x_incident)²/(2σ²)]

where:
  ε = bottleneck strength (0-1), 0.9 = 90% capacity reduction
  σ = spatial spread (meters), controls bottleneck width
  x = position along road (meters)
  x_incident = bottleneck center position
```

#### `scenario_paya_lebar.py`
Paya Lebar localized incident scenario:
- Lane closure near Paya Lebar MRT interchange
- Baseline vs. incident comparison
- Queue spillback analysis
- Forced northbound flow through bottleneck zone

**Usage:**
```python
from scenario_paya_lebar import run_paya_lebar_scenario

# Run baseline
run_paya_lebar_scenario(
    num_vehicles=800,
    apply_gaussian_bottleneck=False,  # Baseline
    seed=42,
    output_file="output/paya_lebar_baseline.csv"
)

# Run with incident
run_paya_lebar_scenario(
    num_vehicles=800,
    apply_gaussian_bottleneck=True,   # With bottleneck
    seed=42,
    output_file="output/paya_lebar_closure.csv"
)
```

#### `visualize_urban.py`
Visualization and analysis tools:
- **`setup_plot_style()`**: Consistent plot styling
- **`plot_paya_lebar_comparison()`**: Creates multi-panel comparison plots
  - Trip completion comparison
  - Queue length over time
  - Congested roads count
  - Network speed degradation
  - Delay comparison bar chart
  - Peak queue length comparison

**Usage:**
```python
from visualize_urban import plot_paya_lebar_comparison

plot_paya_lebar_comparison(
    baseline_file="output/paya_lebar_baseline.csv",
    closure_file="output/paya_lebar_closure.csv",
    output_dir="output/urban_planning"
)
```

#### `create_scenario_maps.py`
Generates spatial visualization maps:
- Road network with hierarchy (major/secondary/residential)
- Origin and destination zones
- Paya Lebar MRT junction location
- Gaussian bottleneck zone visualization

**Usage:**
```bash
uv run python create_scenario_maps.py
```

#### `plot_utils.py`
Shared plotting utilities for consistent visualization styling:
- **`setup_high_res_plot_style()`**: High-resolution plot configuration
- **`COLORS`**: Consistent color scheme across plots

### Execution Scripts

#### `run_urban_study.py`
Main entry point for running the Paya Lebar scenario with baseline comparison.

**Usage:**
```bash
uv run python run_urban_study.py
```

---

## How to Run

### Prerequisites

Ensure project dependencies are installed:
```bash
cd /path/to/pc5253-traffic-simulation
uv sync
```

### Running the Paya Lebar Scenario

Navigate to the urban_planning directory:
```bash
cd report/scenario_testing/urban_planning
```

**Run Full Study (Baseline + Incident):**
```bash
uv run python run_urban_study.py
```

This runs:
1. Paya Lebar baseline (no bottleneck)
2. Paya Lebar with Gaussian bottleneck (ε=0.9)

Results saved to: `output/urban_planning/`

**Run Individual Scenario:**
```bash
uv run python scenario_paya_lebar.py
```

**Generate Visualization Maps:**
```bash
uv run python create_scenario_maps.py
```

**Generate Analysis Plots:**
```bash
uv run python visualize_urban.py
```

---

## Model Parameters

### Network Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `bbox_north` | 1.3280 | Northern boundary (latitude) |
| `bbox_south` | 1.3070 | Southern boundary (latitude) |
| `bbox_east` | 103.9040 | Eastern boundary (longitude) |
| `bbox_west` | 103.8810 | Western boundary (longitude) |

**Coverage Area**: Paya Lebar MRT vicinity (~2.1km × 2.3km)

### Simulation Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `num_vehicles` | 800 | Number of vehicles (attempted) |
| `delta_t` | 1.0 seconds | Time step duration |
| `max_steps` | 7200 | Maximum simulation steps (2 hours) |
| `seed` | 42 | Random seed for reproducibility |
| `use_idm` | True | Use IDM instead of NaSch |

### IDM Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `idm_a` | 1.0 m/s² | Maximum acceleration |
| `idm_b` | 1.5 m/s² | Comfortable deceleration |
| `idm_s0` | 2.0 m | Minimum gap |
| `idm_T` | 1.5 s | Safe time headway |
| `idm_delta` | 4.0 | Acceleration exponent |

### Gaussian Bottleneck Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `bottleneck_epsilon` (ε) | 0.9 | Bottleneck strength (90% capacity reduction) |
| `bottleneck_sigma` (σ) | 50.0 m | Spatial spread of bottleneck |
| `incident_position_m` | None | Position along road (None = midpoint) |

**Gaussian Bottleneck Formula:**
```
B(x) = 1 - ε·exp[-(x - x_incident)²/(2σ²)]

Effect on desired velocity:
v_desired(x) = v_max · B(x)
```

### Origin-Destination Zones

**Origin Zone (South of Incident):**
- Latitude: 1.3130 - 1.3160
- Longitude: 103.891 - 103.896
- Distance from MRT: 350-650m south

**Destination Zone (North of Incident):**
- Latitude: 1.3190 - 1.3220
- Longitude: 103.891 - 103.896
- Distance from MRT: 350-650m north

**Paya Lebar MRT Junction**: 1.31765°N, 103.89271°E

---

## Metrics

### Primary Metrics

1. **Delay**
   - Sum of all vehicle delays (vehicle-minutes)
   - Comparison: Baseline vs. Incident
   - Shows aggregate impact of bottleneck

2. **Queue Length**
   - Number of vehicles queued at bottleneck
   - Tracked over time
   - Peak queue length comparison

3. **Completed Trips**
   - Cumulative vehicles completing journeys
   - Completion rate comparison

4. **Mean Network Speed**
   - Average vehicle speed across all roads (km/h)
   - Indicates overall network performance

5. **Congested Roads Count**
   - Number of roads with degraded flow
   - Shows congestion propagation

---

## Output Files

All outputs are saved to `output/urban_planning/`:

### CSV Files (Time Series Data)

- `paya_lebar_baseline.csv` - Normal traffic (no bottleneck)
- `paya_lebar_closure.csv` - With Gaussian bottleneck

**Columns:**
- `time_step`: Simulation time (seconds)
- `completed_trips`: Cumulative trips completed
- `network_flow`: Vehicles per time step
- `mean_speed_kph`: Average speed (km/h)
- `congested_roads`: Count of congested roads
- `total_queue_length`: Total vehicles in queues
- `total_delay`: Cumulative delay (vehicle-seconds)

### Visualization Outputs

- `paya_lebar_scenario_map.png` - Scenario setup map with O-D zones
- `paya_lebar_comparison.png` - Multi-panel time series comparison
- `paya_lebar_impact.png` - Delay and queue length bar charts


## AI Tools
Claude code was used to assist with the implementation of this scenario. All final content was reviewed and edited to ensure accuracy.
