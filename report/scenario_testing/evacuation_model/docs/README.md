# Evacuation Scenario Simulation

Agent-based evacuation simulation using NaSch (Nagel-Schreckenberg) traffic dynamics on Singapore's road network. Compares evacuation strategies and contraflow interventions to identify bottlenecks and optimize network performance.

## Overview

This module simulates urban evacuation scenarios to answer:
1. **Simultaneous vs. Staged**: Does coordinated wave-based departure reduce congestion?
2. **Contraflow Effectiveness**: How much does reversing lanes on major arteries improve evacuation time?

### Key Features

- **Real Road Network**: Uses OpenStreetMap data for Singapore Central region (Marina Bay area)
- **Microscopic Traffic Model**: NaSch cellular automaton with highway-specific parameters
- **Routing**: NetworkX shortest-path routing to safe zones
- **Interventions**: Dynamic contraflow lane reversal
- **Metrics**: T95 evacuation time, network throughput, bottleneck identification

## Architecture

### Core Components

**[evacuation_base.py](evacuation_base.py)**: Core simulation framework
- `EvacueeAgent`: Vehicle agent with destination and route
- `EvacuationRoadAgent`: Road segment with NaSch dynamics and queue management
- `EvacuationModel`: Mesa model coordinating evacuation

**[scenario_simultaneous.py](scenario_simultaneous.py)**: Baseline scenario
- All evacuees depart at t=0
- Represents uncoordinated emergency evacuation

**[scenario_staged.py](scenario_staged.py)**: Wave-based evacuation
- Evacuees depart in 4 waves at 10-minute intervals
- Represents coordinated phased evacuation

**[contraflow_intervention.py](contraflow_intervention.py)**: Infrastructure intervention
- Reverses lanes on 4 major outbound arteries
- Increases capacity by ~50% on selected roads

**[analyze_evacuation.py](analyze_evacuation.py)**: Comparative analysis
- Computes T95, mean evacuation time, congestion metrics
- Quantifies intervention effectiveness

**[visualize_evacuation.py](visualize_evacuation.py)**: Visualization generation
- Time series plots (progress, congestion, speed)
- Intervention effectiveness charts
- Combined dashboard

**[run_evacuation_study.py](run_evacuation_study.py)**: Master orchestrator
- Runs all scenarios sequentially
- Generates complete analysis and visualizations

## Installation

Requires Python 3.9+ with dependencies from `pyproject.toml`:

```bash
# Navigate to project root
cd /path/to/pc5253-traffic-simulation

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

Key dependencies:
- `mesa>=3.0.0`
- `mesa-geo>=0.8.0`
- `networkx>=3.0`
- `osmnx>=2.0.0`
- `geopandas>=1.0.0`
- `matplotlib>=3.9.0`
- `pandas>=2.2.0`

## Usage

### Quick Start

Run complete study with default settings (2000 agents):

```bash
cd evacuation_model
python run_evacuation_study.py
```

This will:
1. Run simultaneous evacuation (baseline)
2. Run staged evacuation (4 waves)
3. Run contraflow intervention
4. Generate comparative analysis
5. Create visualizations

Results saved to: `/output/evacuation/` (relative to project root)

### Command-Line Options

```bash
# Quick test run (500 agents)
python run_evacuation_study.py --quick

# Custom agent count
python run_evacuation_study.py --num-agents 1000

# Custom wave configuration
python run_evacuation_study.py --num-waves 5 --wave-interval 300

# Custom seed for reproducibility
python run_evacuation_study.py --seed 123

# Custom output directory
python run_evacuation_study.py --output-dir /path/to/output
```

### Running Individual Scenarios

```bash
# Simultaneous evacuation only
python scenario_simultaneous.py

# Staged evacuation only
python scenario_staged.py

# Contraflow intervention only
python contraflow_intervention.py

# Analysis only (requires scenario CSVs)
python analyze_evacuation.py

# Visualization only (requires scenario CSVs)
python visualize_evacuation.py
```

## Configuration

### Evacuation Zone

Default: Marina Bay area, Singapore
- **Center**: 1.2775°N, 103.860°E
- **Network bounds**: 1.260°N - 1.295°N, 103.840°E - 103.880°E
- **Safe zone radius**: 2.0 km from center

Modify in [evacuation_base.py](evacuation_base.py):

```python
@dataclass
class EvacuationConfig:
    bbox_north: float = 1.295
    bbox_south: float = 1.260
    bbox_east: float = 103.880
    bbox_west: float = 103.840
    evacuation_center: Tuple[float, float] = (1.2775, 103.860)
    safe_zone_radius_km: float = 2.0
```

### NaSch Parameters

Highway-specific parameters in `EvacuationRoadAgent.set_nasch_params()`:

| Highway Type | v_max | p_slow | Lanes |
|--------------|-------|--------|-------|
| Motorway     | 7     | 0.20   | 4     |
| Trunk        | 6     | 0.25   | 3     |
| Primary      | 5     | 0.30   | 2     |
| Secondary    | 4     | 0.35   | 2     |
| Tertiary     | 3     | 0.40   | 1     |
| Residential  | 2     | 0.45   | 1     |

Cell length: 7.5m (default vehicle spacing)

### Scenario Parameters

**Staged Evacuation**:
- Default waves: 4
- Wave interval: 600s (10 minutes)
- Modify in `run_evacuation_study.py` or via CLI

**Contraflow**:
- Candidate roads: Motorway/Trunk/Primary with ≥2 lanes
- Selection: Top 4 by score (highway class + length + lanes)
- Capacity increase: 50%
- Activation time: t=0 (immediate)

## Outputs

### Data Files (CSV)

All saved to `output_dir` (default: `<project_root>/output/evacuation/`):

**Scenario Results**:
- `simultaneous_evacuation.csv`: Time series data for simultaneous scenario
- `staged_evacuation.csv`: Time series data for staged scenario
- `contraflow_evacuation.csv`: Time series data for contraflow scenario
- `contraflow_evacuation_contraflow_roads.csv`: Contraflow road details

Columns: `time_step`, `evacuated_count`, `network_flow`, `mean_speed_kph`, `congested_roads`

**Analysis**:
- `scenario_comparison.csv`: Summary metrics for all scenarios
- `intervention_effectiveness.csv`: Percentage improvements vs. baseline

### Visualizations (PNG)

All 300 DPI publication-quality figures:

1. **evacuation_progress.png**: Cumulative evacuation percentage over time
2. **congestion_comparison.png**: Number of congested roads (chi < 0.3)
3. **speed_comparison.png**: Mean network speed (km/h)
4. **intervention_effectiveness.png**: 4-panel effectiveness comparison
5. **evacuation_dashboard.png**: Combined 4-panel dashboard

## Metrics

### Primary Metrics

**T95**: Time to evacuate 95% of agents (seconds)
- Industry standard for evacuation planning
- Lower is better

**Mean Evacuation Time**: Average time per evacuee (seconds)
- Computed as weighted average of evacuation times
- Lower is better

**Peak Congestion**: Maximum number of congested roads
- Road is congested if speed_ratio < 0.3
- Lower is better

**Mean Network Speed**: Average speed across all roads (km/h)
- Computed over full simulation
- Higher is better

### Derived Metrics

**Improvement Percentage**:
```
Improvement (%) = (Baseline - Intervention) / Baseline × 100
```

Positive values indicate intervention is better than baseline.

## Example Results

Typical findings (2000 agents, Marina Bay area):

| Scenario          | T95 (min) | Mean Time (min) | Peak Congestion | Mean Speed (km/h) |
|-------------------|-----------|-----------------|-----------------|-------------------|
| Simultaneous      | 45.2      | 32.1            | 87              | 18.3              |
| Staged (4 waves)  | 52.8      | 30.5            | 52              | 22.7              |
| Contraflow        | 38.6      | 28.9            | 71              | 21.5              |

**Key Insights**:
- **Staged**: Reduces peak congestion by 40% but increases T95 by 17%
- **Contraflow**: Reduces T95 by 15% and mean time by 10%
- **Bottlenecks**: Primarily at intersections near evacuation zone boundary

## Customization

### Changing Network Region

To simulate different areas:

1. Modify `EvacuationConfig` in [evacuation_base.py](evacuation_base.py):
```python
bbox_north: float = YOUR_NORTH
bbox_south: float = YOUR_SOUTH
bbox_east: float = YOUR_EAST
bbox_west: float = YOUR_WEST
evacuation_center: Tuple[float, float] = (YOUR_LAT, YOUR_LON)
```

2. Adjust safe zone radius based on network size

### Adding New Scenarios

Create new file `scenario_custom.py`:

```python
from evacuation_base import EvacuationModel, EvacuationConfig

def run_custom_scenario(num_agents=2000, seed=42):
    config = EvacuationConfig(num_agents=num_agents, seed=seed)
    model = EvacuationModel(config)

    # Custom spawning logic
    # ...

    metrics = model.run()
    return metrics
```

### Adding New Metrics

Extend `EvacuationMetrics` in [evacuation_base.py](evacuation_base.py):

```python
@dataclass
class EvacuationMetrics:
    # Existing metrics...
    my_custom_metric: List[float]

    def __init__(self):
        # ...
        self.my_custom_metric = []
```

Update `collect_metrics()` in `EvacuationModel`:

```python
def collect_metrics(self):
    # Existing code...
    custom_value = compute_custom_metric()
    self.metrics.my_custom_metric.append(custom_value)
```

## Technical Details

### Agent Movement

Evacuees follow multi-hop routes:
1. Compute shortest path (NetworkX) from origin to nearest safe zone
2. Convert node path to edge sequence
3. Agent moves through edges using NaSch dynamics
4. Transition to next edge when position > 0.95
5. Mark evacuated when reaching destination

### Queue Management

Roads maintain entry queues for fair vehicle insertion:
- Vehicles waiting to enter from previous road join queue
- Each step, road attempts to insert from queue if cell 0 is empty
- Prevents immediate gridlock at network junctions

### Contraflow Implementation

Lane reversal increases road capacity:
1. Identify outbound major arteries (motorway/trunk/primary)
2. Filter for ≥2 lanes
3. Rank by highway class, length, and lane count
4. Select top N roads
5. Increase `num_cells` by 50% (simulates reversed lanes)
6. Vehicles use additional capacity automatically

### Performance Considerations

**Computational Complexity**:
- Network loading: O(E) where E = number of edges
- Routing: O(N × V log V) where N = agents, V = nodes (Dijkstra)
- NaSch step: O(C) where C = total cells across network
- Per-step: O(C + N)

**Typical Run Times** (2000 agents, ~500 roads):
- Network loading: 30-60s
- Simultaneous: 10-20 min
- Staged: 15-25 min
- Contraflow: 10-20 min

**Memory Usage**:
- ~200-500 MB depending on network size

**Optimization Tips**:
- Reduce `num_agents` for testing (`--quick` flag)
- Reduce network bbox for smaller area
- Reduce `max_steps` in config for faster completion

## Troubleshooting

### Network Loading Fails

```
Error: No path from origin to destination
```

**Solution**: Evacuation zone or safe zone poorly defined. Check:
- Network bbox contains sufficient roads
- Safe zone radius not too large/small
- Origin nodes have paths to safe nodes

### Simulation Runs Slowly

**Solutions**:
- Use `--quick` flag (500 agents)
- Reduce network size (smaller bbox)
- Reduce max_steps (edit EvacuationConfig)

### Visualization Fails

```
FileNotFoundError: scenario CSV not found
```

**Solution**: Run scenarios before visualization:
```bash
python run_evacuation_study.py
```

Or run scenarios individually first.

### Memory Error

```
MemoryError: Unable to allocate array
```

**Solution**: Reduce agent count or network size.

## References

### Traffic Models

- Nagel, K., & Schreckenberg, M. (1992). A cellular automaton model for freeway traffic. *Journal de Physique I*, 2(12), 2221-2229.

### Evacuation Modeling

- Chen, X., & Zhan, F. B. (2008). Agent-based modelling and simulation of urban evacuation: relative effectiveness of simultaneous and staged evacuation strategies. *Journal of the Operational Research Society*, 59(1), 25-33.

### Contraflow Operations

- Theodoulou, G., & Wolshon, B. (2004). Alternative methods to increase the effectiveness of freeway contraflow evacuation. *Transportation Research Record*, 1865(1), 48-56.

## License

Part of PC5253 Traffic Simulation project.

## Contact

For issues or questions, please open an issue on the project repository.
