# Urban Planning Simulation

Agent-based models for testing network resilience to localized disruptions and long-term closures. Demonstrates ABM utility for urban planning decisions by modeling:

1. **Localized Incidents**: Queue spillback from lane closures (Paya Lebar case study)
2. **Long-Term Disruptions**: Rat-running behavior from major road closures (PIE case study)

## Overview

While evacuation scenarios test network-wide demand, urban planning scenarios focus on:
- **Localized disruptions**: Common incidents (accidents, construction) affecting specific road segments
- **Congestion propagation**: How bottlenecks create queue spillback
- **Emergent behavior**: Rat-running (drivers diverting to residential roads)
- **Network resilience**: System adaptation to disruptions

### Scenarios

**1. Paya Lebar Localized Incident**
- Replicates methodology from Othman et al. (2023) SUMO study
- Partial lane closure near Paya Lebar MRT interchange
- Validates ABM's ability to reproduce realistic queue spillback
- **Output**: Congestion metrics, queue lengths, trip time comparison

**2. PIE Long-Term Disruption**
- Simulates 1km closure on Pan-Island Expressway (PIE)
- Tests emergence of rat-running behavior
- Identifies residential roads receiving diverted traffic
- **Output**: Rat-running statistics, residential road usage analysis

## Architecture

### Core Components

**[urban_base.py](urban_base.py)**: Base framework
- `UrbanPlanningModel`: Mesa model for urban traffic
- `UrbanRoadAgent`: Road with disruption support (lane/road closures)
- `UrbanVehicleAgent`: Commuter vehicles with OD pairs
- `DisruptionConfig`: Configuration for network disruptions

**[scenario_paya_lebar.py](scenario_paya_lebar.py)**: Localized incident
- Lane closure near Paya Lebar MRT
- Queue spillback analysis
- Baseline vs. closure comparison

**[scenario_pie_closure.py](scenario_pie_closure.py)**: Long-term disruption
- PIE section closure (1km)
- Rat-running detection
- Residential road usage analysis

**[run_urban_study.py](run_urban_study.py)**: Master orchestrator
- Runs both scenarios with baselines
- Generates comparative analysis

## Installation

Same dependencies as evacuation model (already installed):

```bash
cd /path/to/pc5253-traffic-simulation
uv sync
```

## Usage

### Quick Start

Run both scenarios with default settings:

```bash
cd urban_planning
uv run python run_urban_study.py
```

This runs:
1. Paya Lebar baseline + lane closure (800 vehicles each)
2. PIE baseline + road closure (1500 vehicles each)

Results saved to: `/output/urban_planning/`

### Quick Test

Reduced vehicle counts for faster testing (~10-15 min):

```bash
uv run python run_urban_study.py --quick
```

### Individual Scenarios

```bash
# Paya Lebar only
uv run python scenario_paya_lebar.py

# PIE only
uv run python scenario_pie_closure.py
```

### Custom Configuration

```bash
# Custom vehicle counts
uv run python run_urban_study.py --paya-lebar-vehicles 1000 --pie-vehicles 2000

# Custom seed
uv run python run_urban_study.py --seed 123
```

## Configuration

### Paya Lebar Scenario

Location: East Singapore, near Paya Lebar MRT
- **Network bounds**: 1.305°N - 1.325°N, 103.885°E - 103.905°E
- **Vehicles**: 800 (default)
- **Disruption**: 50% capacity reduction on major road near MRT
- **Duration**: Permanent (full simulation)
- **Focus**: Queue spillback, trip time increase

Modify in [scenario_paya_lebar.py](scenario_paya_lebar.py:34-51).

### PIE Scenario

Location: Wider area covering PIE section
- **Network bounds**: 1.310°N - 1.360°N, 103.830°E - 103.890°E
- **Vehicles**: 1500 (default)
- **Disruption**: Complete closure of 1km PIE section
- **Duration**: Permanent (long-term construction)
- **Focus**: Rat-running, residential road usage

Modify in [scenario_pie_closure.py](scenario_pie_closure.py:40-56).

## Outputs

All saved to `<project_root>/output/urban_planning/`:

### Paya Lebar

**CSV Files**:
- `paya_lebar_baseline.csv` - Normal traffic (no closure)
- `paya_lebar_closure.csv` - With lane closure

Columns: `time_step`, `completed_trips`, `network_flow`, `mean_speed_kph`, `congested_roads`

### PIE

**CSV Files**:
- `pie_baseline.csv` - Normal traffic
- `pie_closure.csv` - With PIE closure
- `pie_baseline_ratrunning.csv` - Residential road usage (baseline)
- `pie_closure_ratrunning.csv` - Residential road usage (with closure)

Rat-running CSV columns: `edge`, `throughput`, `total_passed`, `max_queue`, `highway_type`

## Metrics

### Paya Lebar (Queue Spillback)

**Primary Metrics**:
- **Mean trip time**: Average travel time per vehicle
- **Peak congestion**: Maximum number of congested roads simultaneously
- **Queue length**: Number of vehicles waiting at bottleneck

**Expected Findings**:
- Lane closure increases mean trip time by 20-40%
- Queue spillback creates congestion on upstream roads
- Peak congestion occurs shortly after disruption starts

### PIE (Rat-Running)

**Primary Metrics**:
- **Residential fraction**: % of total traffic using residential roads
- **Top rat-run roads**: Residential roads with highest throughput
- **Throughput comparison**: Baseline vs. closure

**Expected Findings**:
- PIE closure increases residential fraction by 10-30 percentage points
- Specific residential roads become "shortcut" routes
- Primary road network shows reduced flow, residential roads show increased flow

### Comparison

| Metric | Baseline | Disruption | Change |
|--------|----------|------------|--------|
| **Paya Lebar** ||||
| Mean trip time | ~300s | ~400s | +33% |
| Peak congestion | ~20 roads | ~50 roads | +150% |
| Queue length | 0 | ~30 veh | - |
| **PIE** ||||
| Residential fraction | ~15% | ~35% | +20 pp |
| Completed trips | ~95% | ~85% | -10% |
| Mean speed | ~40 km/h | ~25 km/h | -38% |

## Technical Details

### Disruption Implementation

**Lane Closure** (Paya Lebar):
```python
road.apply_lane_closure(
    capacity_reduction=0.5,  # Remove 50% of capacity
    start_time=0,
    duration=None  # Permanent
)
```

Reduces `num_cells` (road capacity) by 50%, forcing more vehicles into queue.

**Road Closure** (PIE):
```python
road.apply_road_closure(
    start_time=0,
    duration=None  # Permanent
)
```

Sets `num_cells=0` and `is_blocked=True`, preventing any vehicle entry. Forces rerouting.

### Rat-Running Detection

Compares residential road throughput between baseline and disruption:

```python
def analyze_rat_running(model):
    residential_throughput = sum(r.throughput for r in residential_roads)
    primary_throughput = sum(r.throughput for r in primary_roads)
    residential_fraction = residential_throughput / (residential_throughput + primary_throughput)
    return residential_fraction
```

Increase in `residential_fraction` indicates rat-running behavior.

### Queue Spillback

Roads track entry queue length:

```python
class UrbanRoadAgent:
    def __init__(self, ...):
        self.entry_queue = deque()  # Vehicles waiting to enter
        self.queue_length = 0

    def step(self):
        self.queue_length = len(self.entry_queue)
```

Queue builds when road capacity is exceeded (lane closure) or blocked (road closure).

## Validation

### Paya Lebar

**Reference**: Othman et al. (2023) used SUMO to model Paya Lebar lane closure
- Their findings: Queue spillback, increased trip times, congestion propagation
- **Our validation**: Compare qualitative patterns (queue formation, congestion spread)
- **Expected**: Similar congestion patterns, queue lengths proportional to capacity reduction

### PIE

**Speculative experiment** (no direct reference)
- Goal: Test ABM's ability to capture emergent behavior
- **Rat-running**: Known phenomenon in traffic engineering
- **Expected**: Residential roads near PIE closure show increased usage

## Troubleshooting

### Problem: No disruption applied

**Solution**: Check that affected edges exist in network. Run with `--quick` first to verify network loading.

### Problem: No vehicles spawned

**Cause**: Origin-destination pairs have no valid routes.

**Solution**:
- Ensure network bbox is large enough
- Check that major nodes are identified (`model.major_nodes`)
- Try larger network area

### Problem: Simulation runs very slowly

**Solutions**:
- Use `--quick` flag
- Reduce vehicle counts
- Reduce simulation duration (edit `max_steps` in config)

### Problem: No rat-running detected

**Possible causes**:
- Network too small (no residential alternatives)
- Disruption not forcing rerouting (try longer closure)
- Vehicles not reaching disrupted area

**Solution**: Expand network bounds or increase closure length.

## References

### Localized Incidents

Othman, N. B., et al. (2023). Agent-based traffic simulation for Paya Lebar region. (Citation needed - user to provide)

### Rat-Running

General traffic engineering concept:
- Drivers seek shortcuts through residential neighborhoods
- Caused by congestion on main routes
- Common during construction/closures

### ABM for Urban Planning

(Citations needed - user to provide)

## Future Enhancements

1. **Dynamic Rerouting**: Agents update routes when encountering congestion
2. **Time-Varying Demand**: Morning/evening peak patterns
3. **Multiple Disruptions**: Compound effects
4. **Traffic Signals**: Intersection delays
5. **Validation Data**: Compare against real Paya Lebar traffic data

## License

Part of PC5253 Traffic Simulation project.
