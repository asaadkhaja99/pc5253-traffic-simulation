# Evacuation Model - Quick Start Guide

## Installation Verification

All dependencies should already be installed via `pyproject.toml`. Verify:

```bash
cd /Users/asaad/Documents/School-[NUS_MPT]/Final_Project/pc5253-traffic-simulation
uv sync
```

## Running the Complete Study

### Option 1: Full Study (Recommended for final results)

Run all scenarios with 2000 agents (~30-60 min):

```bash
cd evacuation_model
uv run python run_evacuation_study.py
```

### Option 2: Quick Test (Recommended for testing)

Run with 500 agents (~10-15 min):

```bash
cd evacuation_model
uv run python run_evacuation_study.py --quick
```

### Option 3: Custom Configuration

```bash
# 1000 agents with 5 waves
uv run python run_evacuation_study.py --num-agents 1000 --num-waves 5

# Change wave interval to 5 minutes
uv run python run_evacuation_study.py --wave-interval 300

# Change random seed
uv run python run_evacuation_study.py --seed 123

# Combine options
uv run python run_evacuation_study.py --num-agents 1500 --num-waves 3 --seed 99
```

## Running Individual Scenarios

If you want to run scenarios separately:

```bash
cd evacuation_model

# Simultaneous evacuation (baseline)
uv run python scenario_simultaneous.py

# Staged evacuation
uv run python scenario_staged.py

# Contraflow intervention
uv run python contraflow_intervention.py

# Analysis (after running scenarios)
uv run python analyze_evacuation.py

# Visualization (after running scenarios)
uv run python visualize_evacuation.py
```

## Expected Output

Results will be saved to: `<project_root>/output/evacuation/`

(Where `<project_root>` is `/Users/asaad/Documents/School-[NUS_MPT]/Final_Project/pc5253-traffic-simulation/`)

### Data Files (CSV)

- `simultaneous_evacuation.csv` - Baseline scenario time series
- `staged_evacuation.csv` - Staged scenario time series
- `contraflow_evacuation.csv` - Contraflow scenario time series
- `contraflow_evacuation_contraflow_roads.csv` - Contraflow road details
- `scenario_comparison.csv` - Summary metrics comparison
- `intervention_effectiveness.csv` - Intervention improvements

### Visualizations (PNG)

- `evacuation_progress.png` - Evacuation progress over time
- `congestion_comparison.png` - Congestion levels comparison
- `speed_comparison.png` - Network speed comparison
- `intervention_effectiveness.png` - Effectiveness metrics
- `evacuation_dashboard.png` - Combined dashboard

## Typical Runtime

| Configuration | Agents | Expected Time |
|---------------|--------|---------------|
| Quick test    | 500    | 10-15 min     |
| Default       | 2000   | 30-60 min     |
| Large         | 5000   | 60-120 min    |

Runtime depends on:
- Number of agents
- Network size (current: Marina Bay area)
- Number of simulation steps
- Machine specifications

## Interpreting Results

### Key Metrics

**T95** - Time to evacuate 95% of agents
- Lower is better
- Industry standard metric

**Mean Evacuation Time** - Average time per agent
- Lower is better
- Indicates individual experience

**Peak Congestion** - Maximum number of congested roads
- Lower is better
- Congested = speed ratio < 0.3

**Mean Network Speed** - Average speed across network
- Higher is better
- Indicates overall flow quality

### Expected Findings

Based on typical urban evacuation scenarios:

1. **Simultaneous (Baseline)**:
   - Highest peak congestion
   - Moderate T95
   - Lowest network speed
   - Identifies worst-case scenario

2. **Staged Evacuation**:
   - Lowest peak congestion (~40-50% reduction)
   - Potentially higher T95 (last wave delayed)
   - Higher network speed
   - Better for congestion management

3. **Contraflow Intervention**:
   - Lowest T95 (~10-20% improvement)
   - Moderate peak congestion
   - Higher network speed
   - Best for time minimization

## Troubleshooting

### Problem: Network Loading Takes Forever

**Solution**: The first run downloads OSM data and caches it. Subsequent runs are faster.

### Problem: Simulation Runs Very Slowly

**Solutions**:
- Use `--quick` flag (500 agents)
- Reduce agents: `--num-agents 1000`
- Check system resources (memory, CPU)

### Problem: ImportError or ModuleNotFoundError

**Solution**:
```bash
cd /path/to/pc5253-traffic-simulation
uv sync
```

### Problem: No Results Generated

**Check**:
1. Did simulation complete? Look for "STUDY COMPLETE" message
2. Check `<project_root>/output/evacuation/` directory exists
3. Look for error messages in console output

### Problem: Visualizations Look Strange

**Possible causes**:
- Insufficient data (simulation ended too early)
- No variance in metrics (all scenarios identical)
- Check CSV files have data

## Advanced Usage

### Changing Evacuation Zone

Edit [evacuation_base.py](evacuation_base.py:24-30):

```python
@dataclass
class EvacuationConfig:
    # Change bounding box
    bbox_north: float = YOUR_NORTH
    bbox_south: float = YOUR_SOUTH
    bbox_east: float = YOUR_EAST
    bbox_west: float = YOUR_WEST

    # Change evacuation center
    evacuation_center: Tuple[float, float] = (YOUR_LAT, YOUR_LON)
```

### Changing NaSch Parameters

Edit [evacuation_base.py:252-265] `EvacuationRoadAgent.set_nasch_params()`:

```python
highway_params = {
    'motorway': {'v_max': 7, 'p_slow': 0.2, 'lanes': 4},
    # ... modify as needed
}
```

### Changing Contraflow Selection

Edit [contraflow_intervention.py:88] `select_contraflow_roads()`:

```python
# Change max_roads parameter
selected = select_contraflow_roads(candidates, max_roads=6)  # Select 6 instead of 4
```

Or via CLI:
```bash
uv run python run_evacuation_study.py --num-contraflow 6
```

## Next Steps

After running the study:

1. **Review Results**: Check CSV files and visualizations
2. **Analyze Bottlenecks**: Look at `contraflow_evacuation_contraflow_roads.csv`
3. **Iterate**: Modify parameters and re-run
4. **Document Findings**: Add notes to results directory

## Support

For detailed documentation, see [README.md](README.md).

For implementation details, see code comments in:
- [evacuation_base.py](evacuation_base.py) - Core model
- [run_evacuation_study.py](run_evacuation_study.py) - Master script
