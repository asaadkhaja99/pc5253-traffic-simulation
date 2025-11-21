# All Tunable Parameters in Evacuation Simulation

Complete reference of all parameters you can adjust to tune the simulation behavior.

---

## 1. GEOGRAPHIC PARAMETERS

### Study Area Bounds
**Location:** `EvacuationConfig` in [evacuation_base.py:34-37](evacuation_base.py#L34-L37)

```python
bbox_north: float = 1.295      # Northern boundary (degrees latitude)
bbox_south: float = 1.260      # Southern boundary (degrees latitude)
bbox_east: float = 103.880     # Eastern boundary (degrees longitude)
bbox_west: float = 103.840     # Western boundary (degrees longitude)
```

**What it controls:** Size and location of road network downloaded from OpenStreetMap

**Current setting:** Marina Bay, Singapore (~3.9 km × ~4.5 km area)

**How to tune:**
- Larger area = more roads, more computation, more escape routes
- Smaller area = faster simulation, fewer routes, more congestion

---

### Evacuation Zone

**Location:** `EvacuationConfig` in [evacuation_base.py:40-41](evacuation_base.py#L40-L41)

```python
evacuation_center: Tuple[float, float] = (1.2775, 103.860)  # (lat, lon)
safe_zone_radius_km: float = 2.0
```

**What it controls:**
- `evacuation_center`: Geographic point of danger (epicenter)
- `safe_zone_radius_km`: Distance from center to safety threshold

**How nodes are categorized:**
- **Origin nodes:** distance < 2.0 km (inside danger zone)
- **Safe nodes:** distance ≥ 2.0 km (destinations)

**How to tune:**
- **Increase radius** (e.g., 3.0 km) → Fewer safe zone nodes, longer evacuation distances
- **Decrease radius** (e.g., 1.0 km) → More safe zones closer, faster evacuation

---

## 2. SIMULATION PARAMETERS

**Location:** `EvacuationConfig` in [evacuation_base.py:43-54](evacuation_base.py#L43-L54)

```python
# Agent population
num_agents: int = 2000

# Temporal resolution
delta_t: float = 1.0           # Time step (seconds)
max_steps: int = 7200          # Max simulation time (7200s = 2 hours)

# NaSch cellular automaton
cell_length_m: float = 7.5     # Length of each CA cell (meters)
default_v_max: int = 5         # Default max velocity (cells/step)
default_p_slow: float = 0.3    # Default randomization probability

# Reproducibility
seed: int = 42                 # Random seed
```

**What each controls:**

### `num_agents`
- **Current:** 2000 evacuees
- **Impact:** More agents → more congestion, longer evacuation time
- **Realistic range:** 500-5000 for Marina Bay

### `delta_t`
- **Current:** 1.0 second per timestep
- **Impact:** Temporal resolution (usually keep at 1.0)

### `max_steps`
- **Current:** 7200 steps = 2 hours
- **Impact:** When simulation stops
- **Tune if:** Evacuation not completing, or want longer observation

### `cell_length_m`
- **Current:** 7.5 meters (standard for NaSch)
- **Impact:** Spatial resolution of traffic model
- **Formula:** `num_cells = road_length / cell_length_m`
- **Smaller cells** → finer resolution, more computation

### `default_v_max`
- **Current:** 5 cells/step
- **At cell_length=7.5m, delta_t=1s:** 5 × 7.5m = 37.5 m/s = 135 km/h
- **Impact:** Maximum possible speed
- **Note:** Overridden by highway-specific values (see below)

### `default_p_slow`
- **Current:** 0.3 (30% chance to slow down randomly)
- **Impact:** Driver randomness/caution
- **Higher** → more conservative driving, more slowdowns
- **Lower** → more aggressive driving, smoother flow

---

## 3. HIGHWAY-SPECIFIC PARAMETERS

**Location:** `set_nasch_params()` in [evacuation_base.py:253-260](evacuation_base.py#L253-L260)

```python
highway_params = {
    'motorway':    {'v_max': 7, 'p_slow': 0.2,  'lanes': 4},
    'trunk':       {'v_max': 6, 'p_slow': 0.25, 'lanes': 3},
    'primary':     {'v_max': 5, 'p_slow': 0.30, 'lanes': 2},
    'secondary':   {'v_max': 4, 'p_slow': 0.35, 'lanes': 2},
    'tertiary':    {'v_max': 3, 'p_slow': 0.40, 'lanes': 1},
    'residential': {'v_max': 2, 'p_slow': 0.45, 'lanes': 1}
}
```

**What each parameter means:**

### `v_max` (cells/step)
Maximum velocity for that road type

| Road Type   | v_max | Speed @ 7.5m cells | km/h |
|-------------|-------|-------------------|------|
| Motorway    | 7     | 52.5 m/s          | 189  |
| Trunk       | 6     | 45.0 m/s          | 162  |
| Primary     | 5     | 37.5 m/s          | 135  |
| Secondary   | 4     | 30.0 m/s          | 108  |
| Tertiary    | 3     | 22.5 m/s          | 81   |
| Residential | 2     | 15.0 m/s          | 54   |

### `p_slow`
Randomization probability (driver caution)
- **Lower** for highways (0.2) = aggressive, smooth flow
- **Higher** for residential (0.45) = cautious, more stops

### `lanes`
Number of lanes (affects capacity)
- **Formula:** `capacity ∝ num_cells × num_lanes`
- More lanes → more vehicles can fit

**How to tune:**
- **Increase v_max** → Faster speeds, quicker evacuation
- **Decrease p_slow** → Less randomness, smoother flow
- **Increase lanes** → More capacity, less congestion

---

## 4. CONTRAFLOW PARAMETERS

**Location:** `enable_contraflow()` in [evacuation_base.py:278](evacuation_base.py#L278)

```python
capacity_multiplier = 1.5      # 50% capacity increase
```

**Location:** `spawn_contraflow()` in [run_folium_visualization.py:120-123](run_folium_visualization.py#L120-L123)

```python
# Which roads get contraflow
if road_agent.highway_type in ['motorway', 'trunk', 'primary']:
    road_agent.contraflow_enabled = True
    road_agent.max_vehicles = int(road_agent.max_vehicles * 1.5)
```

**What it controls:**
- Which road types get contraflow lanes
- How much capacity increases

**How to tune:**

### `capacity_multiplier`
- **Current:** 1.5 (50% increase)
- **Options:**
  - 1.3 = 30% increase (partial contraflow)
  - 2.0 = 100% increase (full reversal)

### Road types for contraflow
- **Current:** `['motorway', 'trunk', 'primary']`
- **More aggressive:** Add `'secondary'`
- **Less aggressive:** Only `['motorway']`

---

## 5. STAGED EVACUATION PARAMETERS

**Location:** `spawn_staged()` in [run_folium_visualization.py:82-89](run_folium_visualization.py#L82-L89)

```python
# Number of waves
num_per_wave = num_agents // 3   # Divide into 3 equal waves

# Wave departure times (seconds)
wave_times = [0, 600, 1200]      # 0, 10, 20 minutes

# Zone division (by distance from center)
waves = [
    node_distances[:len(node_distances)//3],              # Close (0-33%)
    node_distances[len(node_distances)//3:2*len(node_distances)//3],  # Medium (33-66%)
    node_distances[2*len(node_distances)//3:]             # Far (66-100%)
]
```

**What it controls:**
- How many waves
- When each wave departs
- Which zones evacuate first

**How to tune:**

### Number of waves
```python
# 2 waves
num_per_wave = num_agents // 2
wave_times = [0, 900]  # 0, 15 minutes

# 4 waves
num_per_wave = num_agents // 4
wave_times = [0, 300, 600, 900]  # 0, 5, 10, 15 minutes
```

### Wave timing
```python
# Faster staggering (5-minute intervals)
wave_times = [0, 300, 600]

# Slower staggering (20-minute intervals)
wave_times = [0, 1200, 2400]
```

### Zone priority
**Current:** Closest zones evacuate first (makes sense - most danger)

**Reverse:** Farthest first (pre-position safe zones)
```python
waves = [
    node_distances[2*len(node_distances)//3:],            # Far first
    node_distances[len(node_distances)//3:2*len(node_distances)//3],
    node_distances[:len(node_distances)//3]               # Close last
]
```

---

## 6. CONGESTION THRESHOLD

**Location:** Used throughout for metrics

```python
# A road is "congested" if:
speed_ratio < 0.3   # Current speed < 30% of max speed
```

**What it controls:** What counts as "congested" in metrics

**How to tune:**
```python
# More strict (only severe congestion counts)
speed_ratio < 0.2   # < 20%

# Less strict (even light congestion counts)
speed_ratio < 0.5   # < 50%
```

---

## 7. DERIVED PARAMETERS (Calculated, not directly set)

### Max vehicles per road
```python
# Calculated from:
max_vehicles = num_cells × num_lanes × density_factor
```
Where:
- `num_cells = road_length_m / cell_length_m`
- `num_lanes` from highway_params
- `density_factor` typically ~0.7 (can't pack perfectly)

### Road capacity
```python
capacity = num_cells  # For single lane
capacity = num_cells × num_lanes  # Multi-lane
```

---

## PARAMETER SENSITIVITY GUIDE

### Most Impactful Parameters (Tune These First)

1. **`num_agents`** - Directly controls congestion severity
2. **`safe_zone_radius_km`** - Controls how far people must travel
3. **`wave_times`** - Controls staging effectiveness
4. **`capacity_multiplier`** - Controls contraflow effectiveness
5. **Highway `v_max`** - Controls maximum speeds

### Calibration Parameters (Fine-tuning)

6. **`p_slow`** - Driver behavior randomness
7. **`cell_length_m`** - Spatial resolution
8. **`bbox` bounds** - Network size
9. **Highway `lanes`** - Road capacity

### Usually Keep Fixed

10. **`delta_t`** - Keep at 1.0 second
11. **`seed`** - Change only for different random runs
12. **`max_steps`** - Only if evacuation not finishing

---

## QUICK TUNING RECIPES

### To Reduce Congestion
- ✅ Decrease `num_agents` (e.g., 1000 instead of 2000)
- ✅ Increase `safe_zone_radius_km` (e.g., 1.5 km - closer safe zones)
- ✅ Increase highway `lanes` (e.g., motorway: 6 lanes)
- ✅ Decrease `p_slow` (e.g., 0.1 - smoother flow)

### To Make Staged More Effective
- ✅ Increase number of waves (4-5 instead of 3)
- ✅ Increase time between waves (15-20 min instead of 10)
- ✅ Use smaller `num_per_wave` (staggers more)

### To Make Contraflow More Effective
- ✅ Increase `capacity_multiplier` (e.g., 2.0 = double capacity)
- ✅ Apply to more road types (add 'secondary')
- ✅ Target specific bottleneck roads manually

### To Speed Up Evacuation
- ✅ Increase all highway `v_max` by 1-2
- ✅ Decrease all `p_slow` by 0.05-0.1
- ✅ Increase highway `lanes`

### For More Realistic Simulation
- ✅ Use speed limit data to set `v_max` realistically
- ✅ Calibrate `p_slow` from real traffic data
- ✅ Match `num_agents` to actual population density

---

## WHERE TO CHANGE PARAMETERS

### 1. In Code (Permanent Changes)

**[evacuation_base.py](evacuation_base.py):**
- Edit `EvacuationConfig` dataclass (lines 30-54)
- Edit `highway_params` dict (lines 253-260)
- Edit `capacity_multiplier` (line 278)

**[run_folium_visualization.py](run_folium_visualization.py):**
- Edit `wave_times` (line 89)
- Edit wave division logic (lines 82-87)
- Edit contraflow road types (line 120)

### 2. At Runtime (Command-line)

```bash
# Change number of agents
python evacuation_model/run_folium_visualization.py --num-agents 1000

# Change simulation duration
python evacuation_model/run_folium_visualization.py --max-steps 3600

# Change capture interval
python evacuation_model/run_folium_visualization.py --interval 30
```

### 3. By Creating Config Objects

```python
from evacuation_base import EvacuationConfig, EvacuationModel

# Custom configuration
config = EvacuationConfig(
    num_agents=1500,
    safe_zone_radius_km=1.5,
    cell_length_m=5.0,
    default_p_slow=0.2
)

model = EvacuationModel(config)
```

---

## EXAMPLE: Creating a Sensitivity Analysis

```python
# Test different population sizes
for num_agents in [500, 1000, 1500, 2000, 2500]:
    config = EvacuationConfig(num_agents=num_agents)
    model = EvacuationModel(config)
    # Run simulation...

# Test different safe zone radii
for radius in [1.0, 1.5, 2.0, 2.5, 3.0]:
    config = EvacuationConfig(safe_zone_radius_km=radius)
    model = EvacuationModel(config)
    # Run simulation...

# Test different contraflow capacities
for multiplier in [1.2, 1.5, 1.8, 2.0]:
    # Need to edit capacity_multiplier in code
    # or create parameterized version
```

---

## SUMMARY TABLE

| Parameter | Location | Current Value | Impact | Tune Range |
|-----------|----------|---------------|--------|------------|
| `num_agents` | Config | 2000 | Population size | 500-5000 |
| `safe_zone_radius_km` | Config | 2.0 km | Evacuation distance | 1.0-3.0 km |
| `cell_length_m` | Config | 7.5 m | Spatial resolution | 5.0-10.0 m |
| `default_v_max` | Config | 5 cells/s | Max speed | 3-7 |
| `default_p_slow` | Config | 0.3 | Randomness | 0.1-0.5 |
| `max_steps` | Config | 7200 s | Sim duration | 3600-14400 s |
| Highway `v_max` | Road params | 2-7 | Road speeds | ±1-2 |
| Highway `p_slow` | Road params | 0.2-0.45 | Road behavior | ±0.05-0.1 |
| Highway `lanes` | Road params | 1-4 | Road capacity | ±1-2 |
| `capacity_multiplier` | Contraflow | 1.5 | Contraflow boost | 1.2-2.0 |
| `wave_times` | Staged | [0,600,1200] | Wave timing | 300-2400 s |
| Number of waves | Staged | 3 | Wave count | 2-5 |

