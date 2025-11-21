"""
NaSch Traffic Simulation - Static Hysteresis Visualization

Generates a single static plot showing the complete hysteresis cycle:
- Reference FD curve from density sweep
- Complete trajectory showing density increase and decrease
- Critical point marked
- Color-coded trajectory showing direction (increase vs decrease)
- Arrows showing flow direction

This is ideal for presentations and papers where you need a single
publication-quality figure instead of an animated GIF.

USAGE:
    python nasch_fd_hysteresis_static.py

OUTPUT:
    nasch_hysteresis_cycle.png - High-resolution static figure

Note: AI Tools:
Claude code was used to assist with the implementation of this script. 
All final content was reviewed and edited to ensure accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import osmnx as ox
import geopandas as gpd
import mesa
import mesa_geo as mg

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mesa'))
from critical_point_analysis import run_nasch_sweep, analyze_fd

ox.settings.use_cache = True


# =============================================================================
# NaSch Model Functions
# =============================================================================

def nasch_step(road, v_max=5, p_slow=0.3, rng=None):
    """Single step of Nagel-Schreckenberg model."""
    if rng is None:
        rng = np.random.default_rng()

    num_cells = len(road)
    new_road = -np.ones(num_cells, dtype=np.int16)
    car_positions = np.where(road >= 0)[0]

    if len(car_positions) == 0:
        return new_road

    next_positions = np.roll(car_positions, -1)
    gaps = (next_positions - car_positions - 1) % num_cells

    update_order = rng.permutation(len(car_positions))
    for idx in update_order:
        pos = car_positions[idx]
        v = road[pos]

        v = min(v + 1, v_max)
        v = min(v, gaps[idx])
        if v > 0 and rng.random() < p_slow:
            v -= 1

        new_pos = (pos + v) % num_cells
        new_road[new_pos] = v

    return new_road


# =============================================================================
# Agent Classes
# =============================================================================

class VehicleAgent(mg.GeoAgent):
    """Vehicle agent for visualization."""

    def __init__(self, model, geometry, crs, road_segment, position, velocity, vehicle_index):
        super().__init__(model, geometry, crs)
        self.road_segment = road_segment
        self.position = position
        self.velocity = velocity
        self.vehicle_index = vehicle_index

    def update_position_on_road(self):
        road_geom = self.road_segment.geometry
        point_on_road = road_geom.interpolate(self.position, normalized=True)
        self.geometry = point_on_road

    def step(self):
        self.update_position_on_road()


class RoadAgent(mg.GeoAgent):
    """Road segment agent managing NaSch CA."""

    def __init__(self, model, geometry, crs, edge_id, u, v, key,
                 length_m, speed_kph, highway_type, num_vehicles):
        super().__init__(model, geometry, crs)
        self.edge_id = edge_id
        self.u = u
        self.v = v
        self.key = key
        self.length_m = length_m
        self.speed_kph = speed_kph
        self.highway_type = highway_type
        self.num_vehicles = num_vehicles

        self.set_nasch_params()

        self.road = None
        self.num_cells = None
        self.cell_length_m = 7.5
        self.vehicles = []

        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0

    def set_nasch_params(self):
        """Set NaSch parameters based on highway type."""
        highway_params = {
            'motorway': {'v_max': 7, 'p_slow': 0.2},
            'trunk': {'v_max': 6, 'p_slow': 0.25},
            'primary': {'v_max': 5, 'p_slow': 0.30},
            'secondary': {'v_max': 4, 'p_slow': 0.35},
            'tertiary': {'v_max': 3, 'p_slow': 0.40},
            'residential': {'v_max': 2, 'p_slow': 0.45}
        }

        highway_type = self.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'residential'
        elif not isinstance(highway_type, str):
            highway_type = str(highway_type) if highway_type else 'residential'

        params = highway_params.get(highway_type, highway_params['residential'])
        self.v_max = params['v_max']
        self.p_slow = params['p_slow']

    def set_num_vehicles(self, num_vehicles):
        """Dynamically change number of vehicles on this road."""
        self.num_vehicles = num_vehicles

        self.num_cells = max(5, int(self.length_m / self.cell_length_m))
        self.road = -np.ones(self.num_cells, dtype=np.int16)

        num_cars = min(num_vehicles, self.num_cells)

        if num_cars > 0:
            car_positions = self.model.rng.choice(self.num_cells, size=num_cars, replace=False)
            self.road[car_positions] = self.model.rng.integers(0, self.v_max + 1, size=num_cars)

        self._update_speed_metrics()

    def initialize_vehicles(self):
        """Initialize NaSch cellular automaton."""
        self.num_cells = max(5, int(self.length_m / self.cell_length_m))
        self.road = -np.ones(self.num_cells, dtype=np.int16)

        num_cars = min(self.num_vehicles, self.num_cells)

        if num_cars > 0:
            car_positions = self.model.rng.choice(self.num_cells, size=num_cars, replace=False)
            self.road[car_positions] = self.model.rng.integers(0, self.v_max + 1, size=num_cars)

        car_positions = np.where(self.road >= 0)[0]
        for i, cell_pos in enumerate(car_positions):
            position = cell_pos / self.num_cells
            velocity = self.road[cell_pos] / self.v_max if self.v_max > 0 else 0

            point_geom = self.geometry.interpolate(position, normalized=True)
            vehicle = VehicleAgent(
                model=self.model, geometry=point_geom, crs=self.crs,
                road_segment=self, position=position, velocity=velocity,
                vehicle_index=i
            )
            self.vehicles.append(vehicle)
            self.model.space.add_agents(vehicle)

    def _update_speed_metrics(self):
        """Update speed metrics from current road state."""
        moving_vehicles = self.road[self.road >= 0]

        if len(moving_vehicles) > 0:
            mean_v_cells = moving_vehicles.mean()
            mean_speed_kph = mean_v_cells * (self.cell_length_m / 1000) * 3600 / self.model.delta_t

            speed_ratio = mean_speed_kph / self.speed_kph if self.speed_kph > 0 else 1.0
            speed_ratio = np.clip(speed_ratio, 0.0, 1.2)

            self.mean_speed_kph = mean_speed_kph
            self.speed_ratio = speed_ratio
        else:
            self.mean_speed_kph = 0.0
            self.speed_ratio = 0.0

    def step(self):
        """Update NaSch CA and vehicles."""
        self.road = nasch_step(self.road, self.v_max, self.p_slow, self.model.rng)
        self._update_speed_metrics()

        car_positions = np.where(self.road >= 0)[0]
        for i, vehicle in enumerate(self.vehicles):
            if i < len(car_positions):
                cell_pos = car_positions[i]
                vehicle.position = cell_pos / self.num_cells
                vehicle.velocity = self.road[cell_pos] / self.v_max if self.v_max > 0 else 0


class NaSchTrafficModel(mesa.Model):
    """NaSch traffic model with dynamic density control."""

    def __init__(self, edges_gdf, num_vehicles_per_road=5, delta_t=1.0, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        self.delta_t = delta_t

        self.edges_gdf = edges_gdf.to_crs(4326)
        self.road_agents = []

        self._initialize_roads(num_vehicles_per_road)

        self.step_count = 0
        self.fd_history = {'density': [], 'flow': [], 'speed': [], 'phase': []}

    def _initialize_roads(self, num_vehicles_per_road):
        for idx, row in self.edges_gdf.iterrows():
            highway = row.get('highway', 'residential')
            if isinstance(highway, list):
                highway = highway[0] if highway else 'residential'
            elif not isinstance(highway, str):
                highway = str(highway) if highway else 'residential'

            road = RoadAgent(
                model=self, geometry=row.geometry, crs=self.edges_gdf.crs,
                edge_id=idx, u=row.get('u', idx), v=row.get('v', idx),
                key=row.get('key', 0), length_m=row.get('length', 100),
                speed_kph=row.get('speed_kph', 50), highway_type=highway,
                num_vehicles=num_vehicles_per_road
            )
            self.road_agents.append(road)
            self.space.add_agents(road)
            road.initialize_vehicles()

    def set_density_multiplier(self, multiplier):
        for road in self.road_agents:
            base_vehicles = 5
            new_vehicles = max(1, int(base_vehicles * multiplier))
            road.set_num_vehicles(new_vehicles)

    def step(self, phase='increasing'):
        for road in self.road_agents:
            road.step()

        if self.step_count % 5 == 0:
            total_vehicles = sum((r.road >= 0).sum() for r in self.road_agents)
            total_cells = sum(r.num_cells for r in self.road_agents)
            avg_speed = np.mean([r.mean_speed_kph for r in self.road_agents if r.mean_speed_kph > 0])

            density = total_vehicles / total_cells if total_cells > 0 else 0
            flow = density * (avg_speed / 3.6) if not np.isnan(avg_speed) else 0

            self.fd_history['density'].append(density)
            self.fd_history['flow'].append(flow)
            self.fd_history['speed'].append(avg_speed if not np.isnan(avg_speed) else 0)
            self.fd_history['phase'].append(phase)

        self.step_count += 1


# =============================================================================
# Static Visualization Function
# =============================================================================

def create_hysteresis_plot(model, fd_reference, analysis_result, output_filename='nasch_hysteresis_cycle.png'):
    """Create a static publication-quality plot showing the complete hysteresis cycle."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), facecolor='white')

    densities = np.array(model.fd_history['density'])
    flows = np.array(model.fd_history['flow'])
    speeds = np.array(model.fd_history['speed'])
    phases = model.fd_history['phase']

    increasing_mask = np.array([p == 'increasing' for p in phases])
    decreasing_mask = ~increasing_mask

    # PLOT 1: Flow vs Density
    ax1.plot(fd_reference.density, fd_reference.flow, 'o-',
             color='lightgray', linewidth=3, markersize=8, alpha=0.7,
             label='Reference FD', zorder=1)

    if analysis_result and analysis_result['breakpoint']:
        bp = analysis_result['breakpoint']
        ax1.axvline(bp.x_star, color='red', linestyle='--', linewidth=3,
                   alpha=0.7, label=f'Critical ρ*', zorder=2)
        ax1.axvspan(bp.x_star * 0.9, bp.x_star * 1.1,
                   color='red', alpha=0.1, zorder=0)

    if increasing_mask.any():
        ax1.plot(densities[increasing_mask], flows[increasing_mask],
                'o-', color='#1f77b4', linewidth=4, markersize=7,
                label='Increasing ↑', zorder=4, alpha=0.8)

        inc_dens = densities[increasing_mask]
        inc_flow = flows[increasing_mask]
        if len(inc_dens) > 10:
            for i in [len(inc_dens)//4, len(inc_dens)//2, 3*len(inc_dens)//4]:
                if i < len(inc_dens) - 1:
                    ax1.annotate('', xy=(inc_dens[i+1], inc_flow[i+1]),
                               xytext=(inc_dens[i], inc_flow[i]),
                               arrowprops=dict(arrowstyle='->', color='#1f77b4',
                                             lw=3, alpha=0.7))

    if decreasing_mask.any():
        ax1.plot(densities[decreasing_mask], flows[decreasing_mask],
                's--', color='#ff7f0e', linewidth=4, markersize=7,
                label='Decreasing ↓', zorder=4, alpha=0.8)

        dec_dens = densities[decreasing_mask]
        dec_flow = flows[decreasing_mask]
        if len(dec_dens) > 10:
            for i in [len(dec_dens)//4, len(dec_dens)//2, 3*len(dec_dens)//4]:
                if i < len(dec_dens) - 1:
                    ax1.annotate('', xy=(dec_dens[i+1], dec_flow[i+1]),
                               xytext=(dec_dens[i], dec_flow[i]),
                               arrowprops=dict(arrowstyle='->', color='#ff7f0e',
                                             lw=3, alpha=0.7, linestyle='dashed'))

    if len(densities) > 0:
        ax1.plot(densities[0], flows[0], 'go', markersize=18,
                label='Start', zorder=10, markeredgecolor='darkgreen', markeredgewidth=3)
        ax1.plot(densities[-1], flows[-1], 'r*', markersize=24,
                label='End', zorder=10, markeredgecolor='darkred', markeredgewidth=3)

    ax1.set_xlabel('Density ρ (veh/cell)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Flow q (veh/s)', fontsize=20, fontweight='bold')
    ax1.set_title('Flow vs Density', fontsize=22, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=18, framealpha=0.95, ncol=2)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # PLOT 2: Speed vs Density
    ax2.plot(fd_reference.density, fd_reference.speed, 's-',
             color='lightgray', linewidth=3, markersize=8, alpha=0.7,
             label='Reference', zorder=1)

    if analysis_result and analysis_result['breakpoint']:
        bp = analysis_result['breakpoint']
        ax2.axvline(bp.x_star, color='red', linestyle='--', linewidth=3, alpha=0.7, zorder=2)
        ax2.axvspan(bp.x_star * 0.9, bp.x_star * 1.1, color='red', alpha=0.1, zorder=0)

    if increasing_mask.any():
        ax2.plot(densities[increasing_mask], speeds[increasing_mask] / 3.6,
                'o-', color='#1f77b4', linewidth=4, markersize=7,
                label='Increasing ↑', zorder=4, alpha=0.8)

    if decreasing_mask.any():
        ax2.plot(densities[decreasing_mask], speeds[decreasing_mask] / 3.6,
                's--', color='#ff7f0e', linewidth=4, markersize=7,
                label='Decreasing ↓', zorder=4, alpha=0.8)

    ax2.set_xlabel('Density ρ (veh/cell)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Speed v (m/s)', fontsize=20, fontweight='bold')
    ax2.set_title('Speed vs Density', fontsize=22, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=18, framealpha=0.95)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Overall title
    fig.suptitle('NaSch Model',
                fontsize=24, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_filename}")
    plt.close()


# =============================================================================
# Main Function
# =============================================================================

def generate_hysteresis_cycle(
    place="Singapore",
    network_type='drive',
    num_steps=300,
    delta_t=1.0,
    output_filename='nasch_hysteresis_cycle.png',
    fd_densities=None,
    fd_L=600,
    fd_v_max=5,
    fd_p_slow=0.3,
    fd_steps=2000,
    fd_warm=500,
    fd_n_seeds=5
):
    """Generate static hysteresis plot showing complete cycle."""
    print("=" * 80)
    print("NASCH HYSTERESIS CYCLE - STATIC VISUALIZATION")
    print("=" * 80)

    print("\n[1/4] Computing reference fundamental diagram...")
    if fd_densities is None:
        fd_densities = np.linspace(0.02, 0.8, 20)

    fd_reference = run_nasch_sweep(
        fd_densities, L=fd_L, v_max=fd_v_max, p_slow=fd_p_slow,
        steps=fd_steps, warm=fd_warm, n_seeds=fd_n_seeds
    )
    analysis_result = analyze_fd(fd_reference)

    if analysis_result['breakpoint']:
        print(f"  ✓ Critical density: ρ* = {analysis_result['breakpoint'].x_star:.4f} veh/cell")

    print(f"\n[2/4] Loading road network for {place}...")
    G = ox.graph_from_place(place, network_type=network_type)
    G = ox.add_edge_speeds(G)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    if not {'u', 'v', 'key'}.issubset(edges_gdf.columns):
        edges_gdf = edges_gdf.reset_index()

    print(f"  ✓ Loaded {len(edges_gdf)} road segments")

    print("\n[3/4] Running simulation with cyclical density...")
    model = NaSchTrafficModel(edges_gdf, num_vehicles_per_road=5, delta_t=delta_t)

    for i in range(num_steps):
        if i < 150:
            progress = i / 150.0
            multiplier = 0.5 + progress * 1.3
            phase = 'increasing'
        else:
            progress = (i - 150) / 150.0
            multiplier = 1.8 - progress * 1.3
            phase = 'decreasing'

        if i % 10 == 0:
            model.set_density_multiplier(multiplier)

        if i % 50 == 0:
            print(f"  Step {i}/{num_steps} [{phase}] (density {multiplier:.2f}x)")

        model.step(phase=phase)

    print(f"  ✓ Simulation complete: {len(model.fd_history['density'])} data points")

    print("\n[4/4] Creating static visualization...")
    create_hysteresis_plot(model, fd_reference, analysis_result, output_filename)

    print(f"\n✓ Output: {output_filename}")
    print(f"  - Resolution: 300 DPI")
    print(f"  - Size: {os.path.getsize(output_filename) / 1024:.1f} KB")
    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    CONFIG = {
        'place': 'Singapore',
        'network_type': 'drive',
        'num_steps': 300,
        'delta_t': 1.0,
        'output_filename': 'nasch_hysteresis_cycle.png',
        'fd_densities': np.linspace(0.02, 0.8, 20),
        'fd_L': 600,
        'fd_v_max': 5,
        'fd_p_slow': 0.3,
        'fd_steps': 2000,
        'fd_warm': 500,
        'fd_n_seeds': 5,
    }

    generate_hysteresis_cycle(**CONFIG)
