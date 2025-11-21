"""
IDM Traffic Simulation - Static Hysteresis Visualization

Generates a single static plot showing the complete hysteresis cycle:
- Reference FD curve from density sweep
- Complete trajectory showing density increase and decrease
- Critical point marked
- Color-coded trajectory showing direction (increase vs decrease)
- Arrows showing flow direction


USAGE:
    python idm_fd_hysteresis_static.py

OUTPUT:
    idm_hysteresis_cycle.png - High-resolution static figure

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
from shapely.geometry import Point

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'critical_points'))
from critical_point_analysis import run_idm_sweep, analyze_fd

ox.settings.use_cache = True


# =============================================================================
# IDM Dynamics Functions
# =============================================================================

def idm_acceleration(x, v, N, v0, s0, T, a_max, b, delta, L):
    """Calculate IDM acceleration for all vehicles."""
    x_lead = np.append(x[1:], x[0] + L)
    v_lead = np.append(v[1:], v[0])

    s = x_lead - x
    s = np.where(s < 0, s + L, s)

    dv = v - v_lead
    s_star = s0 + v * T + (v * dv) / (2 * np.sqrt(a_max * b))

    free_road_term = 1 - (v / v0)**delta
    interaction_term = (s_star / np.maximum(s, s0))**2

    acceleration = a_max * (free_road_term - interaction_term)
    return acceleration


# =============================================================================
# Agent Classes
# =============================================================================

class VehicleAgent(mg.GeoAgent):
    """Vehicle agent following IDM dynamics."""

    def __init__(self, model, geometry, crs, road_segment, position, velocity, vehicle_index):
        super().__init__(model, geometry, crs)
        self.road_segment = road_segment
        self.position = position
        self.velocity = velocity
        self.vehicle_index = vehicle_index
        self.acceleration = 0.0

    def update_position_on_road(self):
        road_geom = self.road_segment.geometry
        point_on_road = road_geom.interpolate(self.position, normalized=True)
        self.geometry = point_on_road

    def step(self):
        self.update_position_on_road()


class RoadAgent(mg.GeoAgent):
    """Road segment agent managing IDM vehicle flow."""

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

        self.set_idm_params()

        self.positions = None
        self.velocities = None
        self.vehicles = []
        self.velocity_history = []
        self.internal_step_count = 0
        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0

    def set_idm_params(self):
        """Set IDM parameters based on highway type."""
        highway_params = {
            'motorway': {'v0': 1.0, 's0': 0.05, 'T': 1.2, 'a_max': 2.0, 'b': 3.0, 'delta': 4},
            'trunk': {'v0': 0.9, 's0': 0.04, 'T': 1.3, 'a_max': 1.8, 'b': 2.8, 'delta': 4},
            'primary': {'v0': 0.85, 's0': 0.04, 'T': 1.4, 'a_max': 1.6, 'b': 2.5, 'delta': 4},
            'secondary': {'v0': 0.8, 's0': 0.03, 'T': 1.5, 'a_max': 1.4, 'b': 2.2, 'delta': 4},
            'tertiary': {'v0': 0.75, 's0': 0.03, 'T': 1.6, 'a_max': 1.2, 'b': 2.0, 'delta': 4},
            'residential': {'v0': 0.7, 's0': 0.025, 'T': 1.8, 'a_max': 1.0, 'b': 1.8, 'delta': 4}
        }

        highway_type = self.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'residential'
        elif not isinstance(highway_type, str):
            highway_type = str(highway_type) if highway_type else 'residential'

        params = highway_params.get(highway_type, highway_params['residential'])
        self.v0 = params['v0']
        self.s0 = params['s0']
        self.T = params['T']
        self.a_max = params['a_max']
        self.b = params['b']
        self.delta = params['delta']
        self.L = 1.0

    def set_num_vehicles(self, num_vehicles):
        """Dynamically change number of vehicles on this road."""
        old_N = self.num_vehicles
        self.num_vehicles = num_vehicles

        if num_vehicles > old_N:
            self._add_vehicles(num_vehicles - old_N)
        elif num_vehicles < old_N:
            self._remove_vehicles(old_N - num_vehicles)

    def _add_vehicles(self, count):
        for _ in range(count):
            new_pos = self.model.rng.random()
            new_vel = self.velocities.mean() if len(self.velocities) > 0 else self.v0 * 0.5

            self.positions = np.append(self.positions, new_pos)
            self.velocities = np.append(self.velocities, new_vel)

            order = np.argsort(self.positions)
            self.positions = self.positions[order]
            self.velocities = self.velocities[order]

            point_geom = self.geometry.interpolate(new_pos, normalized=True)
            vehicle = VehicleAgent(
                model=self.model, geometry=point_geom, crs=self.crs,
                road_segment=self, position=new_pos, velocity=new_vel,
                vehicle_index=len(self.vehicles)
            )
            self.vehicles.append(vehicle)
            self.model.space.add_agents(vehicle)

    def _remove_vehicles(self, count):
        for _ in range(min(count, len(self.vehicles))):
            if len(self.vehicles) > 0:
                vehicle = self.vehicles.pop()
                self.model.space.remove_agent(vehicle)
                self.positions = self.positions[:-1]
                self.velocities = self.velocities[:-1]

    def initialize_vehicles(self):
        """Initialize vehicle positions and velocities for IDM."""
        N = self.num_vehicles
        self.positions = np.linspace(0, self.L, N, endpoint=False)
        perturbation = self.model.rng.normal(0, 0.02, N)
        self.positions = (self.positions + perturbation) % self.L
        self.positions = np.sort(self.positions)

        self.velocities = np.ones(N) * self.v0 * 0.8
        perturbation_v = self.model.rng.normal(0, self.v0 * 0.1, N)
        self.velocities = np.clip(self.velocities + perturbation_v, 0, self.v0)

        for i in range(N):
            point_geom = self.geometry.interpolate(self.positions[i], normalized=True)
            vehicle = VehicleAgent(
                model=self.model, geometry=point_geom, crs=self.crs,
                road_segment=self, position=self.positions[i],
                velocity=self.velocities[i], vehicle_index=i
            )
            self.vehicles.append(vehicle)
            self.model.space.add_agents(vehicle)

    def update_idm_dynamics(self, delta_t):
        """Update vehicle dynamics using IDM."""
        if self.num_vehicles < 2:
            return

        a = idm_acceleration(self.positions, self.velocities, self.num_vehicles,
                           self.v0, self.s0, self.T, self.a_max, self.b,
                           self.delta, self.L)

        self.positions = self.positions + delta_t * self.velocities
        self.velocities = self.velocities + delta_t * a

        self.positions = self.positions % self.L
        self.velocities = np.clip(self.velocities, 0, self.v0 * 1.2)

        order = np.argsort(self.positions)
        self.positions = self.positions[order]
        self.velocities = self.velocities[order]

        for i, vehicle in enumerate(self.vehicles):
            if i < len(self.positions):
                vehicle.position = self.positions[i]
                vehicle.velocity = self.velocities[i]
                vehicle.acceleration = a[i] if i < len(a) else 0

        self.velocity_history.append(self.velocities.copy())
        self.internal_step_count += 1

        if self.internal_step_count % 10 == 0 and len(self.velocity_history) > 5:
            recent_v = np.array(self.velocity_history[-5:])
            mean_v_normalized = np.mean(recent_v)
            self.speed_ratio = mean_v_normalized / self.v0 if self.v0 > 0 else 0
            self.speed_ratio = np.clip(self.speed_ratio, 0, 1.2)
            self.mean_speed_kph = self.speed_kph * self.speed_ratio

    def step(self):
        self.update_idm_dynamics(delta_t=self.model.delta_t)


class IDMTrafficModel(mesa.Model):
    """IDM traffic model with dynamic density control."""

    def __init__(self, edges_gdf, num_vehicles_per_road=5, delta_t=0.02, seed=42):
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
            total_vehicles = sum(r.num_vehicles for r in self.road_agents)
            total_length = sum(r.length_m for r in self.road_agents)
            avg_speed = np.mean([r.mean_speed_kph for r in self.road_agents])

            density = total_vehicles / total_length if total_length > 0 else 0
            flow = density * (avg_speed / 3.6)

            self.fd_history['density'].append(density)
            self.fd_history['flow'].append(flow)
            self.fd_history['speed'].append(avg_speed)
            self.fd_history['phase'].append(phase)

        self.step_count += 1


# =============================================================================
# Static Visualization Function
# =============================================================================

def create_legend_figure(output_filename='legend.png'):
    """Create a separate figure with just the legend."""
    fig, ax = plt.subplots(figsize=(12, 3), facecolor='white')
    ax.axis('off')

    # Create dummy plots for legend
    ax.plot([], [], 'o-', color='lightgray', linewidth=3, markersize=8, label='Reference FD')
    ax.plot([], [], color='red', linestyle='--', linewidth=3, label='Critical ρ*')
    ax.plot([], [], 'o-', color='#1f77b4', linewidth=4, markersize=7, label='Increasing ↑')
    ax.plot([], [], 's--', color='#ff7f0e', linewidth=4, markersize=7, label='Decreasing ↓')
    ax.plot([], [], 'go', markersize=18, markeredgecolor='darkgreen', markeredgewidth=3, label='Start')
    ax.plot([], [], 'r*', markersize=24, markeredgecolor='darkred', markeredgewidth=3, label='End')

    # Create legend
    legend = ax.legend(loc='center', fontsize=20, framealpha=0.95, ncol=6,
                      columnspacing=2, handletextpad=1)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved legend: {output_filename}")
    plt.close()


def create_hysteresis_plot(model, fd_reference, analysis_result, output_filename='idm_hysteresis_cycle.png'):
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

    ax1.set_xlabel('Density ρ (veh/m)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Flow q (veh/s)', fontsize=20, fontweight='bold')
    ax1.set_xlim(0, 0.13)
    ax1.legend().set_visible(False)  # Hide legend
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

    ax2.set_xlabel('Density ρ (veh/m)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Speed v (m/s)', fontsize=20, fontweight='bold')
    ax2.set_xlim(0, 0.13)
    ax2.legend().set_visible(False)  # Hide legend
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # No title
    plt.tight_layout()

    # Save main figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_filename}")
    plt.close()

    # Create separate legend
    create_legend_figure(output_filename.replace('.png', '_legend.png'))


# =============================================================================
# Main Function
# =============================================================================

def generate_hysteresis_cycle(
    place="Singapore",
    network_type='drive',
    num_steps=300,
    delta_t=0.02,
    output_filename='idm_hysteresis_cycle.png',
    fd_densities=None,
    fd_L=1000.0,
    fd_v0=30.0,
    fd_s0=2.0,
    fd_T=1.5,
    fd_a_max=1.0,
    fd_b=2.0,
    fd_delta=4.0
):
    """Generate static hysteresis plot showing complete cycle."""
    print("=" * 80)
    print("IDM HYSTERESIS CYCLE - STATIC VISUALIZATION")
    print("=" * 80)

    print("\n[1/4] Computing reference fundamental diagram...")
    if fd_densities is None:
        fd_densities = np.linspace(0.02, 0.25, 20)

    fd_reference = run_idm_sweep(
        fd_densities, L=fd_L, v0=fd_v0, s0=fd_s0, T=fd_T,
        a_max=fd_a_max, b=fd_b, delta=fd_delta
    )
    analysis_result = analyze_fd(fd_reference)

    if analysis_result['breakpoint']:
        print(f"  ✓ Critical density: ρ* = {analysis_result['breakpoint'].x_star:.4f} veh/m")

    print(f"\n[2/4] Loading road network for {place}...")
    G = ox.graph_from_place(place, network_type=network_type)
    G = ox.add_edge_speeds(G)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    if not {'u', 'v', 'key'}.issubset(edges_gdf.columns):
        edges_gdf = edges_gdf.reset_index()

    print(f"  ✓ Loaded {len(edges_gdf)} road segments")

    print("\n[3/4] Running simulation with cyclical density...")
    model = IDMTrafficModel(edges_gdf, num_vehicles_per_road=5, delta_t=delta_t)

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
        'delta_t': 0.02,
        'output_filename': 'idm_hysteresis_cycle.png',
        'fd_densities': np.linspace(0.02, 0.25, 20),
        'fd_L': 1000.0,
        'fd_v0': 30.0,
        'fd_s0': 2.0,
        'fd_T': 1.5,
        'fd_a_max': 1.0,
        'fd_b': 2.0,
        'fd_delta': 4.0,
    }

    generate_hysteresis_cycle(**CONFIG)
