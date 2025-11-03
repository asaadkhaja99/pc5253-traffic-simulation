"""
Bando Traffic Simulation - Fundamental Diagram with Hysteresis Detection

Generates animated GIF showing fundamental diagram with:
- Reference FD curve from density sweep
- Current simulation state traversing through density space
- Critical point marked with red zone
- Scenario labels indicating traffic regime
- CYCLICAL density variation to reveal hysteresis

HYSTERESIS DETECTION:
    The simulation increases density (blue solid line ↑) then decreases it
    (orange dashed line ↓) to test if the system returns via the same path.

    IF HYSTERESIS EXISTS:
    - Blue and orange trajectories will NOT overlap
    - Forms a LOOP in the FD space
    - Different flow/speed at same density depending on history

    IF NO HYSTERESIS:
    - Blue and orange trajectories overlap
    - System retraces the same path
    - Flow/speed depends only on current density

USAGE:
    python bando_fd_critical_animation.py

CONFIGURATION:
    Edit CONFIG dictionary at bottom to customize parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import io
import os
import osmnx as ox
import geopandas as gpd
import mesa
import mesa_geo as mg
from shapely.geometry import Point

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'mesa'))
from critical_point_analysis import run_bando_sweep, analyze_fd

ox.settings.use_cache = True


# =============================================================================
# Bando Dynamics Functions
# =============================================================================

def optimal_vel(h, tau, L):
    """Optimal velocity function - returns normalized velocity (0-1)"""
    opt_v = tau/L * (np.tanh(h-2) + np.tanh(2)) / (1+np.tanh(2))
    return opt_v


def optimal_vel_bottleneck(h, tau, L, x, epsilon):
    """Optimal velocity with bottleneck effect"""
    xi = x % L
    bottleneck_factor = (1 - epsilon * np.exp(- (xi - (L/2))**2 ))
    opt_v = optimal_vel(h, tau, L)
    return bottleneck_factor * opt_v


def bando_acceleration(x, v, N, tau, alpha, bottleneck, epsilon, L):
    """Calculate Bando model acceleration for all vehicles"""
    xi1, xi2 = x, np.append(x, x+L)[1:N+1]
    vi1, vi2 = v, np.append(v, v)[1:N+1]
    h = xi2 - xi1

    if bottleneck:
        bando_term = optimal_vel_bottleneck(h, tau, L, x, epsilon) - vi1
    else:
        bando_term = optimal_vel(h, tau, L) - vi1

    aggre_term = vi2 - vi1
    acceleration = alpha * bando_term + (1-alpha) * aggre_term
    return acceleration


# =============================================================================
# Agent Classes
# =============================================================================

class VehicleAgent(mg.GeoAgent):
    """Vehicle agent following Bando dynamics."""

    def __init__(self, model, geometry, crs, road_segment, position, velocity, vehicle_index):
        super().__init__(model, geometry, crs)
        self.road_segment = road_segment
        self.position = position
        self.velocity = velocity
        self.vehicle_index = vehicle_index
        self.acceleration = 0.0

    def update_position_on_road(self):
        """Update geometry based on normalized position along road."""
        road_geom = self.road_segment.geometry
        point_on_road = road_geom.interpolate(self.position, normalized=True)
        self.geometry = point_on_road

    def step(self):
        """Update vehicle position based on Bando dynamics."""
        self.update_position_on_road()


class RoadAgent(mg.GeoAgent):
    """Road segment agent managing vehicle flow."""

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

        self.set_bando_params()

        self.positions = None
        self.velocities = None
        self.vehicles = []

        self.velocity_history = []
        self.internal_step_count = 0
        self.warmup_steps = 0

        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0

    def set_bando_params(self):
        """Set Bando parameters based on highway type."""
        highway_params = {
            'motorway': {'tau': 40, 'alpha': 0.9, 'epsilon': 0.2, 'base_ratio': 0.75},
            'trunk': {'tau': 35, 'alpha': 0.85, 'epsilon': 0.3, 'base_ratio': 0.80},
            'primary': {'tau': 30, 'alpha': 0.8, 'epsilon': 0.4, 'base_ratio': 0.85},
            'secondary': {'tau': 25, 'alpha': 0.75, 'epsilon': 0.5, 'base_ratio': 0.90},
            'tertiary': {'tau': 20, 'alpha': 0.7, 'epsilon': 0.6, 'base_ratio': 0.95},
            'residential': {'tau': 15, 'alpha': 0.6, 'epsilon': 0.7, 'base_ratio': 0.95}
        }

        highway_type = self.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'residential'
        elif not isinstance(highway_type, str):
            highway_type = str(highway_type) if highway_type else 'residential'

        params = highway_params.get(highway_type, highway_params['residential'])
        self.tau = params['tau']
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.base_ratio = params['base_ratio']
        self.L = 1.0
        self.bottleneck = highway_type in ['motorway', 'trunk', 'primary']

    def set_num_vehicles(self, num_vehicles):
        """Dynamically change number of vehicles on this road."""
        old_N = self.num_vehicles
        self.num_vehicles = num_vehicles

        if num_vehicles > old_N:
            # Add vehicles
            self._add_vehicles(num_vehicles - old_N)
        elif num_vehicles < old_N:
            # Remove vehicles
            self._remove_vehicles(old_N - num_vehicles)

    def _add_vehicles(self, count):
        """Add vehicles to this road."""
        for _ in range(count):
            # Add at random position
            new_pos = self.model.rng.random()
            new_vel = self.velocities.mean() if len(self.velocities) > 0 else 0.5

            self.positions = np.append(self.positions, new_pos)
            self.velocities = np.append(self.velocities, new_vel)

            point_geom = self.geometry.interpolate(new_pos, normalized=True)
            vehicle = VehicleAgent(
                model=self.model,
                geometry=point_geom,
                crs=self.crs,
                road_segment=self,
                position=new_pos,
                velocity=new_vel,
                vehicle_index=len(self.vehicles)
            )
            self.vehicles.append(vehicle)
            self.model.space.add_agents(vehicle)

    def _remove_vehicles(self, count):
        """Remove vehicles from this road."""
        for _ in range(min(count, len(self.vehicles))):
            if len(self.vehicles) > 0:
                vehicle = self.vehicles.pop()
                self.model.space.remove_agent(vehicle)
                self.positions = self.positions[:-1]
                self.velocities = self.velocities[:-1]

    def initialize_vehicles(self):
        """Initialize vehicle positions and velocities."""
        N = self.num_vehicles

        self.positions = np.linspace(0, self.L, N, endpoint=False)
        perturbation = self.model.rng.random(N) * 0.1
        self.positions = (self.positions + perturbation) % self.L

        if self.bottleneck:
            self.velocities = np.array([optimal_vel_bottleneck(self.L/N, self.tau, self.L, xi, self.epsilon)
                                       for xi in self.positions])
        else:
            self.velocities = np.full(N, optimal_vel(self.L/N, self.tau, self.L))

        self.velocities *= self.model.rng.uniform(0.8, 1.0, size=N)

        for i in range(N):
            point_geom = self.geometry.interpolate(self.positions[i], normalized=True)
            vehicle = VehicleAgent(
                model=self.model,
                geometry=point_geom,
                crs=self.crs,
                road_segment=self,
                position=self.positions[i],
                velocity=self.velocities[i],
                vehicle_index=i
            )
            self.vehicles.append(vehicle)
            self.model.space.add_agents(vehicle)

    def update_vehicles(self, delta_t):
        """Update all vehicles on this road using Bando dynamics."""
        N = self.num_vehicles
        if N < 2:
            return

        acceleration = bando_acceleration(
            self.positions, self.velocities, N,
            self.tau, self.alpha, self.bottleneck, self.epsilon, self.L
        )

        self.velocities = np.clip(self.velocities + acceleration * delta_t, 0, 1)
        self.positions = (self.positions + self.velocities * delta_t) % self.L

        for i, vehicle in enumerate(self.vehicles):
            if i < len(self.positions):
                vehicle.position = self.positions[i]
                vehicle.velocity = self.velocities[i]
                vehicle.acceleration = acceleration[i] if i < len(acceleration) else 0

        self.velocity_history.append(self.velocities.copy())
        self.internal_step_count += 1

        if self.internal_step_count > self.warmup_steps:
            recent_vels = np.concatenate(self.velocity_history[-10:])
            avg_speed_normalized = recent_vels.mean()
            self.speed_ratio = avg_speed_normalized
            self.mean_speed_kph = self.speed_ratio * self.speed_kph
        else:
            self.speed_ratio = self.velocities.mean() if len(self.velocities) > 0 else 0
            self.mean_speed_kph = self.speed_ratio * self.speed_kph

    def step(self):
        """Road agent step - update vehicles."""
        self.update_vehicles(self.model.delta_t)


# =============================================================================
# Model
# =============================================================================

class BandoTrafficModel(mesa.Model):
    """Bando traffic model with dynamic density control."""

    def __init__(self, edges_gdf, num_vehicles_per_road=5, delta_t=0.02, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.space = mg.GeoSpace(warn_crs_conversion=False)
        self.delta_t = delta_t

        self.edges_gdf = edges_gdf.to_crs(4326)
        self.road_agents = []

        self._initialize_roads(num_vehicles_per_road)

        # Tracking for fundamental diagram
        self.step_count = 0
        self.fd_history = {'density': [], 'flow': [], 'speed': []}

    def _initialize_roads(self, num_vehicles_per_road):
        """Initialize road agents with vehicles."""
        for idx, row in self.edges_gdf.iterrows():
            highway = row.get('highway', 'residential')
            if isinstance(highway, list):
                highway = highway[0] if highway else 'residential'
            elif not isinstance(highway, str):
                highway = str(highway) if highway else 'residential'

            road = RoadAgent(
                model=self,
                geometry=row.geometry,
                crs=self.edges_gdf.crs,
                edge_id=idx,
                u=row.get('u', idx),
                v=row.get('v', idx),
                key=row.get('key', 0),
                length_m=row.get('length', 100),
                speed_kph=row.get('speed_kph', 50),
                highway_type=highway,
                num_vehicles=num_vehicles_per_road
            )
            self.road_agents.append(road)
            self.space.add_agents(road)
            road.initialize_vehicles()

    def set_density_multiplier(self, multiplier):
        """Adjust density across all roads by multiplier."""
        for road in self.road_agents:
            base_vehicles = 5  # Base density
            new_vehicles = max(1, int(base_vehicles * multiplier))
            road.set_num_vehicles(new_vehicles)

    def step(self):
        """Execute one step of the model."""
        for road in self.road_agents:
            road.step()

        # Update fundamental diagram metrics
        if self.step_count % 5 == 0:
            total_vehicles = sum(r.num_vehicles for r in self.road_agents)
            total_length = sum(r.length_m for r in self.road_agents)
            avg_speed = np.mean([r.mean_speed_kph for r in self.road_agents])

            density = total_vehicles / total_length if total_length > 0 else 0
            flow = density * (avg_speed / 3.6)  # Convert km/h to m/s

            self.fd_history['density'].append(density)
            self.fd_history['flow'].append(flow)
            self.fd_history['speed'].append(avg_speed)

        self.step_count += 1


# =============================================================================
# Visualization Function
# =============================================================================

def create_fd_frame(model, fd_reference, analysis_result, step_num, density_multiplier, scenario_label):
    """
    Create fundamental diagram frame with current state.

    Args:
        model: Traffic model
        fd_reference: Reference FD from density sweep
        analysis_result: Critical point analysis
        step_num: Current step number
        density_multiplier: Current density level
        scenario_label: Text label for scenario
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor='white')

    # Plot 1: Flow vs Density
    ax1.plot(fd_reference.density, fd_reference.flow, 'o-',
             color='#2ca02c', linewidth=3, markersize=8, alpha=0.7,
             label='Reference FD', zorder=1)

    # Mark critical point
    if analysis_result and analysis_result['breakpoint']:
        bp = analysis_result['breakpoint']
        ax1.axvline(bp.x_star, color='red', linestyle='--', linewidth=2.5,
                   label=f'Critical ρ* = {bp.x_star:.4f}', zorder=2)
        ax1.axvspan(bp.x_star * 0.9, bp.x_star * 1.1,
                   color='red', alpha=0.15, label='Critical zone', zorder=0)

    # Plot current state
    if len(model.fd_history['density']) > 0:
        current_density = model.fd_history['density'][-1]
        current_flow = model.fd_history['flow'][-1]
        ax1.plot(current_density, current_flow, 'r*',
                markersize=30, label='Current state', zorder=10,
                markeredgecolor='darkred', markeredgewidth=2)

        # Plot trajectory (single color to see if it retraces or forms loop)
        if len(model.fd_history['density']) > 1:
            ax1.plot(model.fd_history['density'], model.fd_history['flow'],
                    'k-', alpha=0.4, linewidth=2, label='Trajectory', zorder=3)

    ax1.set_xlabel('Density ρ (veh/m)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Flow q (veh/s)', fontsize=14, fontweight='bold')
    ax1.set_title('Fundamental Diagram - Flow vs Density', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Speed vs Density
    ax2.plot(fd_reference.density, fd_reference.speed, 's-',
             color='#ff7f0e', linewidth=3, markersize=8, alpha=0.7,
             label='Reference speed', zorder=1)

    if analysis_result and analysis_result['breakpoint']:
        bp = analysis_result['breakpoint']
        ax2.axvline(bp.x_star, color='red', linestyle='--', linewidth=2.5, zorder=2)
        ax2.axvspan(bp.x_star * 0.9, bp.x_star * 1.1,
                   color='red', alpha=0.15, zorder=0)

    # Plot current state
    if len(model.fd_history['density']) > 0:
        current_density = model.fd_history['density'][-1]
        current_speed = model.fd_history['speed'][-1] / 3.6  # km/h to m/s
        ax2.plot(current_density, current_speed, 'r*',
                markersize=30, label='Current state', zorder=10,
                markeredgecolor='darkred', markeredgewidth=2)

        # Plot trajectory (single color to see if it retraces or forms loop)
        if len(model.fd_history['density']) > 1:
            densities = model.fd_history['density']
            speeds_ms = [s/3.6 for s in model.fd_history['speed']]
            ax2.plot(densities, speeds_ms,
                    'k-', alpha=0.4, linewidth=2, label='Trajectory', zorder=3)

    ax2.set_xlabel('Density ρ (veh/m)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Speed v (m/s)', fontsize=14, fontweight='bold')
    ax2.set_title('Fundamental Diagram - Speed vs Density', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Add scenario label
    scenario_color = {
        'Free-flow': '#1a9850',
        'Approaching Critical': '#fee08b',
        'Critical Zone': '#fc8d59',
        'Congested': '#d73027',
        'Heavy Congestion': '#8B0000'
    }
    color = scenario_color.get(scenario_label, '#666666')

    fig.text(0.5, 0.96, f'Step {step_num} | Density Multiplier: {density_multiplier:.2f}x | Scenario: {scenario_label}',
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.7', facecolor=color, alpha=0.3, edgecolor='black', linewidth=2))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)

    return img


# =============================================================================
# Main Animation Function
# =============================================================================

def generate_fd_animation(
    place="Singapore",
    network_type='drive',
    num_steps=200,
    delta_t=0.02,
    output_filename='bando_fd_critical_evolution.gif',
    fps=10,
    # Reference FD parameters
    fd_densities=None,
    fd_L=1000.0,
    fd_alpha=1.0,
    fd_v0=30.0,
    fd_h0=25.0,
    fd_delta=8.0
):
    """
    Generate FD animation showing critical transition with hysteresis detection.

    Density progression (CYCLICAL to reveal hysteresis):

    CYCLE 1 - INCREASING (steps 0-150):
    - Steps 0-50: Low density (0.5x) → Free-flow
    - Steps 50-100: Medium (1.0x) → Approaching critical
    - Steps 100-150: High (1.5x) → Cross critical point

    CYCLE 2 - DECREASING (steps 150-300):
    - Steps 150-200: High (1.5x) → Start recovery
    - Steps 200-250: Medium (1.0x) → Recovering
    - Steps 250-300: Low (0.5x) → Back to free-flow

    If HYSTERESIS exists: trajectory forms a LOOP (different paths up vs down)
    If NO hysteresis: trajectory retraces same path
    """
    print("=" * 80)
    print("BANDO FD CRITICAL TRANSITION ANIMATION")
    print("=" * 80)

    # =========================================================================
    # 1. Compute reference fundamental diagram
    # =========================================================================
    print("\n[1/4] Computing reference fundamental diagram...")
    if fd_densities is None:
        fd_densities = np.linspace(0.02, 0.25, 20)

    fd_reference = run_bando_sweep(
        fd_densities,
        L=fd_L,
        alpha=fd_alpha,
        v0=fd_v0,
        h0=fd_h0,
        delta=fd_delta
    )
    analysis_result = analyze_fd(fd_reference)

    if analysis_result['breakpoint']:
        print(f"  ✓ Critical density: ρ* = {analysis_result['breakpoint'].x_star:.4f} veh/m")

    # =========================================================================
    # 2. Load network
    # =========================================================================
    print(f"\n[2/4] Loading road network for {place}...")
    G = ox.graph_from_place(place, network_type=network_type)
    G = ox.add_edge_speeds(G)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    if not {'u', 'v', 'key'}.issubset(edges_gdf.columns):
        edges_gdf = edges_gdf.reset_index()

    print(f"  ✓ Loaded {len(edges_gdf)} road segments")

    # =========================================================================
    # 3. Initialize model
    # =========================================================================
    print("\n[3/4] Initializing traffic model...")
    model = BandoTrafficModel(edges_gdf, num_vehicles_per_road=5, delta_t=delta_t)
    total_vehicles = sum(r.num_vehicles for r in model.road_agents)
    print(f"  ✓ Created {total_vehicles} vehicles")

    # =========================================================================
    # 4. Generate frames with varying density
    # =========================================================================
    print(f"\n[4/4] Generating {num_steps} frames with density variation...")
    frames = []

    for i in range(num_steps):
        # Define CYCLICAL density progression with SMOOTH transitions
        # to better reveal hysteresis loops

        # CYCLE 1: Gradually INCREASE density (0 → 150)
        if i < 150:
            # Linear increase from 0.5x to 1.8x
            progress = i / 150.0
            multiplier = 0.5 + progress * 1.3  # 0.5 → 1.8
            phase = 'Increasing'

            if multiplier < 0.8:
                scenario = 'Free-flow ↑'
            elif multiplier < 1.2:
                scenario = 'Approaching Critical ↑'
            else:
                scenario = 'Critical Zone ↑'

        # CYCLE 2: Gradually DECREASE density (150 → 300)
        else:
            # Linear decrease from 1.8x back to 0.5x
            progress = (i - 150) / 150.0
            multiplier = 1.8 - progress * 1.3  # 1.8 → 0.5
            phase = 'Decreasing'

            if multiplier > 1.2:
                scenario = 'Critical Zone ↓'
            elif multiplier > 0.8:
                scenario = 'Recovering ↓'
            else:
                scenario = 'Free-flow ↓'

        # Update density every 10 steps for smoother transitions
        if i % 10 == 0:
            model.set_density_multiplier(multiplier)

        # Print major transitions
        if i in [0, 50, 100, 150, 200, 250]:
            print(f"  Step {i}: [{phase}] {scenario} (density {multiplier:.2f}x)")

        if i % 20 == 0:
            print(f"  Step {i}/{num_steps}...")

        # Create frame
        frame = create_fd_frame(model, fd_reference, analysis_result, i, multiplier, scenario)
        frames.append(frame)

        # Step model
        model.step()

    # =========================================================================
    # 5. Save GIF
    # =========================================================================
    print(f"\nSaving GIF to {output_filename}...")
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000/fps),
        loop=0,
        optimize=False
    )

    print(f"✓ GIF saved: {output_filename}")
    print(f"  - Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")
    print(f"  - Frames: {len(frames)}")
    print(f"  - FPS: {fps}")
    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    CONFIG = {
        'place': 'Singapore',
        'network_type': 'drive',
        'num_steps': 300,  # Increased for full cycle
        'delta_t': 0.02,
        'output_filename': 'bando_fd_critical_evolution.gif',
        'fps': 10,
        'fd_densities': np.linspace(0.02, 0.25, 20),
        'fd_L': 1000.0,
        'fd_alpha': 1.0,
        'fd_v0': 30.0,
        'fd_h0': 25.0,
        'fd_delta': 8.0,
    }

    generate_fd_animation(**CONFIG)
