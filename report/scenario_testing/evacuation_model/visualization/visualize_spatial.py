"""
Spatial Evacuation Visualization with Map Background

Creates animated spatial visualizations of evacuation progress on Singapore road network
with map background showing vehicle positions and congestion levels at each timestep.

Features:
- Basemap with Singapore road network
- Vehicle density visualization on roads
- Congestion heatmap (speed ratio)
- Evacuee positions (optional)
- Screenshot capture at regular intervals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import geopandas as gpd
from pathlib import Path
from typing import Optional
import contextily as cx


class SpatialEvacuationVisualizer:
    """
    Visualizer for spatial evacuation dynamics on map background.
    """

    def __init__(self, model, output_dir, screenshot_interval=60):
        """
        Initialize visualizer.

        Args:
            model: EvacuationModel instance
            output_dir: Directory to save screenshots
            screenshot_interval: Capture screenshot every N timesteps
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_interval = screenshot_interval

        # Convert to Web Mercator for basemap
        self.edges_gdf_wm = self.model.edges_gdf.to_crs(epsg=3857)
        self.nodes_gdf_wm = self.model.nodes_gdf.to_crs(epsg=3857)

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 12), facecolor='white')

        # Setup static elements
        self._setup_basemap()

        print(f"Spatial visualizer initialized. Screenshots will be saved to: {self.output_dir}")
        print(f"Capture interval: every {screenshot_interval} timesteps")

    def _setup_basemap(self):
        """Setup basemap with road network."""
        # Plot roads as base layer (all roads in gray)
        self.edges_gdf_wm.plot(
            ax=self.ax,
            linewidth=0.5,
            color='#cccccc',
            alpha=0.5,
            zorder=1
        )

        # Add basemap
        try:
            cx.add_basemap(
                self.ax,
                crs=self.edges_gdf_wm.crs.to_string(),
                source=cx.providers.CartoDB.Positron,
                alpha=0.6,
                zorder=0
            )
        except Exception as e:
            print(f"Warning: Could not load basemap: {e}")
            print("Continuing without basemap...")

        # Set bounds
        minx, miny, maxx, maxy = self.edges_gdf_wm.total_bounds
        margin = (maxx - minx) * 0.05
        self.ax.set_xlim(minx - margin, maxx + margin)
        self.ax.set_ylim(miny - margin, maxy + margin)

        # Remove axis labels
        self.ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        self.ax.tick_params(labelsize=9)

    def capture_snapshot(self, timestep, force=False):
        """
        Capture spatial snapshot at current timestep.

        Args:
            timestep: Current simulation timestep
            force: Force capture even if not at interval
        """
        # Check if we should capture
        if not force and timestep % self.screenshot_interval != 0:
            return

        print(f"  Capturing snapshot at t={timestep}s...", end='', flush=True)

        # Clear dynamic elements (keep basemap)
        for artist in self.ax.collections[1:]:  # Skip first collection (basemap roads)
            artist.remove()
        for artist in self.ax.patches:
            artist.remove()
        for txt in self.ax.texts:
            txt.remove()

        # Collect road states
        road_colors = []
        road_widths = []
        road_geometries = []

        for idx, row in self.edges_gdf_wm.iterrows():
            edge_tuple = (row['u'], row['v'], row['key'])

            if edge_tuple not in self.model.road_agents:
                continue

            road_agent = self.model.road_agents[edge_tuple]

            # Get road state
            vehicle_count = np.sum(road_agent.road >= 0)
            occupancy = vehicle_count / road_agent.num_cells if road_agent.num_cells > 0 else 0
            speed_ratio = road_agent.mean_speed_kph / road_agent.speed_limit_kph if road_agent.speed_limit_kph > 0 else 1.0

            # Color by congestion (speed ratio)
            if speed_ratio < 0.3:
                color = '#e74c3c'  # Red - severe congestion
            elif speed_ratio < 0.6:
                color = '#f39c12'  # Orange - moderate congestion
            elif speed_ratio < 0.8:
                color = '#f1c40f'  # Yellow - light congestion
            else:
                color = '#2ecc71'  # Green - free flow

            # Width by occupancy
            width = 0.5 + occupancy * 3.0  # 0.5 to 3.5

            road_colors.append(color)
            road_widths.append(width)
            road_geometries.append(row.geometry)

        # Plot roads with congestion colors
        for geom, color, width in zip(road_geometries, road_colors, road_widths):
            if geom.geom_type == 'LineString':
                coords = np.array(geom.coords)
                self.ax.plot(
                    coords[:, 0], coords[:, 1],
                    color=color,
                    linewidth=width,
                    alpha=0.8,
                    zorder=2,
                    solid_capstyle='round'
                )

        # Add evacuation zone circle
        evac_center_wm = self.nodes_gdf_wm.geometry.centroid.centroid
        circle = plt.Circle(
            (evac_center_wm.x, evac_center_wm.y),
            self.model.config.safe_zone_radius_km * 1000 * 0.5,  # Half radius in meters
            color='red',
            fill=False,
            linewidth=2,
            linestyle='--',
            alpha=0.5,
            zorder=3,
            label='Evacuation Zone'
        )
        self.ax.add_patch(circle)

        # Add statistics overlay
        evacuated_count = sum(1 for e in self.model.evacuees if e.evacuated)
        evacuated_pct = (evacuated_count / len(self.model.evacuees) * 100) if len(self.model.evacuees) > 0 else 0

        congested_roads = sum(1 for ra in self.model.road_agents.values()
                             if ra.mean_speed_kph / ra.speed_limit_kph < 0.3 if ra.speed_limit_kph > 0)

        mean_speeds = [ra.mean_speed_kph for ra in self.model.road_agents.values() if ra.mean_speed_kph > 0]
        mean_network_speed = np.mean(mean_speeds) if mean_speeds else 0

        stats_text = (
            f"Time: {timestep}s ({timestep/60:.1f} min)\n"
            f"Evacuated: {evacuated_count}/{len(self.model.evacuees)} ({evacuated_pct:.1f}%)\n"
            f"Congested Roads: {congested_roads}\n"
            f"Mean Speed: {mean_network_speed:.1f} km/h"
        )

        self.ax.text(
            0.02, 0.98, stats_text,
            transform=self.ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5),
            family='monospace',
            zorder=10
        )

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#2ecc71', label='Free Flow (>80% speed)'),
            mpatches.Patch(facecolor='#f1c40f', label='Light Congestion (60-80%)'),
            mpatches.Patch(facecolor='#f39c12', label='Moderate Congestion (30-60%)'),
            mpatches.Patch(facecolor='#e74c3c', label='Severe Congestion (<30%)')
        ]

        legend = self.ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9,
            edgecolor='black',
            fancybox=True,
            shadow=True
        )
        legend.set_zorder(10)

        # Title
        scenario_name = getattr(self.model, 'scenario_name', 'Evacuation')
        self.ax.set_title(
            f'{scenario_name} - Singapore Marina Bay Evacuation',
            fontsize=16,
            fontweight='bold',
            pad=15
        )

        # Save screenshot
        filename = f"snapshot_t{timestep:05d}.png"
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f" saved to {filename}")

    def close(self):
        """Close visualization."""
        plt.close(self.fig)


def create_spatial_animation(
    scenario_name,
    num_agents,
    spawn_function,
    output_dir,
    screenshot_interval=60,
    max_steps=7200
):
    """
    Run evacuation scenario with spatial visualization.

    Args:
        scenario_name: Name of scenario (e.g., 'Simultaneous', 'Staged')
        num_agents: Number of evacuees
        spawn_function: Function to spawn agents (takes model, num_agents)
        output_dir: Directory to save screenshots
        screenshot_interval: Capture interval in timesteps
        max_steps: Maximum simulation steps

    Returns:
        EvacuationModel instance after simulation
    """
    from evacuation_base import EvacuationModel, EvacuationConfig

    print("=" * 80)
    print(f"SPATIAL EVACUATION VISUALIZATION: {scenario_name}")
    print("=" * 80)

    # Create model
    config = EvacuationConfig(num_agents=num_agents, max_steps=max_steps)
    model = EvacuationModel(config)
    model.scenario_name = scenario_name

    # Create visualizer
    viz_output_dir = Path(output_dir) / scenario_name.lower().replace(' ', '_')
    visualizer = SpatialEvacuationVisualizer(
        model=model,
        output_dir=viz_output_dir,
        screenshot_interval=screenshot_interval
    )

    # Spawn agents
    print(f"\n[1/3] Spawning {num_agents} evacuees...")
    spawn_function(model, num_agents)

    # Capture initial state
    print(f"\n[2/3] Running simulation with spatial visualization...")
    visualizer.capture_snapshot(0, force=True)

    # Run simulation
    step = 0
    while step < max_steps:
        step += 1
        model.step()

        # Capture snapshot
        visualizer.capture_snapshot(step)

        # Progress update
        if step % 300 == 0:
            evacuated = sum(1 for e in model.evacuees if e.evacuated)
            pct = evacuated / len(model.evacuees) * 100 if model.evacuees else 0
            print(f"  Step {step}/{max_steps} - Evacuated: {evacuated}/{len(model.evacuees)} ({pct:.1f}%)")

        # Check if evacuation complete
        evacuated_count = sum(1 for e in model.evacuees if e.evacuated)
        if evacuated_count >= len(model.evacuees) * 0.99:
            print(f"  Evacuation complete at t={step}s")
            visualizer.capture_snapshot(step, force=True)
            break

    # Final snapshot
    if step >= max_steps:
        print(f"  Max steps reached: {max_steps}")
        visualizer.capture_snapshot(max_steps, force=True)

    # Cleanup
    visualizer.close()

    print(f"\n[3/3] Spatial visualization complete!")
    print(f"  Snapshots saved to: {viz_output_dir}")
    print(f"  Total snapshots: {len(list(viz_output_dir.glob('snapshot_*.png')))}")

    return model


def main():
    """Example: Visualize simultaneous evacuation."""
    from evacuation_base import get_origin_nodes, assign_destinations

    def spawn_simultaneous(model, num_agents):
        """Spawn all agents at t=0."""
        origins = get_origin_nodes(model, num_agents)
        destinations = assign_destinations(model, origins)

        for origin, dest in zip(origins, destinations):
            evacuee = model.spawn_evacuee(origin, dest)
            if evacuee:
                model.evacuees.append(evacuee)
                model.space.add_agents(evacuee)

        print(f"  Spawned {len(model.evacuees)} evacuees")

    # Run with spatial visualization
    output_dir = Path(__file__).parent.parent / "output" / "evacuation" / "spatial_viz"

    create_spatial_animation(
        scenario_name='Simultaneous',
        num_agents=500,  # Use smaller number for faster testing
        spawn_function=spawn_simultaneous,
        output_dir=output_dir,
        screenshot_interval=60,  # Every 60 seconds = 1 minute
        max_steps=3600  # 1 hour max
    )

    print("\n" + "=" * 80)
    print("To create a video from screenshots, run:")
    print(f"  cd {output_dir}/simultaneous")
    print("  ffmpeg -framerate 10 -pattern_type glob -i 'snapshot_*.png' \\")
    print("         -c:v libx264 -pix_fmt yuv420p evacuation_animation.mp4")
    print("=" * 80)


if __name__ == '__main__':
    main()
