"""
Create static evacuation zone map for reports.

Generates a high-quality static map showing:
- Evacuation zone (danger area)
- Safe zones (destinations)
- Road network
- Clear legend and labels

For use in academic reports and presentations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import geopandas as gpd
import osmnx as ox
import contextily as cx
from shapely.geometry import Point
import numpy as np
from pathlib import Path
from ..core.evacuation_base import EvacuationConfig, EvacuationModel
from ..core.plot_utils import setup_high_res_plot_style, COLORS


def create_evacuation_zone_map(
    output_path: str = "output/evacuation/evacuation_zone_map.png",
    dpi: int = 300,
    figsize: tuple = (12, 10)
):
    """
    Create static map showing evacuation and safe zones.

    Args:
        output_path: Path to save the output image
        dpi: Resolution (300 for publication quality)
        figsize: Figure size in inches
    """
    print("Creating evacuation zone map...")

    # Initialize model to get network
    config = EvacuationConfig(
        bbox_south=1.26,
        bbox_north=1.295,
        bbox_west=103.84,
        bbox_east=103.88,
        evacuation_center=(1.2775, 103.860),
        safe_zone_radius_km=2.0
    )

    print("Loading road network...")
    model = EvacuationModel(config)

    # Convert to Web Mercator for visualization
    edges_gdf_wm = model.edges_gdf.to_crs(epsg=3857)
    nodes_gdf_wm = model.nodes_gdf.to_crs(epsg=3857)

    # Create evacuation center point
    evac_center_point = Point(config.evacuation_center[1], config.evacuation_center[0])  # lon, lat
    evac_center_gdf = gpd.GeoDataFrame(
        [{'geometry': evac_center_point}],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    # Create evacuation zone circle (danger area)
    evac_zone_radius_m = config.safe_zone_radius_km * 1000
    evac_zone_buffer = evac_center_gdf.geometry.buffer(evac_zone_radius_m)

    # Identify origin nodes (inside evacuation zone)
    origin_nodes = []
    safe_zone_nodes = []

    for node_id in model.G.nodes():
        if node_id not in model.nodes_gdf.index:
            continue

        node_data = model.nodes_gdf.loc[node_id]
        node_point = node_data.geometry

        # Calculate distance from evacuation center
        distance_m = evac_center_point.distance(node_point) * 111139  # Approx meters

        if distance_m < evac_zone_radius_m:
            origin_nodes.append(node_id)
        else:
            safe_zone_nodes.append(node_id)

    # Convert node lists to GeoDataFrames
    origin_gdf = nodes_gdf_wm.loc[nodes_gdf_wm.index.isin(origin_nodes)]
    safe_gdf = nodes_gdf_wm.loc[nodes_gdf_wm.index.isin(safe_zone_nodes)]

    print(f"  Origin nodes (danger zone): {len(origin_nodes)}")
    print(f"  Safe zone nodes: {len(safe_zone_nodes)}")

    # Setup plot style
    setup_high_res_plot_style(fontsize=23, dpi=dpi)

    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Add basemap
    print("Adding basemap...")
    try:
        cx.add_basemap(
            ax,
            crs=edges_gdf_wm.crs.to_string(),
            source=cx.providers.CartoDB.Positron,
            alpha=0.8,
            zorder=0
        )
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")

    # Plot road network
    edges_gdf_wm.plot(
        ax=ax,
        color=COLORS['road_network'],
        linewidth=0.8,
        alpha=0.6,
        zorder=1
    )

    # Plot evacuation zone (danger area)
    evac_zone_buffer.plot(
        ax=ax,
        color=COLORS['danger_zone'],
        alpha=0.2,
        edgecolor='#c0392b',
        linewidth=2.5,
        linestyle='--',
        zorder=2
    )

    # Plot evacuation center
    evac_center_gdf.plot(
        ax=ax,
        color=COLORS['danger_zone'],
        markersize=300,
        marker='*',
        edgecolor='white',
        linewidth=1.5,
        zorder=5
    )

    # Plot origin nodes (sample to avoid clutter)
    if len(origin_gdf) > 100:
        origin_sample = origin_gdf.sample(n=100, random_state=42)
    else:
        origin_sample = origin_gdf

    origin_sample.plot(
        ax=ax,
        color=COLORS['origin'],
        markersize=15,
        alpha=0.6,
        zorder=3
    )

    # Plot safe zone nodes (sample to avoid clutter)
    if len(safe_gdf) > 100:
        safe_sample = safe_gdf.sample(n=100, random_state=42)
    else:
        safe_sample = safe_gdf

    safe_sample.plot(
        ax=ax,
        color=COLORS['safe_zone'],
        markersize=15,
        alpha=0.6,
        zorder=3
    )

    # Add scale indicator
    # Get bounds in lat/lon
    bounds = model.edges_gdf.total_bounds  # [minx, miny, maxx, maxy]
    lat_center = (bounds[1] + bounds[3]) / 2

    # Add text annotations
    # Evacuation center label
    evac_center_wm = evac_center_gdf.geometry.iloc[0]
    ax.annotate(
        'EVACUATION\nCENTER\n(Marina Bay)',
        xy=(evac_center_wm.x, evac_center_wm.y),
        xytext=(20, 40),
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
        color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#c0392b', linewidth=2),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='#c0392b', lw=2)
    )

    # Evacuation zone radius label
    zone_edge_point = evac_zone_buffer.iloc[0].boundary.representative_point()
    ax.annotate(
        f'{config.safe_zone_radius_km} km radius\n(Danger Zone)',
        xy=(zone_edge_point.x, zone_edge_point.y),
        xytext=(-80, -40),
        textcoords='offset points',
        fontsize=9,
        color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#c0392b', linewidth=1.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='#c0392b', lw=1.5)
    )

    # Safe zone label (outside circle)
    # Get a point outside the evacuation zone
    bounds_wm = edges_gdf_wm.total_bounds
    safe_label_x = bounds_wm[2] - (bounds_wm[2] - bounds_wm[0]) * 0.15
    safe_label_y = bounds_wm[3] - (bounds_wm[3] - bounds_wm[1]) * 0.1

    ax.text(
        safe_label_x, safe_label_y,
        'SAFE ZONE\n(Destinations)',
        fontsize=11,
        fontweight='bold',
        color='#27ae60',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#27ae60', linewidth=2),
        ha='center',
        va='top',
        zorder=6
    )

    # Add directional arrow showing evacuation flow
    # From center outward
    arrow_start_x = evac_center_wm.x + evac_zone_radius_m * 0.5
    arrow_start_y = evac_center_wm.y
    arrow_end_x = evac_center_wm.x + evac_zone_radius_m * 1.3
    arrow_end_y = evac_center_wm.y

    ax.annotate(
        '',
        xy=(arrow_end_x, arrow_end_y),
        xytext=(arrow_start_x, arrow_start_y),
        arrowprops=dict(
            arrowstyle='->',
            lw=4,
            color='#2c3e50',
            alpha=0.7
        ),
        zorder=4
    )

    ax.text(
        (arrow_start_x + arrow_end_x) / 2,
        arrow_end_y + evac_zone_radius_m * 0.15,
        'Evacuation Direction',
        fontsize=10,
        fontweight='bold',
        color='#2c3e50',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8),
        zorder=4
    )

    # Set axis limits to road network bounds
    bounds_wm = edges_gdf_wm.total_bounds
    margin = (bounds_wm[2] - bounds_wm[0]) * 0.05  # 5% margin
    ax.set_xlim(bounds_wm[0] - margin, bounds_wm[2] + margin)
    ax.set_ylim(bounds_wm[1] - margin, bounds_wm[3] + margin)

    # Set aspect ratio to equal
    ax.set_aspect('equal', adjustable='box')

    # Styling
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.set_title(
        'Evacuation Zone Map: Marina Bay, Singapore\nOrigins (Red) → Safe Zones (Green)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # Create custom legend handles
    legend_handles = [
        mlines.Line2D([], [], color=COLORS['road_network'], linewidth=2, label='Road Network'),
        mpatches.Patch(facecolor=COLORS['danger_zone'], edgecolor='#c0392b', alpha=0.2, linestyle='--', linewidth=2, label='Evacuation Zone (Danger)'),
        mlines.Line2D([], [], color=COLORS['danger_zone'], marker='*', linestyle='None', markersize=15, markeredgecolor='white', markeredgewidth=1.5, label='Evacuation Center'),
        mlines.Line2D([], [], color=COLORS['origin'], marker='o', linestyle='None', markersize=8, alpha=0.6, label=f'Origin Nodes (n={len(origin_nodes)})'),
        mlines.Line2D([], [], color=COLORS['safe_zone'], marker='o', linestyle='None', markersize=8, alpha=0.6, label=f'Safe Zone Nodes (n={len(safe_zone_nodes)})')
    ]

    # Legend
    legend = ax.legend(
        handles=legend_handles,
        loc='upper left',
        fontsize=10,
        framealpha=0.95,
        edgecolor='black',
        fancybox=True,
        shadow=True
    )
    legend.set_zorder(10)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=0)

    # Add info box
    info_text = (
        f"Study Area: Marina Bay, Singapore\n"
        f"Bbox: {config.bbox_south:.3f}°S - {config.bbox_north:.3f}°N, "
        f"{config.bbox_west:.3f}°E - {config.bbox_east:.3f}°E\n"
        f"Road Segments: {len(model.edges_gdf)}\n"
        f"Network Nodes: {len(model.nodes_gdf)}\n"
        f"Evacuation Radius: {config.safe_zone_radius_km} km"
    )

    ax.text(
        0.02, 0.02,
        info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1),
        family='monospace',
        zorder=10
    )

    # Add north arrow
    # Position in upper right
    arrow_x = 0.95
    arrow_y = 0.95
    arrow_length = 0.05

    ax.annotate(
        '',
        xy=(arrow_x, arrow_y),
        xytext=(arrow_x, arrow_y - arrow_length),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', lw=2.5, color='black'),
        zorder=10
    )
    ax.text(
        arrow_x, arrow_y + 0.02,
        'N',
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='bottom',
        zorder=10
    )

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving map to {output_path}...")
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    print(f"✓ Map saved: {output_path}")

    plt.close()

    return str(output_path)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate static evacuation zone map for reports'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/evacuation/evacuation_zone_map.png',
        help='Output file path (default: output/evacuation/evacuation_zone_map.png)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution in DPI (default: 300 for publication quality)'
    )

    parser.add_argument(
        '--figsize',
        type=str,
        default='12,10',
        help='Figure size as "width,height" in inches (default: 12,10)'
    )

    args = parser.parse_args()

    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(',')))

    # Create map
    output_path = create_evacuation_zone_map(
        output_path=args.output,
        dpi=args.dpi,
        figsize=figsize
    )

    print("\n" + "=" * 80)
    print("EVACUATION ZONE MAP CREATED")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Resolution: {args.dpi} DPI")
    print(f"Size: {figsize[0]}\" x {figsize[1]}\"")
    print("=" * 80)


if __name__ == '__main__':
    main()
