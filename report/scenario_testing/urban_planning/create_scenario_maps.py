"""
Create Urban Planning Scenario Map - Paya Lebar

Generates map visualization for Paya Lebar localized incident scenario:
- Lane closure near Paya Lebar MRT interchange
- Shows road network hierarchy
- Highlights disruption area
- Uses IDM with Gaussian bottleneck factor (Eq. 17)
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextily as cx
from pathlib import Path
from shapely.geometry import Point, LineString

from plot_utils import setup_high_res_plot_style, COLORS


def create_paya_lebar_map(output_path="output/urban_planning/paya_lebar_scenario_map.png"):
    """
    Create map showing Paya Lebar scenario setup.

    Centered on Paya Lebar MRT junction with tight zoom showing:
    - Road network in immediate vicinity of MRT
    - Paya Lebar MRT station location (junction center)
    - Lane closure location (disruption)
    - Major vs residential roads
    """
    print("=" * 80)
    print("CREATING PAYA LEBAR SCENARIO MAP")
    print("=" * 80)

    # Apply consistent styling - increase DPI for zoomed view
    setup_high_res_plot_style(fontsize=20, dpi=300)

    # Import model to get network
    from scenario_paya_lebar import UrbanPlanningModel, UrbanPlanningConfig

    # Create model to load network - ZOOMED OUT VIEW
    config = UrbanPlanningConfig(
        # Zoomed out bbox around Paya Lebar MRT (wider view)
        bbox_north=1.3280,  # ~1.1km north of MRT
        bbox_south=1.3070,  # ~1.2km south of MRT
        bbox_east=103.9040,  # ~1.2km east of MRT
        bbox_west=103.8810,  # ~1.3km west of MRT

        num_vehicles=100,  # Small number, we just need the network
        seed=42
    )
    print("\nLoading Paya Lebar network (tight bounds matching simulation)...")
    model = UrbanPlanningModel(config)

    # Convert to Web Mercator for basemap
    edges_gdf_wm = model.edges_gdf.to_crs(epsg=3857)
    nodes_gdf_wm = model.nodes_gdf.to_crs(epsg=3857)

    # Normalize highway type (can be list or string)
    def normalize_highway(hw):
        if isinstance(hw, list):
            return hw[0] if hw else 'unknown'
        return hw if hw else 'unknown'
    edges_gdf_wm['highway'] = edges_gdf_wm['highway'].apply(normalize_highway)

    # Create figure - square aspect ratio for junction view
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')

    # Plot roads first to establish axis limits
    print("Plotting road network...")

    # Categorize roads
    major_types = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link']
    secondary_types = ['secondary', 'tertiary', 'secondary_link', 'tertiary_link']

    # Plot residential roads (thin, gray)
    residential_edges = edges_gdf_wm[~edges_gdf_wm['highway'].isin(major_types + secondary_types)]
    if len(residential_edges) > 0:
        residential_edges.plot(
            ax=ax,
            color='#95a5a6',
            linewidth=0.5,
            alpha=0.4,
            zorder=1,
            label='Residential Roads'
        )

    # Plot secondary roads (medium, blue)
    secondary_edges = edges_gdf_wm[edges_gdf_wm['highway'].isin(secondary_types)]
    if len(secondary_edges) > 0:
        secondary_edges.plot(
            ax=ax,
            color='#3498db',
            linewidth=1.5,
            alpha=0.7,
            zorder=2,
            label='Secondary Roads'
        )

    # Plot major roads (thick, dark blue)
    major_edges = edges_gdf_wm[edges_gdf_wm['highway'].isin(major_types)]
    if len(major_edges) > 0:
        major_edges.plot(
            ax=ax,
            color='#2c3e50',
            linewidth=2.5,
            alpha=0.9,
            zorder=3,
            label='Major Roads'
        )

    # Mark Paya Lebar MRT location (junction center)
    # Paya Lebar MRT: 1.31765°N, 103.89271°E
    paya_lebar_point = gpd.GeoDataFrame(
        geometry=[Point(103.89271, 1.31765)],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    paya_lebar_point.plot(
        ax=ax,
        color='#e74c3c',
        markersize=600,  # Larger marker for zoomed view
        marker='*',  # Star for junction
        edgecolor='white',
        linewidth=3,
        zorder=10,
        label='Paya Lebar MRT Junction'
    )

    # Mark lane closure area (circle around junction)
    closure_radius_m = 150  # 150m radius from junction
    closure_circle = paya_lebar_point.geometry.iloc[0].buffer(closure_radius_m)
    gpd.GeoSeries([closure_circle], crs='EPSG:3857').plot(
        ax=ax,
        facecolor='#e74c3c',
        edgecolor='#c0392b',
        alpha=0.2,
        linewidth=4,
        linestyle='--',
        zorder=4,
        label='Lane Closure Zone (IDM Bottleneck)'
    )

    # Mark nodes in geographic zones (matching simulation logic)
    print("Plotting origin and destination zones...")

    from shapely.geometry import box

    # Paya Lebar MRT Coordinates (Incident Center)
    # Lat: 1.3177, Lon: 103.8926

    # Define Origin Zone (South of Incident) - 50% smaller
    # We want traffic moving North, so they start South.
    # 350m to 650m South of MRT (was 200m to 800m)
    # Keep Longitude tight to ensure they don't take parallel streets
    origin_zone = {
        'lat_min': 1.3130,
        'lat_max': 1.3160,
        'lon_min': 103.891,
        'lon_max': 103.896,
    }

    # Define Destination Zone (North of Incident) - 50% smaller
    # 350m to 650m North of MRT (was 200m to 800m)
    destination_zone = {
        'lat_min': 1.3190,
        'lat_max': 1.3220,
        'lon_min': 103.891,
        'lon_max': 103.896,
    }

    # Create zone polygons
    origin_box = box(origin_zone['lon_min'], origin_zone['lat_min'],
                     origin_zone['lon_max'], origin_zone['lat_max'])
    destination_box = box(destination_zone['lon_min'], destination_zone['lat_min'],
                          destination_zone['lon_max'], destination_zone['lat_max'])

    # Convert to Web Mercator
    origin_box_gdf = gpd.GeoDataFrame(geometry=[origin_box], crs='EPSG:4326').to_crs(epsg=3857)
    destination_box_gdf = gpd.GeoDataFrame(geometry=[destination_box], crs='EPSG:4326').to_crs(epsg=3857)

    # Plot origin zone (blue rectangle)
    origin_box_gdf.boundary.plot(ax=ax, color='#3498db', linewidth=3, linestyle='--', zorder=7, alpha=0.9)
    origin_box_gdf.plot(ax=ax, facecolor='#3498db', alpha=0.1, zorder=5)

    # Plot destination zone (green rectangle)
    destination_box_gdf.boundary.plot(ax=ax, color='#27ae60', linewidth=3, linestyle='--', zorder=7, alpha=0.9)
    destination_box_gdf.plot(ax=ax, facecolor='#27ae60', alpha=0.1, zorder=5)

    # Select nodes within zones (matching simulation logic exactly)
    origin_nodes = []
    for node, data in model.nodes_gdf.iterrows():
        if (origin_zone['lat_min'] < data.y < origin_zone['lat_max']) and \
           (origin_zone['lon_min'] < data.x < origin_zone['lon_max']):
            origin_nodes.append(node)

    destination_nodes = []
    for node, data in model.nodes_gdf.iterrows():
        if (destination_zone['lat_min'] < data.y < destination_zone['lat_max']) and \
           (destination_zone['lon_min'] < data.x < destination_zone['lon_max']):
            destination_nodes.append(node)

    # Plot Origin nodes (before bottleneck) in BLUE
    if len(origin_nodes) > 0:
        origin_node_geoms = nodes_gdf_wm.loc[origin_nodes]
        origin_node_geoms.plot(
            ax=ax,
            color='#3498db',  # Blue for origins
            markersize=150,
            marker='o',
            edgecolor='white',
            linewidth=2.5,
            alpha=0.8,
            zorder=8,
            label=f'Origin Nodes (Before Bottleneck, n={len(origin_nodes)})'
        )

    # Plot Destination nodes (after bottleneck) in GREEN
    if len(destination_nodes) > 0:
        destination_node_geoms = nodes_gdf_wm.loc[destination_nodes]
        destination_node_geoms.plot(
            ax=ax,
            color='#27ae60',  # Green for destinations
            markersize=150,
            marker='s',  # Square for destinations
            edgecolor='white',
            linewidth=2.5,
            alpha=0.8,
            zorder=8,
            label=f'Destination Nodes (After Bottleneck, n={len(destination_nodes)})'
        )

    print(f"  Plotted {len(origin_nodes)} origin nodes (before bottleneck)")
    print(f"  Plotted {len(destination_nodes)} destination nodes (after bottleneck)")

    # Add basemap after plotting to get proper bounds
    print("Adding basemap...")
    try:
        cx.add_basemap(
            ax,
            crs=edges_gdf_wm.crs.to_string(),
            source=cx.providers.CartoDB.Positron,
            alpha=0.5,
            zorder=0,
            zoom=16  # Higher zoom for junction detail
        )
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")

    # Set axis limits to show ALL origin and destination nodes
    # Get bounds from all O-D nodes
    all_od_nodes = origin_nodes + destination_nodes
    if len(all_od_nodes) > 0:
        od_node_geoms = nodes_gdf_wm.loc[all_od_nodes]

        # Get bounding box of all O-D nodes
        minx, miny, maxx, maxy = od_node_geoms.total_bounds

        # Add margin to ensure all nodes are visible
        margin_x = (maxx - minx) * 0.15  # 15% margin
        margin_y = (maxy - miny) * 0.15

        # Ensure minimum margin of 200m
        margin_x = max(margin_x, 200)
        margin_y = max(margin_y, 200)

        ax.set_xlim(minx - margin_x, maxx + margin_x)
        ax.set_ylim(miny - margin_y, maxy + margin_y)
    else:
        # Fallback: center on junction
        junction_x = paya_lebar_point.geometry.iloc[0].x
        junction_y = paya_lebar_point.geometry.iloc[0].y
        margin = 400
        ax.set_xlim(junction_x - margin, junction_x + margin)
        ax.set_ylim(junction_y - margin, junction_y + margin)

    # Add scale bar and title
    ax.set_title('Paya Lebar Localized Incident - Zoned O-D Pairs\nIDM with Gaussian Bottleneck (ε=0.8, σ=50m)',
                fontweight='bold', pad=20, fontsize=22)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c',
                  markersize=18, markeredgecolor='white', markeredgewidth=2,
                  label='Paya Lebar Junction', linestyle='None'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='#c0392b', alpha=0.2,
                      linestyle='--', linewidth=2.5, label='Bottleneck Zone (ε=0.8)'),
        mpatches.Patch(facecolor='#3498db', edgecolor='#3498db', alpha=0.1,
                      linestyle='--', linewidth=2.5, label=f'Origin Zone (SW, n={len(origin_nodes)})'),
        mpatches.Patch(facecolor='#27ae60', edgecolor='#27ae60', alpha=0.1,
                      linestyle='--', linewidth=2.5, label=f'Destination Zone (Geylang, n={len(destination_nodes)})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
                  markersize=12, markeredgecolor='white', markeredgewidth=1.5,
                  label='Origin Nodes', linestyle='None', alpha=0.8),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
                  markersize=12, markeredgecolor='white', markeredgewidth=1.5,
                  label='Destination Nodes', linestyle='None', alpha=0.8),
        mpatches.Patch(facecolor='#2c3e50', edgecolor='black', label='Major Roads', alpha=0.9),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.95,
             title='Localized Flow Zones', title_fontsize=13)

    # Remove axis
    ax.set_axis_off()

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create Paya Lebar scenario map')
    parser.add_argument('--output-dir', default='output/urban_planning',
                       help='Output directory for map')

    args = parser.parse_args()

    # Create Paya Lebar map
    create_paya_lebar_map(
        output_path=f"{args.output_dir}/paya_lebar_scenario_map.png"
    )

    print("\n" + "=" * 80)
    print("SCENARIO MAP COMPLETE")
    print("=" * 80)
    print(f"\nSaved to: {args.output_dir}/")
    print("  - paya_lebar_scenario_map.png")
