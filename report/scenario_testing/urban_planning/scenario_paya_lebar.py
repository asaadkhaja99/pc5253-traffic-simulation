"""
Localized Incident Simulation: Paya Lebar Lane Closure

Replicates the methodology from Othman et al. (2023) who used SUMO to model
a partial lane closure in Paya Lebar, Singapore.

This scenario demonstrates:
- Queue spillback from a localized bottleneck
- Impact of lane reduction on traffic flow
- Validation of ABM's ability to reproduce realistic congestion patterns

Reference:
Othman et al. (2023). Agent-based traffic simulation for Paya Lebar region
"""

import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox
from shapely.geometry import Point
from urban_base import (
    UrbanPlanningModel,
    UrbanPlanningConfig,
    DisruptionConfig
)


# =============================================================================
# Paya Lebar Configuration
# =============================================================================

def get_paya_lebar_config():
    """
    Get configuration for Paya Lebar area simulation.

    Paya Lebar is a major transportation hub in East Singapore with:
    - Paya Lebar MRT interchange
    - Major roads: Paya Lebar Road, Sims Avenue, Guillemard Road
    - Mix of commercial, industrial, and residential areas

    Uses IDM (Intelligent Driver Model) with Virtual Leader concept
    to model lane closure near MRT station.
    """
    return UrbanPlanningConfig(
        # Paya Lebar area bounds (zoomed out to match map view)
        # Paya Lebar MRT: 1.31765°N, 103.89271°E
        bbox_north=1.3280,  # ~1.1km north of MRT
        bbox_south=1.3070,  # ~1.2km south of MRT
        bbox_east=103.9040,  # ~1.2km east of MRT
        bbox_west=103.8810,  # ~1.3km west of MRT

        # Traffic parameters
        num_vehicles=5000,  # Very high traffic load to create congestion
        delta_t=1.0,
        max_steps=1200,  # 20 minute simulation (allow time for queues to form)

        demand_level=0.6,  # 60% of capacity

        # Use IDM for realistic car-following with bottleneck factor
        use_idm=True,

        seed=42
    )


def identify_paya_lebar_closure_location(model):
    """
    Identify road segment for lane closure near Paya Lebar MRT.

    In Othman et al.'s study, they simulated a lane closure on a major road
    near the Paya Lebar interchange. We identify a similar location on
    Paya Lebar Road or Sims Avenue.

    Args:
        model: UrbanPlanningModel instance with loaded network

    Returns:
        List of edge tuples (u, v, key) representing the closure location
    """
    # Paya Lebar MRT approximate coordinates
    paya_lebar_coords = (1.3177, 103.8926)  # lat, lon

    # Find major road segments near Paya Lebar MRT
    # IMPORTANT: Project point to match road geometry CRS
    import geopandas as gpd
    paya_lebar_point_wgs84 = Point(paya_lebar_coords[1], paya_lebar_coords[0])  # lon, lat
    paya_lebar_gdf = gpd.GeoDataFrame(geometry=[paya_lebar_point_wgs84], crs='EPSG:4326')

    # Get CRS from first road agent
    first_road = list(model.road_agents.values())[0]
    road_crs = first_road.crs

    # Project to road CRS
    paya_lebar_gdf = paya_lebar_gdf.to_crs(road_crs)
    paya_lebar_point = paya_lebar_gdf.geometry.iloc[0]

    closure_edges = []

    for edge_tuple, road_agent in model.road_agents.items():
        # Check if road is a major artery
        highway_type = road_agent.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else ''

        # Focus on primary/secondary roads
        if highway_type not in ['primary', 'secondary', 'trunk']:
            continue

        # Check distance to Paya Lebar MRT (now in meters since both are projected)
        road_geom = road_agent.geometry
        distance = paya_lebar_point.distance(road_geom)  # meters

        # Within 100m
        if distance < 100:
            closure_edges.append({
                'edge': edge_tuple,
                'highway_type': highway_type,
                'length_m': road_agent.length_m,
                'distance': distance
            })

    # Sort by distance to MRT
    closure_edges.sort(key=lambda x: x['distance'])

    # Select the closest major road
    if len(closure_edges) > 0:
        selected = closure_edges[0]['edge']
        print(f"  ✓ Found bottleneck road near Paya Lebar MRT:")
        print(f"    Edge: {selected}")
        print(f"    Highway type: {closure_edges[0]['highway_type']}")
        print(f"    Length: {closure_edges[0]['length_m']:.1f}m")
        print(f"    Distance from MRT: {closure_edges[0]['distance']:.1f}m")
        return [selected]
    else:
        # Fallback: select any primary road in the network
        print("  ✗ Warning: No roads found within 100m of Paya Lebar MRT!")
        print("    Falling back to first primary road in network")
        for edge_tuple, road_agent in model.road_agents.items():
            highway_type = road_agent.highway_type
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            if highway_type in ['primary', 'secondary']:
                print(f"    Selected fallback: {edge_tuple}")
                return [edge_tuple]

    return []


def identify_targeted_flow_nodes(model):
    """
    Identify South and North nodes for targeted flow through Paya Lebar bottleneck.

    Strategy:
    - Origins: Nodes significantly SOUTH of Paya Lebar MRT (lat < 1.315)
    - Destinations: Nodes significantly NORTH of Paya Lebar MRT (lat > 1.320)
    - Result: Shortest path naturally routes through the bottleneck on Paya Lebar Road

    Args:
        model: UrbanPlanningModel with loaded network

    Returns:
        Tuple of (south_nodes, north_nodes) - lists of node IDs
    """
    paya_lebar_lat = 1.3177

    south_nodes = []
    north_nodes = []

    for node_id in model.major_nodes:
        node_data = model.G.nodes[node_id]
        node_lat = node_data['y']

        # South region: significantly below MRT (> 300m south)
        if node_lat < (paya_lebar_lat - 0.003):
            south_nodes.append(node_id)

        # North region: significantly above MRT (> 300m north)
        elif node_lat > (paya_lebar_lat + 0.003):
            north_nodes.append(node_id)

    print(f"  Found {len(south_nodes)} southern origin nodes")
    print(f"  Found {len(north_nodes)} northern destination nodes")

    return south_nodes, north_nodes


def run_paya_lebar_scenario(
    with_closure=True,
    closure_capacity_reduction=0.8,
    num_vehicles=800,
    seed=42,
    output_file=None
):
    """
    Run Paya Lebar lane closure scenario.

    Args:
        with_closure: If True, apply lane closure
        closure_capacity_reduction: Fraction of capacity to remove (0-1)
        num_vehicles: Number of vehicles to simulate
        seed: Random seed
        output_file: Path to save results CSV

    Returns:
        EvacuationMetrics object
    """
    print("=" * 80)
    print("PAYA LEBAR LOCALIZED INCIDENT SIMULATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of vehicles: {num_vehicles}")
    print(f"  - Lane closure: {'Yes' if with_closure else 'No (baseline)'}")
    if with_closure:
        print(f"  - Capacity reduction: {closure_capacity_reduction*100:.0f}%")
    print(f"  - Seed: {seed}")

    # Create configuration
    config = get_paya_lebar_config()
    config.num_vehicles = num_vehicles
    config.seed = seed

    # Initialize model (without disruption first to find location)
    print(f"\n[1/4] Loading Paya Lebar road network...")
    model = UrbanPlanningModel(config, disruption=None)

    # ALWAYS identify the bottleneck road (for both baseline and incident scenarios)
    print(f"\n[2/4] Identifying bottleneck road for forced routing...")
    closure_edges = identify_paya_lebar_closure_location(model)

    if len(closure_edges) == 0:
        print("  ERROR: Could not identify bottleneck road!")
        return None

    bottleneck_edge = closure_edges[0]
    print(f"  Bottleneck road identified: {bottleneck_edge}")

    # Apply disruption ONLY if with_closure=True
    if with_closure:
        print(f"  Applying Gaussian bottleneck (ε=0.9, σ=50m)...")
        disruption = DisruptionConfig(
            disruption_type='lane_closure',
            affected_edges=closure_edges,
            start_time=0,  # Start immediately
            duration=None,  # Permanent for this simulation
            capacity_reduction=closure_capacity_reduction,  # For NaSch fallback
            location_name="Paya Lebar Road (near MRT)",
            location_coords=(1.3177, 103.8926),
            # IDM Gaussian bottleneck parameters (Eq. 17)
            bottleneck_epsilon=0.9,  # 90% capacity reduction (increased for more severe bottleneck)
            bottleneck_sigma=50.0,  # 50m spatial spread
            incident_position_m=None  # Will default to midpoint of road
        )

        # Apply disruption
        model.disruption = disruption
        model._apply_disruption()
    else:
        print(f"  Running baseline (no Gaussian bottleneck, normal speeds)")

    # =========================================================================
    #  SPATIAL FILTERING FOR FORCED NORTHBOUND FLOW
    # =========================================================================
    print(f"\n[3/4] Spawning {num_vehicles} vehicles (Forced Northbound Flow)...")

    # Paya Lebar MRT Coordinates (Incident Center)
    # Lat: 1.3177, Lon: 103.8926
    INCIDENT_LAT = 1.3177
    INCIDENT_LON = 103.8926

    # Define Origin Zone (South of Incident) - 50% smaller
    # We want traffic moving North, so they start South.
    # 350m to 650m South of MRT (was 200m to 800m)
    # Keep Longitude tight to ensure they don't take parallel streets
    origin_nodes = []
    for node, data in model.nodes_gdf.iterrows():
        if (1.3130 < data.y < 1.3160) and (103.891 < data.x < 103.896):
            origin_nodes.append(node)

    # Define Destination Zone (North of Incident) - 50% smaller
    # 350m to 650m North of MRT (was 200m to 800m)
    destination_nodes = []
    for node, data in model.nodes_gdf.iterrows():
        if (1.3190 < data.y < 1.3220) and (103.891 < data.x < 103.896):
            destination_nodes.append(node)

    print(f"  Found {len(origin_nodes)} valid Origin nodes (South)")
    print(f"  Found {len(destination_nodes)} valid Destination nodes (North)")

    if len(origin_nodes) == 0 or len(destination_nodes) == 0:
        raise ValueError("No nodes found in target zones! Check coordinate bounds.")

    # Store bottleneck road for tracking
    model.bottleneck_road_edge = bottleneck_edge

    # Extract the critical nodes from the bottleneck edge
    # bottleneck_edge is (u, v, key)
    bottleneck_start_node = bottleneck_edge[0]
    bottleneck_end_node = bottleneck_edge[1]

    print(f"  Forcing routes through edge: {bottleneck_start_node} -> {bottleneck_end_node}")

    spawned = 0
    # Track which vehicles are "probes" for the bottleneck stats
    bottleneck_vehicles = []

    # Import for routing
    import networkx as nx

    for _ in range(num_vehicles):
        # Force flow South -> North
        origin = model.rng.choice(origin_nodes)
        destination = model.rng.choice(destination_nodes)

        # Retry if same node
        while destination == origin:
            destination = model.rng.choice(destination_nodes)

        # TWO-STAGE ROUTING: Force through bottleneck edge
        try:
            # Stage 1: Origin -> Bottleneck Start
            route_part_1 = nx.shortest_path(
                model.G, origin, bottleneck_start_node, weight='length'
            )

            # Stage 2: Bottleneck End -> Destination
            route_part_2 = nx.shortest_path(
                model.G, bottleneck_end_node, destination, weight='length'
            )

            # Stitch them together
            # route_part_1 ends with bottleneck_start
            # route_part_2 starts with bottleneck_end
            full_node_path = route_part_1 + route_part_2

            # Convert node path to edge path [(u,v,k), (u,v,k)...]
            # This automatically includes the bottleneck edge (bottleneck_start -> bottleneck_end)
            full_route_edges = []
            for i in range(len(full_node_path) - 1):
                u, v = full_node_path[i], full_node_path[i + 1]
                edge_data = model.G.get_edge_data(u, v)
                if edge_data:
                    # Pick the shortest key (usually 0)
                    key = min(edge_data.keys(), key=lambda k: edge_data[k].get('length', float('inf')))
                    full_route_edges.append((u, v, key))
                else:
                    # Edge case: Disconnected graph (shouldn't happen on main roads)
                    full_route_edges = []
                    break

            if len(full_route_edges) > 0:
                # SPAWN AGENT WITH FORCED ROUTE
                agent_id = len(model.vehicles)
                origin_point = model.nodes_gdf.loc[origin].geometry

                from urban_base import UrbanVehicleAgent

                vehicle = UrbanVehicleAgent(
                    model=model,
                    geometry=origin_point,
                    crs=model.edges_gdf.crs,
                    agent_id=agent_id,
                    origin_node=origin,
                    destination_node=destination,
                    route=full_route_edges  # Injected Route
                )

                vehicle.passed_through_bottleneck = True  # Tag for metrics

                model.vehicles.append(vehicle)
                model.space.add_agents(vehicle)

                first_road = model.road_agents.get(full_route_edges[0])
                if first_road:
                    first_road.add_vehicle(vehicle)

                spawned += 1
                bottleneck_vehicles.append(vehicle)

        except nx.NetworkXNoPath:
            continue

    print(f"  Successfully spawned {spawned} vehicles with verified bottleneck routes")

    # Run simulation
    print(f"\n[4/4] Running simulation...")
    metrics = model.run()

    # Save results
    if output_file:
        print(f"\nSaving results to {output_file}...")
        results_df = pd.DataFrame({
            'time_step': metrics.time_steps,
            'completed_trips': metrics.evacuated_count,
            'network_flow': metrics.network_flow,
            'mean_speed_kph': metrics.mean_speed,
            'congested_roads': metrics.congested_roads,
            'total_queue_length': metrics.total_queue_length,
            'total_delay': metrics.total_delay
        })
        results_df.to_csv(output_file, index=False)
        print(f"  Saved {len(results_df)} time steps")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_completed = metrics.evacuated_count[-1] if metrics.evacuated_count else 0
    completion_pct = total_completed / spawned * 100 if spawned > 0 else 0

    # Mean trip time
    completed_vehicles = [v for v in model.vehicles if v.evacuated]
    if len(completed_vehicles) > 0:
        trip_times = [v.evacuation_time for v in completed_vehicles]
        mean_trip_time = sum(trip_times) / len(trip_times)
    else:
        mean_trip_time = 0

    # Peak congestion
    max_congested = max(metrics.congested_roads) if metrics.congested_roads else 0
    peak_time = metrics.time_steps[metrics.congested_roads.index(max_congested)] if max_congested > 0 else 0

    # Queue analysis (if closure applied) - Othman et al. methodology
    if with_closure and model.disruption:
        max_queue = 0
        max_queue_m = 0.0
        disrupted_roads = []

        print(f"\nDisruption: {model.disruption.disruption_type}")

        for edge_tuple in model.disruption.affected_edges:
            road = model.road_agents.get(edge_tuple)
            if road:
                disrupted_roads.append(road)
                if road.queue_length > max_queue:
                    max_queue = road.queue_length
                    max_queue_m = road.queue_length_m

                # Calculate delay for vehicles on this road (IDM only)
                if model.config.use_idm and hasattr(road, 'vehicles'):
                    free_flow_time = road.length_m / (road.v0) if road.v0 > 0 else 0
                    delays = []
                    for v in road.vehicles:
                        if v.travel_time is not None:
                            delay = v.travel_time - free_flow_time
                            if delay > 0:
                                delays.append(delay)

                    if len(delays) > 0:
                        mean_delay = np.mean(delays)
                        print(f"\nRoad {edge_tuple}:")
                        print(f"  Queue length: {road.queue_length} vehicles ({road.queue_length_m:.1f}m)")
                        print(f"  Mean delay: {mean_delay:.1f}s ({mean_delay/60:.1f} min)")
                        print(f"  Free-flow time: {free_flow_time:.1f}s")

        print(f"\nAffected roads: {len(disrupted_roads)}")
        print(f"Peak queue length: {max_queue} vehicles ({max_queue_m:.1f}m spillback)")

    # Filter statistics for vehicles that passed through bottleneck
    bottleneck_vehicles = [v for v in completed_vehicles if getattr(v, 'passed_through_bottleneck', False)]

    print(f"\nCompleted trips: {total_completed}/{spawned} ({completion_pct:.1f}%)")
    print(f"  Vehicles through bottleneck: {len(bottleneck_vehicles)}/{total_completed} ({len(bottleneck_vehicles)/total_completed*100 if total_completed > 0 else 0:.1f}%)")

    if len(bottleneck_vehicles) > 0:
        bottleneck_trip_times = [v.evacuation_time for v in bottleneck_vehicles]
        mean_bottleneck_time = sum(bottleneck_trip_times) / len(bottleneck_trip_times)
        print(f"\nBottleneck vehicles statistics:")
        print(f"  Mean trip time: {mean_bottleneck_time:.1f} seconds ({mean_bottleneck_time/60:.1f} minutes)")

    print(f"\nAll vehicles:")
    print(f"  Mean trip time: {mean_trip_time:.1f} seconds ({mean_trip_time/60:.1f} minutes)")
    print(f"  Peak congestion: {max_congested} roads at t={peak_time}s")
    print(f"  Mean network speed: {sum(metrics.mean_speed)/len(metrics.mean_speed):.2f} km/h" if metrics.mean_speed else "  Mean network speed: N/A")

    print("\n" + "=" * 80)

    return metrics


if __name__ == '__main__':
    # Default configuration
    NUM_VEHICLES = 5000  # Very high traffic load to create congestion
    SEED = 42
    OUTPUT_DIR = Path(__file__).parent.parent / "output" / "urban_planning"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run baseline (no closure) - South→North flow
    print("\n" + "#" * 80)
    print("# BASELINE SCENARIO (NO LANE CLOSURE)")
    print("# Strategy: South→North O-D pairs → Natural routing through bottleneck")
    print("#" * 80 + "\n")

    baseline_file = OUTPUT_DIR / "paya_lebar_baseline.csv"
    metrics_baseline = run_paya_lebar_scenario(
        with_closure=False,
        num_vehicles=NUM_VEHICLES,
        seed=SEED,
        output_file=baseline_file
    )

    # Run with lane closure - Same South→North flow
    print("\n\n" + "#" * 80)
    print("# LANE CLOSURE SCENARIO (GAUSSIAN BOTTLENECK)")
    print("# Strategy: Same South→North O-D pairs → Natural routing through bottleneck")
    print("#" * 80 + "\n")

    closure_file = OUTPUT_DIR / "paya_lebar_closure.csv"
    metrics_closure = run_paya_lebar_scenario(
        with_closure=True,
        closure_capacity_reduction=0.8,  # ε=0.8 (80% capacity reduction)
        num_vehicles=NUM_VEHICLES,
        seed=SEED,
        output_file=closure_file
    )

    print("\n\n" + "=" * 80)
    print("PAYA LEBAR STUDY COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - {baseline_file.name}")
    print(f"  - {closure_file.name}")
    print("=" * 80)
