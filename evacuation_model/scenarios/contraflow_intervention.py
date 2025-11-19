"""
Contraflow Lane Reversal Intervention

Implements contraflow operations on major arteries to increase outbound
capacity during evacuation. Reverses lanes on selected roads to create
additional evacuation routes.

This intervention tests how infrastructure modifications can improve
evacuation efficiency and reduce total evacuation time.

Contraflow strategy:
- Identify major outbound arteries (motorway, trunk, primary)
- Select roads with >2 lanes
- Increase capacity by ~50% (simulating reversed lanes)
- Activate at t=0 or when congestion threshold reached
"""

import pandas as pd
from pathlib import Path
import numpy as np
from ..core.evacuation_base import (
    EvacuationModel,
    EvacuationConfig,
    get_origin_nodes,
    assign_destinations
)


def identify_contraflow_candidates(model, evacuation_center, min_lanes=2):
    """
    Identify roads suitable for contraflow operations.

    Criteria:
    - Major highways (motorway, trunk, primary)
    - At least min_lanes lanes
    - Leading away from evacuation zone (outbound)

    Args:
        model: EvacuationModel instance
        evacuation_center: (lat, lon) tuple of evacuation zone center
        min_lanes: Minimum number of lanes required

    Returns:
        List of edge tuples (u, v, key) for contraflow roads
    """
    from shapely.geometry import Point

    center_point = Point(evacuation_center[1], evacuation_center[0])  # lon, lat
    candidates = []

    for edge_tuple, road_agent in model.road_agents.items():
        # Check highway type
        highway_type = road_agent.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else ''

        if highway_type not in ['motorway', 'trunk', 'primary']:
            continue

        # Check number of lanes
        if road_agent.num_lanes < min_lanes:
            continue

        # Check if road leads away from evacuation center
        # Use start and end points of road geometry
        coords = list(road_agent.geometry.coords)
        start_point = Point(coords[0])
        end_point = Point(coords[-1])

        # Distance from center
        start_dist = center_point.distance(start_point)
        end_dist = center_point.distance(end_point)

        # Outbound if end is farther from center than start
        if end_dist > start_dist:
            candidates.append({
                'edge': edge_tuple,
                'highway_type': highway_type,
                'lanes': road_agent.num_lanes,
                'length_m': road_agent.length_m,
                'road_agent': road_agent
            })

    return candidates


def select_contraflow_roads(candidates, max_roads=5):
    """
    Select top roads for contraflow based on importance.

    Criteria:
    - Highway class (motorway > trunk > primary)
    - Road length (longer roads prioritized)
    - Number of lanes (more lanes prioritized)

    Args:
        candidates: List of candidate dicts from identify_contraflow_candidates
        max_roads: Maximum number of roads to select

    Returns:
        List of selected candidate dicts
    """
    # Score each candidate
    highway_scores = {'motorway': 3, 'trunk': 2, 'primary': 1}

    for candidate in candidates:
        highway_score = highway_scores.get(candidate['highway_type'], 0)
        length_score = candidate['length_m'] / 1000  # km
        lane_score = candidate['lanes']

        # Weighted score
        candidate['score'] = highway_score * 3 + length_score + lane_score

    # Sort by score descending
    candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)

    # Select top max_roads
    selected = candidates_sorted[:max_roads]

    return selected


def run_contraflow_evacuation(
    num_agents=2000,
    num_contraflow_roads=4,
    contraflow_activation_time=0,
    seed=42,
    output_file=None
):
    """
    Run simultaneous evacuation with contraflow intervention.

    Args:
        num_agents: Number of evacuees
        num_contraflow_roads: Number of roads to apply contraflow
        contraflow_activation_time: Time to activate contraflow (seconds)
        seed: Random seed
        output_file: Path to save results CSV (optional)

    Returns:
        Tuple (metrics, contraflow_roads_info)
    """
    print("=" * 80)
    print("CONTRAFLOW INTERVENTION SCENARIO")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of agents: {num_agents}")
    print(f"  - Contraflow roads: {num_contraflow_roads}")
    print(f"  - Activation time: {contraflow_activation_time}s")
    print(f"  - Seed: {seed}")

    # Create configuration
    config = EvacuationConfig(
        num_agents=num_agents,
        seed=seed
    )

    # Initialize model
    model = EvacuationModel(config)

    # Enable contraflow on ALL eligible roads (roads with >1 lane)
    print(f"\n[1/4] Enabling contraflow on all eligible roads...")
    contraflow_roads_info = []
    contraflow_count = 0

    for edge_tuple, road_agent in model.road_agents.items():
        # Enable contraflow on all roads with multiple lanes
        if road_agent.num_lanes > 1:
            old_capacity = road_agent.num_cells
            road_agent.enable_contraflow()
            new_capacity = road_agent.num_cells

            if road_agent.contraflow_enabled:  # Successfully enabled
                contraflow_count += 1
                contraflow_roads_info.append({
                    'edge_u': edge_tuple[0],
                    'edge_v': edge_tuple[1],
                    'edge_key': edge_tuple[2],
                    'highway_type': road_agent.highway_type,
                    'lanes': road_agent.num_lanes,
                    'length_m': road_agent.length_m,
                    'base_capacity': road_agent.base_num_cells,
                    'final_capacity': new_capacity
                })

    print(f"  Enabled contraflow on {contraflow_count} roads (all roads with >1 lane)")

    # Show breakdown by highway type
    from collections import Counter
    highway_counts = Counter([info['highway_type'] for info in contraflow_roads_info])
    print(f"  Breakdown by highway type:")
    for hwy_type, count in sorted(highway_counts.items(), key=lambda x: -x[1]):
        print(f"    - {hwy_type}: {count} roads")

    # Spawn all agents at t=0 (after contraflow is enabled)
    print(f"\n[3/4] Spawning {num_agents} evacuees...")
    origin_nodes = get_origin_nodes(model, num_agents)
    od_pairs = assign_destinations(model, origin_nodes)

    spawned = 0
    for origin, destination in od_pairs:
        evacuee = model.spawn_evacuee(origin, destination)
        if evacuee:
            spawned += 1

    print(f"  Successfully spawned {spawned} evacuees")

    # Run simulation
    print(f"\n[4/4] Running simulation...")
    max_steps = config.max_steps

    while model.step_count < max_steps:

        # Step simulation
        model.step()

        # Progress reporting
        evacuated_count = sum(1 for e in model.evacuees if e.evacuated)
        evacuation_pct = evacuated_count / spawned * 100 if spawned > 0 else 0

        if model.step_count % 300 == 0:  # Every 5 minutes
            print(f"  Step {model.step_count}/{max_steps} - "
                  f"Evacuated: {evacuated_count}/{spawned} ({evacuation_pct:.1f}%) - "
                  f"Congested roads: {model.metrics.congested_roads[-1] if model.metrics.congested_roads else 0}")

        # Stop if 95% evacuated
        if evacuation_pct >= 95.0:
            print(f"  95% evacuation threshold reached at step {model.step_count}")
            break

    metrics = model.metrics

    # Update contraflow roads info with final capacity
    for i, road_info_dict in enumerate(contraflow_roads_info):
        edge = (road_info_dict['edge_u'], road_info_dict['edge_v'], road_info_dict['edge_key'])
        road_agent = model.road_agents.get(edge)
        if road_agent:
            contraflow_roads_info[i]['final_capacity'] = road_agent.num_cells
            contraflow_roads_info[i]['throughput'] = road_agent.throughput

    # Save results
    if output_file:
        print(f"\nSaving results to {output_file}...")
        results_df = pd.DataFrame({
            'time_step': metrics.time_steps,
            'evacuated_count': metrics.evacuated_count,
            'network_flow': metrics.network_flow,
            'mean_speed_kph': metrics.mean_speed,
            'congested_roads': metrics.congested_roads
        })
        results_df.to_csv(output_file, index=False)

        # Save contraflow roads info
        contraflow_file = output_file.parent / (output_file.stem + "_contraflow_roads.csv")
        contraflow_df = pd.DataFrame(contraflow_roads_info)
        contraflow_df.to_csv(contraflow_file, index=False)

        print(f"  Saved {len(results_df)} time steps")
        print(f"  Saved contraflow roads info to {contraflow_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_evacuated = metrics.evacuated_count[-1] if metrics.evacuated_count else 0
    evacuation_pct = total_evacuated / spawned * 100 if spawned > 0 else 0

    # Find T95
    t95 = None
    for i, count in enumerate(metrics.evacuated_count):
        if count >= 0.95 * spawned:
            t95 = metrics.time_steps[i]
            break

    # Mean evacuation time
    evacuated_agents = [e for e in model.evacuees if e.evacuated]
    if evacuated_agents:
        evac_times = [e.evacuation_time for e in evacuated_agents]
        mean_evac_time = sum(evac_times) / len(evac_times)
    else:
        mean_evac_time = 0

    # Peak congestion
    max_congested = max(metrics.congested_roads) if metrics.congested_roads else 0
    peak_congestion_time = metrics.time_steps[metrics.congested_roads.index(max_congested)] if max_congested > 0 else 0

    # Contraflow road throughput
    total_contraflow_throughput = sum(info['throughput'] for info in contraflow_roads_info if 'throughput' in info)

    print(f"Contraflow roads: {len(selected_roads)}")
    print(f"Total contraflow throughput: {total_contraflow_throughput} vehicles")
    print(f"\nTotal evacuated: {total_evacuated}/{spawned} ({evacuation_pct:.1f}%)")
    print(f"Time to 95% evacuation (T95): {t95} seconds ({t95/60:.1f} minutes)" if t95 else "Time to 95% evacuation: Not reached")
    print(f"Mean evacuation time: {mean_evac_time:.1f} seconds ({mean_evac_time/60:.1f} minutes)")
    print(f"Peak congestion: {max_congested} roads at t={peak_congestion_time}s")
    print(f"Mean network speed: {sum(metrics.mean_speed)/len(metrics.mean_speed):.2f} km/h" if metrics.mean_speed else "Mean network speed: N/A")

    print("\n" + "=" * 80)

    return metrics, contraflow_roads_info


if __name__ == '__main__':
    # Default configuration
    NUM_AGENTS = 2000
    NUM_CONTRAFLOW_ROADS = 4
    CONTRAFLOW_ACTIVATION_TIME = 0  # Activate immediately
    SEED = 42
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "evacuation" / "data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "contraflow_evacuation.csv"

    # Run scenario
    metrics, contraflow_info = run_contraflow_evacuation(
        num_agents=NUM_AGENTS,
        num_contraflow_roads=NUM_CONTRAFLOW_ROADS,
        contraflow_activation_time=CONTRAFLOW_ACTIVATION_TIME,
        seed=SEED,
        output_file=OUTPUT_FILE
    )
