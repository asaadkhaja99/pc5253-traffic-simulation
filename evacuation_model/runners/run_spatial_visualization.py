"""
Run Evacuation Scenarios with Spatial Visualization

Executes all three evacuation scenarios with spatial map-based visualization,
capturing screenshots at regular intervals to show evacuation dynamics.

Usage:
    # Quick test (500 agents, 1-minute intervals)
    python run_spatial_visualization.py --quick

    # Full run (2000 agents, 1-minute intervals)
    python run_spatial_visualization.py

    # Custom configuration
    python run_spatial_visualization.py --num-agents 1000 --interval 120 --scenario simultaneous
"""

import argparse
from pathlib import Path
import sys

# Import model and visualization
from ..core.evacuation_base import EvacuationModel, EvacuationConfig, get_origin_nodes, assign_destinations
from ..visualization.visualize_spatial import create_spatial_animation


def spawn_simultaneous(model, num_agents):
    """
    Spawn all agents simultaneously at t=0.

    Args:
        model: EvacuationModel instance
        num_agents: Number of agents to spawn
    """
    origins = get_origin_nodes(model, num_agents)
    destinations = assign_destinations(model, origins)

    spawned = 0
    for origin, dest in zip(origins, destinations):
        evacuee = model.spawn_evacuee(origin, dest)
        if evacuee:
            model.evacuees.append(evacuee)
            model.space.add_agents(evacuee)
            spawned += 1

    print(f"  Spawned {spawned} evacuees at t=0")


def spawn_staged(model, num_agents, num_waves=4, wave_interval=600):
    """
    Spawn agents in staged waves.

    Args:
        model: EvacuationModel instance
        num_agents: Total number of agents
        num_waves: Number of waves
        wave_interval: Time between waves (seconds)
    """
    origins = get_origin_nodes(model, num_agents)
    destinations = assign_destinations(model, origins)

    agents_per_wave = num_agents // num_waves
    wave_spawn_times = []

    spawned = 0
    for wave in range(num_waves):
        wave_start = wave * agents_per_wave
        wave_end = wave_start + agents_per_wave if wave < num_waves - 1 else num_agents
        spawn_time = wave * wave_interval

        for i in range(wave_start, wave_end):
            origin = origins[i]
            dest = destinations[i]

            evacuee = model.spawn_evacuee(origin, dest)
            if evacuee:
                evacuee.spawn_time = spawn_time  # Store when this agent should enter
                model.evacuees.append(evacuee)
                spawned += 1

        wave_spawn_times.append((wave + 1, spawn_time, wave_end - wave_start))

    print(f"  Prepared {spawned} evacuees in {num_waves} waves:")
    for wave_num, spawn_t, count in wave_spawn_times:
        print(f"    Wave {wave_num}: {count} evacuees at t={spawn_t}s ({spawn_t/60:.1f} min)")

    # Mark as staged scenario
    model.is_staged = True
    model.wave_interval = wave_interval


def spawn_contraflow(model, num_agents, num_contraflow_roads=4):
    """
    Spawn agents with contraflow intervention.

    Args:
        model: EvacuationModel instance
        num_agents: Number of agents
        num_contraflow_roads: Number of roads to apply contraflow
    """
    # Identify contraflow candidates
    candidates = []

    for edge_tuple, road_agent in model.road_agents.items():
        highway = road_agent.highway_type

        # Select major roads
        if highway in ['motorway', 'trunk', 'primary']:
            if road_agent.lanes >= 2:
                candidates.append({
                    'edge': edge_tuple,
                    'road': road_agent,
                    'highway': highway,
                    'length': road_agent.length_m,
                    'lanes': road_agent.lanes
                })

    # Score candidates
    highway_scores = {'motorway': 3, 'trunk': 2, 'primary': 1}

    for candidate in candidates:
        highway_score = highway_scores.get(candidate['highway'], 0)
        length_score = candidate['length'] / 1000  # km
        lane_score = candidate['lanes']

        candidate['score'] = highway_score * 10 + length_score + lane_score

    # Select top N
    candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)
    selected_candidates = candidates_sorted[:num_contraflow_roads]

    # Apply contraflow (50% capacity increase)
    contraflow_roads = []
    for candidate in selected_candidates:
        road = candidate['road']
        original_capacity = road.num_cells

        # Increase capacity by 50%
        new_capacity = int(original_capacity * 1.5)
        road.num_cells = new_capacity
        road.road = -np.ones(new_capacity, dtype=np.int16)

        contraflow_roads.append({
            'edge': candidate['edge'],
            'highway': candidate['highway'],
            'length': candidate['length'],
            'original_capacity': original_capacity,
            'new_capacity': new_capacity,
            'capacity_increase': new_capacity - original_capacity
        })

        print(f"    Applied contraflow to {candidate['highway']} road (edge {candidate['edge']}): "
              f"{original_capacity} → {new_capacity} cells (+{new_capacity - original_capacity})")

    # Spawn agents normally
    origins = get_origin_nodes(model, num_agents)
    destinations = assign_destinations(model, origins)

    spawned = 0
    for origin, dest in zip(origins, destinations):
        evacuee = model.spawn_evacuee(origin, dest)
        if evacuee:
            model.evacuees.append(evacuee)
            model.space.add_agents(evacuee)
            spawned += 1

    print(f"  Applied contraflow to {len(contraflow_roads)} major roads")
    print(f"  Spawned {spawned} evacuees at t=0")

    # Store contraflow info
    model.contraflow_roads = contraflow_roads


def run_spatial_visualization_study(
    scenarios=None,
    num_agents=2000,
    screenshot_interval=60,
    max_steps=7200,
    output_dir=None
):
    """
    Run evacuation scenarios with spatial visualization.

    Args:
        scenarios: List of scenarios to run ('simultaneous', 'staged', 'contraflow')
                  If None, runs all scenarios
        num_agents: Number of evacuees
        screenshot_interval: Screenshot capture interval (seconds)
        max_steps: Maximum simulation steps
        output_dir: Output directory for screenshots
    """
    if scenarios is None:
        scenarios = ['simultaneous', 'staged', 'contraflow']

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output" / "evacuation" / "spatial_viz"

    print("=" * 80)
    print("EVACUATION SPATIAL VISUALIZATION STUDY")
    print("=" * 80)
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Agents: {num_agents}")
    print(f"Screenshot interval: {screenshot_interval}s")
    print(f"Max simulation time: {max_steps}s ({max_steps/60:.1f} min)")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    results = {}

    # 1. Simultaneous evacuation
    if 'simultaneous' in scenarios:
        print("\n" + "=" * 80)
        print("[1/3] SIMULTANEOUS EVACUATION")
        print("=" * 80)

        model = create_spatial_animation(
            scenario_name='Simultaneous',
            num_agents=num_agents,
            spawn_function=spawn_simultaneous,
            output_dir=output_dir,
            screenshot_interval=screenshot_interval,
            max_steps=max_steps
        )
        results['simultaneous'] = model

    # 2. Staged evacuation
    if 'staged' in scenarios:
        print("\n" + "=" * 80)
        print("[2/3] STAGED EVACUATION (4 waves, 10-min intervals)")
        print("=" * 80)

        def spawn_staged_wrapper(model, num_agents):
            spawn_staged(model, num_agents, num_waves=4, wave_interval=600)

        model = create_spatial_animation(
            scenario_name='Staged',
            num_agents=num_agents,
            spawn_function=spawn_staged_wrapper,
            output_dir=output_dir,
            screenshot_interval=screenshot_interval,
            max_steps=max_steps
        )
        results['staged'] = model

    # 3. Contraflow intervention
    if 'contraflow' in scenarios:
        print("\n" + "=" * 80)
        print("[3/3] CONTRAFLOW INTERVENTION (4 major roads)")
        print("=" * 80)

        def spawn_contraflow_wrapper(model, num_agents):
            import numpy as np  # Need this for contraflow
            spawn_contraflow(model, num_agents, num_contraflow_roads=4)

        model = create_spatial_animation(
            scenario_name='Contraflow',
            num_agents=num_agents,
            spawn_function=spawn_contraflow_wrapper,
            output_dir=output_dir,
            screenshot_interval=screenshot_interval,
            max_steps=max_steps
        )
        results['contraflow'] = model

    # Print summary
    print("\n" + "=" * 80)
    print("SPATIAL VISUALIZATION STUDY COMPLETE")
    print("=" * 80)
    print(f"\nScreenshots saved to: {output_dir}/")
    print("\nGenerated folders:")
    for scenario in scenarios:
        scenario_dir = output_dir / scenario
        if scenario_dir.exists():
            num_snapshots = len(list(scenario_dir.glob('snapshot_*.png')))
            print(f"  - {scenario}/  ({num_snapshots} snapshots)")

    print("\n" + "=" * 80)
    print("CREATE VIDEOS FROM SCREENSHOTS")
    print("=" * 80)
    print("\nTo create MP4 videos, run these commands:\n")

    for scenario in scenarios:
        scenario_dir = output_dir / scenario
        if scenario_dir.exists():
            print(f"# {scenario.capitalize()} scenario:")
            print(f"cd {scenario_dir}")
            print(f"ffmpeg -framerate 10 -pattern_type glob -i 'snapshot_*.png' \\")
            print(f"       -c:v libx264 -pix_fmt yuv420p {scenario}_evacuation.mp4\n")

    print("=" * 80)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run evacuation scenarios with spatial visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 500 agents
  python run_spatial_visualization.py --quick

  # Full study with all scenarios
  python run_spatial_visualization.py

  # Single scenario with custom settings
  python run_spatial_visualization.py --scenario simultaneous --num-agents 1000 --interval 120

  # High-resolution visualization (30-second intervals)
  python run_spatial_visualization.py --interval 30
        """
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (500 agents, 1 hour sim time)'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        choices=['simultaneous', 'staged', 'contraflow', 'all'],
        default='all',
        help='Scenario to run (default: all)'
    )

    parser.add_argument(
        '--num-agents',
        type=int,
        default=None,
        help='Number of evacuees (default: 2000, or 500 if --quick)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Screenshot interval in seconds (default: 60)'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum simulation steps (default: 7200, or 3600 if --quick)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for screenshots'
    )

    args = parser.parse_args()

    # Apply quick mode defaults
    if args.quick:
        num_agents = args.num_agents if args.num_agents else 500
        max_steps = args.max_steps if args.max_steps else 3600
    else:
        num_agents = args.num_agents if args.num_agents else 2000
        max_steps = args.max_steps if args.max_steps else 7200

    # Determine scenarios to run
    if args.scenario == 'all':
        scenarios = ['simultaneous', 'staged', 'contraflow']
    else:
        scenarios = [args.scenario]

    # Run visualization study
    run_spatial_visualization_study(
        scenarios=scenarios,
        num_agents=num_agents,
        screenshot_interval=args.interval,
        max_steps=max_steps,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
