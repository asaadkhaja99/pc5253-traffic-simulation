"""
Simultaneous Evacuation Scenario

All evacuees depart at t=0, creating immediate network-wide demand.
This represents an uncoordinated or emergency evacuation where everyone
leaves at the same time.

Expected outcome:
- High initial congestion
- Network gridlock at bottlenecks
- Longer total evacuation time
- Identifies critical network vulnerabilities
"""

import pandas as pd
from pathlib import Path
from ..core.evacuation_base import (
    EvacuationModel,
    EvacuationConfig,
    get_origin_nodes,
    assign_destinations
)


def run_simultaneous_evacuation(
    num_agents=2000,
    seed=42,
    output_file=None
):
    """
    Run simultaneous evacuation scenario.

    Args:
        num_agents: Number of evacuees
        seed: Random seed
        output_file: Path to save results CSV (optional)

    Returns:
        EvacuationMetrics object
    """
    print("=" * 80)
    print("SIMULTANEOUS EVACUATION SCENARIO")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of agents: {num_agents}")
    print(f"  - Seed: {seed}")
    print(f"  - Departure strategy: All at t=0")

    # Create configuration
    config = EvacuationConfig(
        num_agents=num_agents,
        seed=seed
    )

    # Initialize model
    model = EvacuationModel(config)

    # Spawn all agents at t=0
    # Keep trying until we spawn exactly the requested number
    print(f"\n[1/3] Spawning {num_agents} evacuees...")
    spawned = 0
    attempts = 0
    max_attempts = num_agents * 5  # Try up to 5x to avoid infinite loops

    while spawned < num_agents and attempts < max_attempts:
        # Get a batch of origin-destination pairs
        batch_size = min((num_agents - spawned) * 2, 500)  # 2x what we need, max 500 at a time
        origin_nodes = get_origin_nodes(model, batch_size)
        od_pairs = assign_destinations(model, origin_nodes)

        for origin, destination in od_pairs:
            if spawned >= num_agents:
                break

            evacuee = model.spawn_evacuee(origin, destination)
            if evacuee:
                spawned += 1

            attempts += 1
            if attempts >= max_attempts:
                break

    if spawned < num_agents:
        print(f"  WARNING: Only spawned {spawned}/{num_agents} evacuees after {attempts} attempts")
        print(f"  Some origin nodes may be isolated from the road network")
    else:
        print(f"  Successfully spawned {spawned} evacuees (attempts: {attempts})")

    # Run simulation
    print(f"\n[2/3] Running simulation...")
    metrics = model.run()

    # Save results
    if output_file:
        print(f"\n[3/3] Saving results to {output_file}...")
        results_df = pd.DataFrame({
            'time_step': metrics.time_steps,
            'evacuated_count': metrics.evacuated_count,
            'network_flow': metrics.network_flow,
            'mean_speed_kph': metrics.mean_speed,
            'congested_roads': metrics.congested_roads
        })
        results_df.to_csv(output_file, index=False)
        print(f"  Saved {len(results_df)} time steps")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_evacuated = metrics.evacuated_count[-1] if metrics.evacuated_count else 0
    evacuation_pct = total_evacuated / spawned * 100 if spawned > 0 else 0

    # Find T95 (time to 95% evacuation)
    t95 = None
    for i, count in enumerate(metrics.evacuated_count):
        if count >= 0.95 * spawned:
            t95 = metrics.time_steps[i]
            break

    # Mean evacuation time (for evacuated agents only)
    evacuated_agents = [e for e in model.evacuees if e.evacuated]
    if evacuated_agents:
        evac_times = [e.evacuation_time for e in evacuated_agents]
        mean_evac_time = sum(evac_times) / len(evac_times)
    else:
        mean_evac_time = 0

    # Peak congestion
    max_congested = max(metrics.congested_roads) if metrics.congested_roads else 0
    peak_congestion_time = metrics.time_steps[metrics.congested_roads.index(max_congested)] if max_congested > 0 else 0

    print(f"Total evacuated: {total_evacuated}/{spawned} ({evacuation_pct:.1f}%)")
    print(f"Time to 95% evacuation (T95): {t95} seconds ({t95/60:.1f} minutes)" if t95 else "Time to 95% evacuation: Not reached")
    print(f"Mean evacuation time: {mean_evac_time:.1f} seconds ({mean_evac_time/60:.1f} minutes)")
    print(f"Peak congestion: {max_congested} roads at t={peak_congestion_time}s")
    print(f"Mean network speed: {sum(metrics.mean_speed)/len(metrics.mean_speed):.2f} km/h" if metrics.mean_speed else "Mean network speed: N/A")

    print("\n" + "=" * 80)

    return metrics


if __name__ == '__main__':
    # Default configuration
    NUM_AGENTS = 2000
    SEED = 42
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "evacuation" / "data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "simultaneous_evacuation.csv"

    # Run scenario
    metrics = run_simultaneous_evacuation(
        num_agents=NUM_AGENTS,
        seed=SEED,
        output_file=OUTPUT_FILE
    )
