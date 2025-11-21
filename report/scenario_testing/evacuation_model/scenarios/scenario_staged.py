"""
Staged Evacuation Scenario

Evacuees depart in waves at fixed time intervals, reducing instantaneous
network demand and potentially preventing gridlock.

This represents a coordinated evacuation strategy where authorities
control departure timing by zones or phases.

Expected outcome:
- Lower peak congestion
- Better network utilization
- Potentially longer total evacuation time (last agent takes longer)
- Reduced gridlock risk
"""

import pandas as pd
from pathlib import Path
from ..core.evacuation_base import (
    EvacuationModel,
    EvacuationConfig,
    get_origin_nodes,
    assign_destinations
)


def run_staged_evacuation(
    num_agents=2000,
    num_waves=4,
    wave_interval_seconds=600,
    seed=42,
    output_file=None
):
    """
    Run staged evacuation scenario.

    Args:
        num_agents: Total number of evacuees
        num_waves: Number of departure waves
        wave_interval_seconds: Time between waves (seconds)
        seed: Random seed
        output_file: Path to save results CSV (optional)

    Returns:
        EvacuationMetrics object
    """
    print("=" * 80)
    print("STAGED EVACUATION SCENARIO")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Number of agents: {num_agents}")
    print(f"  - Number of waves: {num_waves}")
    print(f"  - Wave interval: {wave_interval_seconds}s ({wave_interval_seconds/60:.1f} min)")
    print(f"  - Seed: {seed}")

    # Create configuration
    config = EvacuationConfig(
        num_agents=num_agents,
        seed=seed
    )

    # Initialize model
    model = EvacuationModel(config)

    # Prepare waves
    print(f"\n[1/3] Preparing {num_waves} evacuation waves...")
    origin_nodes = get_origin_nodes(model, num_agents)
    od_pairs = assign_destinations(model, origin_nodes)

    # Divide agents into waves
    agents_per_wave = num_agents // num_waves
    waves = []

    for wave_idx in range(num_waves):
        start_idx = wave_idx * agents_per_wave
        end_idx = start_idx + agents_per_wave if wave_idx < num_waves - 1 else num_agents

        wave_od_pairs = od_pairs[start_idx:end_idx]
        departure_time = wave_idx * wave_interval_seconds

        waves.append({
            'wave_id': wave_idx,
            'departure_time': departure_time,
            'od_pairs': wave_od_pairs,
            'spawned_count': 0
        })

        print(f"  Wave {wave_idx}: {len(wave_od_pairs)} agents departing at t={departure_time}s")

    # Run simulation with staged spawning
    print(f"\n[2/3] Running simulation with staged departures...")
    max_steps = config.max_steps

    spawned_total = 0
    current_wave = 0

    while model.step_count < max_steps:
        # Check if it's time to spawn next wave
        if current_wave < len(waves):
            wave = waves[current_wave]
            if model.step_count >= wave['departure_time']:
                print(f"  Spawning wave {wave['wave_id']} at t={model.step_count}s...")
                for origin, destination in wave['od_pairs']:
                    evacuee = model.spawn_evacuee(origin, destination)
                    if evacuee:
                        wave['spawned_count'] += 1
                        spawned_total += 1
                print(f"    Spawned {wave['spawned_count']} evacuees")
                current_wave += 1

        # Step simulation
        model.step()

        # Progress reporting
        evacuated_count = sum(1 for e in model.evacuees if e.evacuated)
        evacuation_pct = evacuated_count / num_agents * 100 if num_agents > 0 else 0

        if model.step_count % 300 == 0:  # Every 5 minutes
            print(f"  Step {model.step_count}/{max_steps} - "
                  f"Spawned: {spawned_total}/{num_agents} - "
                  f"Evacuated: {evacuated_count}/{spawned_total} ({evacuation_pct:.1f}%) - "
                  f"Congested roads: {model.metrics.congested_roads[-1] if model.metrics.congested_roads else 0}")

        # Stop if 95% of spawned agents evacuated AND all waves spawned
        if current_wave >= len(waves) and spawned_total > 0:
            if evacuated_count >= 0.95 * spawned_total:
                print(f"  95% evacuation threshold reached at step {model.step_count}")
                break

    metrics = model.metrics

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
    evacuation_pct = total_evacuated / spawned_total * 100 if spawned_total > 0 else 0

    # Find T95 (time to 95% evacuation of spawned agents)
    t95 = None
    for i, count in enumerate(metrics.evacuated_count):
        if count >= 0.95 * spawned_total:
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

    # Wave-specific stats
    print("Wave-specific statistics:")
    for wave in waves:
        print(f"  Wave {wave['wave_id']}: {wave['spawned_count']} agents at t={wave['departure_time']}s")

    print(f"\nTotal spawned: {spawned_total}/{num_agents}")
    print(f"Total evacuated: {total_evacuated}/{spawned_total} ({evacuation_pct:.1f}%)")
    print(f"Time to 95% evacuation (T95): {t95} seconds ({t95/60:.1f} minutes)" if t95 else "Time to 95% evacuation: Not reached")
    print(f"Mean evacuation time: {mean_evac_time:.1f} seconds ({mean_evac_time/60:.1f} minutes)")
    print(f"Peak congestion: {max_congested} roads at t={peak_congestion_time}s")
    print(f"Mean network speed: {sum(metrics.mean_speed)/len(metrics.mean_speed):.2f} km/h" if metrics.mean_speed else "Mean network speed: N/A")

    print("\n" + "=" * 80)

    return metrics


if __name__ == '__main__':
    # Default configuration
    NUM_AGENTS = 2000
    NUM_WAVES = 4
    WAVE_INTERVAL = 600  # 10 minutes
    SEED = 42
    OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "evacuation" / "data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "staged_evacuation.csv"

    # Run scenario
    metrics = run_staged_evacuation(
        num_agents=NUM_AGENTS,
        num_waves=NUM_WAVES,
        wave_interval_seconds=WAVE_INTERVAL,
        seed=SEED,
        output_file=OUTPUT_FILE
    )
