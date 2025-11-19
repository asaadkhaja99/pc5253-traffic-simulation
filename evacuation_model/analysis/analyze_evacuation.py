"""
Evacuation Scenario Analysis

Compares results from different evacuation scenarios:
1. Simultaneous evacuation (baseline)
2. Staged evacuation (wave-based)
3. Contraflow intervention

Generates:
- Comparative statistics table
- Performance metrics (T95, mean time, throughput)
- Effectiveness analysis (% improvement)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_scenario_data(scenario_file):
    """
    Load scenario results from CSV.

    Args:
        scenario_file: Path to scenario CSV

    Returns:
        DataFrame with scenario metrics
    """
    if not Path(scenario_file).exists():
        print(f"Warning: {scenario_file} not found")
        return None

    df = pd.read_csv(scenario_file)
    return df


def compute_scenario_metrics(df, num_agents):
    """
    Compute summary metrics for a scenario.

    Args:
        df: DataFrame with columns [time_step, evacuated_count, network_flow,
            mean_speed_kph, congested_roads]
        num_agents: Total number of agents

    Returns:
        Dictionary of metrics
    """
    if df is None or len(df) == 0:
        return None

    metrics = {}

    # Final evacuation count and percentage
    final_evacuated = df['evacuated_count'].iloc[-1]
    metrics['final_evacuated'] = final_evacuated
    metrics['evacuation_percentage'] = (final_evacuated / num_agents * 100) if num_agents > 0 else 0

    # T95: Time to 95% evacuation
    t95_rows = df[df['evacuated_count'] >= 0.95 * num_agents]
    if len(t95_rows) > 0:
        metrics['t95_seconds'] = t95_rows['time_step'].iloc[0]
        metrics['t95_minutes'] = metrics['t95_seconds'] / 60
    else:
        metrics['t95_seconds'] = None
        metrics['t95_minutes'] = None

    # Total simulation time
    metrics['total_time_seconds'] = df['time_step'].iloc[-1]
    metrics['total_time_minutes'] = metrics['total_time_seconds'] / 60

    # Mean evacuation time (approximate from cumulative evacuated)
    # For each timestep, compute incremental evacuees and weight by time
    df['incremental_evacuated'] = df['evacuated_count'].diff().fillna(df['evacuated_count'].iloc[0])
    weighted_time = (df['time_step'] * df['incremental_evacuated']).sum()
    metrics['mean_evacuation_time_seconds'] = weighted_time / final_evacuated if final_evacuated > 0 else 0
    metrics['mean_evacuation_time_minutes'] = metrics['mean_evacuation_time_seconds'] / 60

    # Peak congestion
    metrics['peak_congested_roads'] = df['congested_roads'].max()
    peak_idx = df['congested_roads'].idxmax()
    metrics['peak_congestion_time_seconds'] = df.loc[peak_idx, 'time_step']

    # Mean network speed
    metrics['mean_network_speed_kph'] = df['mean_speed_kph'].mean()

    # Network throughput (total vehicles processed)
    metrics['total_throughput'] = df['network_flow'].sum()

    return metrics


def compare_scenarios(baseline_metrics, intervention_metrics, intervention_name):
    """
    Compare intervention against baseline.

    Args:
        baseline_metrics: Dict of baseline scenario metrics
        intervention_metrics: Dict of intervention scenario metrics
        intervention_name: Name of intervention

    Returns:
        Dict of comparison metrics
    """
    if baseline_metrics is None or intervention_metrics is None:
        return None

    comparison = {
        'intervention': intervention_name
    }

    # T95 improvement
    if baseline_metrics['t95_seconds'] and intervention_metrics['t95_seconds']:
        t95_diff = baseline_metrics['t95_seconds'] - intervention_metrics['t95_seconds']
        t95_pct = (t95_diff / baseline_metrics['t95_seconds']) * 100
        comparison['t95_improvement_seconds'] = t95_diff
        comparison['t95_improvement_percentage'] = t95_pct
    else:
        comparison['t95_improvement_seconds'] = None
        comparison['t95_improvement_percentage'] = None

    # Mean evacuation time improvement
    baseline_mean = baseline_metrics['mean_evacuation_time_seconds']
    intervention_mean = intervention_metrics['mean_evacuation_time_seconds']
    mean_diff = baseline_mean - intervention_mean
    mean_pct = (mean_diff / baseline_mean) * 100 if baseline_mean > 0 else 0
    comparison['mean_time_improvement_seconds'] = mean_diff
    comparison['mean_time_improvement_percentage'] = mean_pct

    # Congestion reduction
    baseline_congestion = baseline_metrics['peak_congested_roads']
    intervention_congestion = intervention_metrics['peak_congested_roads']
    congestion_diff = baseline_congestion - intervention_congestion
    congestion_pct = (congestion_diff / baseline_congestion) * 100 if baseline_congestion > 0 else 0
    comparison['congestion_reduction_roads'] = congestion_diff
    comparison['congestion_reduction_percentage'] = congestion_pct

    # Speed improvement
    baseline_speed = baseline_metrics['mean_network_speed_kph']
    intervention_speed = intervention_metrics['mean_network_speed_kph']
    speed_diff = intervention_speed - baseline_speed
    speed_pct = (speed_diff / baseline_speed) * 100 if baseline_speed > 0 else 0
    comparison['speed_improvement_kph'] = speed_diff
    comparison['speed_improvement_percentage'] = speed_pct

    return comparison


def analyze_all_scenarios(
    simultaneous_file,
    staged_file,
    contraflow_file,
    num_agents,
    output_dir
):
    """
    Analyze and compare all evacuation scenarios.

    Args:
        simultaneous_file: Path to simultaneous scenario CSV
        staged_file: Path to staged scenario CSV
        contraflow_file: Path to contraflow scenario CSV
        num_agents: Number of agents in each scenario
        output_dir: Directory to save analysis results

    Returns:
        Dictionary with all metrics and comparisons
    """
    print("=" * 80)
    print("EVACUATION SCENARIO ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[1/3] Loading scenario data...")
    df_simultaneous = load_scenario_data(simultaneous_file)
    df_staged = load_scenario_data(staged_file)
    df_contraflow = load_scenario_data(contraflow_file)

    # Compute metrics for each scenario
    print("\n[2/3] Computing scenario metrics...")

    metrics_simultaneous = compute_scenario_metrics(df_simultaneous, num_agents) if df_simultaneous is not None else None
    metrics_staged = compute_scenario_metrics(df_staged, num_agents) if df_staged is not None else None
    metrics_contraflow = compute_scenario_metrics(df_contraflow, num_agents) if df_contraflow is not None else None

    # Create summary table
    summary_data = []

    if metrics_simultaneous:
        summary_data.append({
            'scenario': 'Simultaneous (Baseline)',
            'evacuated': metrics_simultaneous['final_evacuated'],
            'evacuation_pct': metrics_simultaneous['evacuation_percentage'],
            't95_min': metrics_simultaneous['t95_minutes'],
            'mean_time_min': metrics_simultaneous['mean_evacuation_time_minutes'],
            'peak_congestion': metrics_simultaneous['peak_congested_roads'],
            'mean_speed_kph': metrics_simultaneous['mean_network_speed_kph'],
            'total_throughput': metrics_simultaneous['total_throughput']
        })

    if metrics_staged:
        summary_data.append({
            'scenario': 'Staged Evacuation',
            'evacuated': metrics_staged['final_evacuated'],
            'evacuation_pct': metrics_staged['evacuation_percentage'],
            't95_min': metrics_staged['t95_minutes'],
            'mean_time_min': metrics_staged['mean_evacuation_time_minutes'],
            'peak_congestion': metrics_staged['peak_congested_roads'],
            'mean_speed_kph': metrics_staged['mean_network_speed_kph'],
            'total_throughput': metrics_staged['total_throughput']
        })

    if metrics_contraflow:
        summary_data.append({
            'scenario': 'Contraflow Intervention',
            'evacuated': metrics_contraflow['final_evacuated'],
            'evacuation_pct': metrics_contraflow['evacuation_percentage'],
            't95_min': metrics_contraflow['t95_minutes'],
            'mean_time_min': metrics_contraflow['mean_evacuation_time_minutes'],
            'peak_congestion': metrics_contraflow['peak_congested_roads'],
            'mean_speed_kph': metrics_contraflow['mean_network_speed_kph'],
            'total_throughput': metrics_contraflow['total_throughput']
        })

    summary_df = pd.DataFrame(summary_data)

    # Compare interventions to baseline
    print("\n[3/3] Comparing interventions to baseline...")

    comparisons = []

    if metrics_simultaneous and metrics_staged:
        comp_staged = compare_scenarios(metrics_simultaneous, metrics_staged, 'Staged Evacuation')
        if comp_staged:
            comparisons.append(comp_staged)

    if metrics_simultaneous and metrics_contraflow:
        comp_contraflow = compare_scenarios(metrics_simultaneous, metrics_contraflow, 'Contraflow Intervention')
        if comp_contraflow:
            comparisons.append(comp_contraflow)

    if len(comparisons) > 0:
        comparison_df = pd.DataFrame(comparisons)
    else:
        comparison_df = pd.DataFrame()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "scenario_comparison.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to: {summary_file}")

    if len(comparison_df) > 0:
        comparison_file = output_dir / "intervention_effectiveness.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Saved comparisons to: {comparison_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON SUMMARY")
    print("=" * 80)
    print("\nPerformance Metrics:")
    print(summary_df.to_string(index=False))

    if len(comparison_df) > 0:
        print("\n" + "=" * 80)
        print("INTERVENTION EFFECTIVENESS")
        print("=" * 80)
        print("\nImprovements vs. Baseline (Simultaneous):")
        print(comparison_df.to_string(index=False))

    print("\n" + "=" * 80)

    return {
        'summary': summary_df,
        'comparisons': comparison_df,
        'metrics': {
            'simultaneous': metrics_simultaneous,
            'staged': metrics_staged,
            'contraflow': metrics_contraflow
        }
    }


if __name__ == '__main__':
    # Default paths
    NUM_AGENTS = 2000
    BASE_DIR = Path(__file__).parent.parent.parent / "output" / "evacuation"
    DATA_DIR = BASE_DIR / "data"

    SIMULTANEOUS_FILE = DATA_DIR / "simultaneous_evacuation.csv"
    STAGED_FILE = DATA_DIR / "staged_evacuation.csv"
    CONTRAFLOW_FILE = DATA_DIR / "contraflow_evacuation.csv"

    # Run analysis (saves to data dir)
    results = analyze_all_scenarios(
        simultaneous_file=SIMULTANEOUS_FILE,
        staged_file=STAGED_FILE,
        contraflow_file=CONTRAFLOW_FILE,
        num_agents=NUM_AGENTS,
        output_dir=DATA_DIR
    )
