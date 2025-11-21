"""
Calculate Total Evacuation Time (TET) Metrics

Computes TET for each evacuation scenario based on Chen & Zhan (2008):
"Total Evacuation Time is the difference between the arrival time of the
last vehicle to reach its destination and the departure time of the first
evacuating vehicle"

Outputs a CSV with TET for each scenario.
"""

import pandas as pd
from pathlib import Path


def calculate_tet_from_timeseries(csv_path: str) -> dict:
    """
    Calculate TET from evacuation time series data.

    Args:
        csv_path: Path to scenario CSV file

    Returns:
        Dict with TET metrics
    """
    df = pd.read_csv(csv_path)

    # Total agents (maximum evacuated count)
    total_agents = df['evacuated_count'].max()

    # First departure time (assumed to be when first agent enters network)
    # For simplicity, we assume t=0, but we could track when first vehicle spawns
    first_departure_time = 0

    # Last arrival time (when all agents evacuated - 100%)
    final_evacuated = df['evacuated_count'].iloc[-1]

    if final_evacuated >= total_agents:
        # Find exact timestep when 100% evacuated
        evacuation_complete_idx = df[df['evacuated_count'] >= total_agents].index[0]
        last_arrival_time = df.loc[evacuation_complete_idx, 'time_step']
    else:
        # Not all evacuated - use final timestep
        last_arrival_time = df['time_step'].iloc[-1]

    # TET = last arrival - first departure
    tet_seconds = last_arrival_time - first_departure_time
    tet_minutes = tet_seconds / 60
    tet_hours = tet_minutes / 60

    return {
        'total_agents': total_agents,
        'evacuated_agents': final_evacuated,
        'evacuation_percentage': (final_evacuated / total_agents * 100) if total_agents > 0 else 0,
        'first_departure_time_seconds': first_departure_time,
        'last_arrival_time_seconds': last_arrival_time,
        'tet_seconds': tet_seconds,
        'tet_minutes': tet_minutes,
        'tet_hours': tet_hours
    }


def create_tet_summary(
    data_dir: str = "output/evacuation/data",
    output_file: str = "output/evacuation/data/total_evacuation_time_metrics.csv"
):
    """
    Create TET summary CSV for all scenarios.

    Args:
        data_dir: Directory containing scenario CSV files
        output_file: Path to save TET summary CSV
    """
    data_dir = Path(data_dir)

    scenarios = {
        'Simultaneous': data_dir / 'simultaneous_evacuation.csv',
        'Staged': data_dir / 'staged_evacuation.csv',
        'Simultaneous + Contraflow': data_dir / 'contraflow_evacuation.csv',
        'Staged + Contraflow': data_dir / 'staged_contraflow_evacuation.csv'
    }

    results = []

    print("=" * 80)
    print("CALCULATING TOTAL EVACUATION TIME (TET) METRICS")
    print("=" * 80)
    print("\nDefinition (Chen & Zhan 2008):")
    print("TET = arrival time of last vehicle - departure time of first vehicle")
    print("\n" + "-" * 80)

    for scenario_name, csv_path in scenarios.items():
        if not csv_path.exists():
            print(f"\n{scenario_name}: SKIPPED (file not found)")
            continue

        print(f"\n{scenario_name}:")

        try:
            metrics = calculate_tet_from_timeseries(str(csv_path))

            result = {
                'Scenario': scenario_name,
                'Total_Agents': metrics['total_agents'],
                'Evacuated_Agents': metrics['evacuated_agents'],
                'Evacuation_Percentage': metrics['evacuation_percentage'],
                'First_Departure_Time_seconds': metrics['first_departure_time_seconds'],
                'Last_Arrival_Time_seconds': metrics['last_arrival_time_seconds'],
                'TET_seconds': metrics['tet_seconds'],
                'TET_minutes': metrics['tet_minutes'],
                'TET_hours': metrics['tet_hours']
            }

            results.append(result)

            # Print summary
            print(f"  Total agents: {metrics['total_agents']}")
            print(f"  Evacuated: {metrics['evacuated_agents']} ({metrics['evacuation_percentage']:.1f}%)")
            print(f"  First departure: t={metrics['first_departure_time_seconds']}s")
            print(f"  Last arrival: t={metrics['last_arrival_time_seconds']}s")
            print(f"  TET: {metrics['tet_seconds']}s = {metrics['tet_minutes']:.1f} min = {metrics['tet_hours']:.2f} hrs")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)

        print("\n" + "=" * 80)
        print("TET SUMMARY TABLE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("\n" + "=" * 80)
        print(f"Saved to: {output_file}")
        print("=" * 80)

        return df
    else:
        print("\nNo results to save!")
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate Total Evacuation Time (TET) metrics for all scenarios'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='output/evacuation/data',
        help='Directory containing scenario CSV files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output/evacuation/data/total_evacuation_time_metrics.csv',
        help='Output CSV file path'
    )

    args = parser.parse_args()

    create_tet_summary(
        data_dir=args.data_dir,
        output_file=args.output
    )
