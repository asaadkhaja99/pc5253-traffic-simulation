"""
Master Script for Urban Planning Study

Runs Paya Lebar localized incident scenario with IDM:
- Lane closure near MRT junction
- IDM with Gaussian bottleneck factor (Eq. 17)
- Queue and delay metrics

USAGE:
    python run_urban_study.py [--quick]

OPTIONS:
    --quick    Run with reduced vehicles for testing
"""

import argparse
from pathlib import Path
import time

from scenario_paya_lebar import run_paya_lebar_scenario
from visualize_urban import visualize_all_urban_scenarios


def run_complete_urban_study(
    paya_lebar_vehicles=8000,
    seed=42,
    output_dir=None
):
    """
    Run Paya Lebar urban planning study.

    Args:
        paya_lebar_vehicles: Number of vehicles for Paya Lebar scenario
        seed: Random seed
        output_dir: Output directory

    Returns:
        Dictionary with results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output" / "urban_planning"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("=" * 80)
    print(" " * 15 + "PAYA LEBAR URBAN PLANNING STUDY - IDM")
    print("=" * 80)
    print(f"\nStudy Configuration:")
    print(f"  Vehicles: {paya_lebar_vehicles}")
    print(f"  Model: IDM with Gaussian Bottleneck (ε=0.8, σ=50m)")
    print(f"  Random seed: {seed}")
    print(f"  Output directory: {output_dir}")
    print("\n" + "=" * 80)

    results = {}
    start_time_total = time.time()

    # =========================================================================
    # Paya Lebar Localized Incident
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# PAYA LEBAR LOCALIZED INCIDENT")
    print("#" * 80)
    print("\n")

    try:
        # Baseline
        print("Running baseline...")
        start_time = time.time()
        baseline_file = output_dir / "paya_lebar_baseline.csv"
        metrics_pl_baseline = run_paya_lebar_scenario(
            with_closure=False,
            num_vehicles=paya_lebar_vehicles,
            seed=seed,
            output_file=baseline_file
        )
        results['paya_lebar_baseline'] = metrics_pl_baseline
        print(f"Baseline completed in {time.time() - start_time:.1f}s\n")

        # With closure
        print("Running with lane closure...")
        start_time = time.time()
        closure_file = output_dir / "paya_lebar_closure.csv"
        metrics_pl_closure = run_paya_lebar_scenario(
            with_closure=True,
            closure_capacity_reduction=0.5,
            num_vehicles=paya_lebar_vehicles,
            seed=seed,
            output_file=closure_file
        )
        results['paya_lebar_closure'] = metrics_pl_closure
        print(f"Closure scenario completed in {time.time() - start_time:.1f}s")

    except Exception as e:
        print(f"\nERROR in Paya Lebar scenario: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# VISUALIZATION GENERATION")
    print("#" * 80)
    print("\n")

    try:
        visualize_all_urban_scenarios(
            paya_lebar_baseline=output_dir / "paya_lebar_baseline.csv",
            paya_lebar_closure=output_dir / "paya_lebar_closure.csv",
            pie_baseline=None,  # PIE scenario removed
            pie_closure=None,
            pie_baseline_ratrun=None,
            pie_closure_ratrun=None,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nERROR in Visualization: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_elapsed = time.time() - start_time_total

    print("\n")
    print("=" * 80)
    print(" " * 25 + "STUDY COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  Data (CSV):")
    print(f"    - paya_lebar_baseline.csv")
    print(f"    - paya_lebar_closure.csv")
    print(f"  Visualizations (PNG):")
    print(f"    - paya_lebar_comparison.png")
    print(f"    - paya_lebar_impact.png")
    print(f"    - paya_lebar_scenario_map.png")
    print("\n" + "=" * 80)

    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Run complete urban planning study'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with reduced vehicles'
    )

    parser.add_argument(
        '--vehicles',
        type=int,
        default=None,
        help='Number of vehicles (default: 800, quick: 300)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Determine vehicle count
    if args.quick:
        vehicles = args.vehicles or 300
    else:
        vehicles = args.vehicles or 800

    # Run study
    results = run_complete_urban_study(
        paya_lebar_vehicles=vehicles,
        seed=args.seed
    )

    return results


if __name__ == '__main__':
    results = main()
