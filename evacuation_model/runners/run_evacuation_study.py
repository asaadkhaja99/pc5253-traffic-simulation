"""
Master Script for Evacuation Study

Runs all evacuation scenarios sequentially:
1. Simultaneous evacuation (baseline)
2. Staged evacuation (4 waves)
3. Contraflow intervention

Then performs:
4. Comparative analysis
5. Visualization generation

This is the main entry point for the complete evacuation study.

USAGE:
    python run_evacuation_study.py [--num-agents N] [--seed S] [--quick]

OPTIONS:
    --num-agents N    Number of evacuees (default: 2000)
    --seed S          Random seed (default: 42)
    --quick           Run with reduced agents for testing (500 agents)
"""

import sys
import argparse
from pathlib import Path
import time

# Import scenario modules
from ..scenarios.scenario_simultaneous import run_simultaneous_evacuation
from ..scenarios.scenario_staged import run_staged_evacuation
from ..scenarios.contraflow_intervention import run_contraflow_evacuation
from ..scenarios.scenario_staged_contraflow import run_staged_contraflow_evacuation
from ..analysis.analyze_evacuation import analyze_all_scenarios
from ..analysis.calculate_tet_metrics import create_tet_summary
from ..visualization.visualize_evacuation import visualize_all_scenarios
from ..visualization.create_intervention_effectiveness import create_intervention_effectiveness_plot


def run_complete_study(
    num_agents=2000,
    num_waves=4,
    wave_interval=600,
    num_contraflow_roads=4,
    seed=42,
    output_dir=None
):
    """
    Run complete evacuation study with all scenarios.

    Args:
        num_agents: Number of evacuees
        num_waves: Number of waves for staged evacuation
        wave_interval: Time between waves (seconds)
        num_contraflow_roads: Number of roads for contraflow
        seed: Random seed
        output_dir: Output directory for results

    Returns:
        Dictionary with all results
    """
    if output_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "output" / "evacuation"
    else:
        base_dir = Path(output_dir)

    # Create data and figures subdirectories
    data_dir = base_dir / "data"
    figures_dir = base_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n")
    print("=" * 80)
    print(" " * 20 + "EVACUATION STUDY - MASTER RUNNER")
    print("=" * 80)
    print(f"\nStudy Configuration:")
    print(f"  Number of agents: {num_agents}")
    print(f"  Staged waves: {num_waves} (interval: {wave_interval}s)")
    print(f"  Contraflow roads: {num_contraflow_roads}")
    print(f"  Random seed: {seed}")
    print(f"  Output directory: {base_dir}")
    print(f"    - Data: {data_dir}")
    print(f"    - Figures: {figures_dir}")
    print("\n" + "=" * 80)

    results = {}
    start_time_total = time.time()

    # =========================================================================
    # SCENARIO 1: Simultaneous Evacuation (Baseline)
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# SCENARIO 1/3: SIMULTANEOUS EVACUATION (BASELINE)")
    print("#" * 80)
    print("\n")

    start_time = time.time()
    simultaneous_file = data_dir / "simultaneous_evacuation.csv"

    try:
        metrics_simultaneous = run_simultaneous_evacuation(
            num_agents=num_agents,
            seed=seed,
            output_file=simultaneous_file
        )
        results['simultaneous'] = metrics_simultaneous
        elapsed = time.time() - start_time
        print(f"\nScenario 1 completed in {elapsed:.1f} seconds")
    except Exception as e:
        print(f"\nERROR in Scenario 1: {e}")
        import traceback
        traceback.print_exc()
        results['simultaneous'] = None

    # =========================================================================
    # SCENARIO 2: Staged Evacuation
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# SCENARIO 2/3: STAGED EVACUATION")
    print("#" * 80)
    print("\n")

    start_time = time.time()
    staged_file = data_dir / "staged_evacuation.csv"

    try:
        metrics_staged = run_staged_evacuation(
            num_agents=num_agents,
            num_waves=num_waves,
            wave_interval_seconds=wave_interval,
            seed=seed,
            output_file=staged_file
        )
        results['staged'] = metrics_staged
        elapsed = time.time() - start_time
        print(f"\nScenario 2 completed in {elapsed:.1f} seconds")
    except Exception as e:
        print(f"\nERROR in Scenario 2: {e}")
        import traceback
        traceback.print_exc()
        results['staged'] = None

    # =========================================================================
    # SCENARIO 3: Contraflow Intervention (Simultaneous)
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# SCENARIO 3/4: CONTRAFLOW INTERVENTION (SIMULTANEOUS)")
    print("#" * 80)
    print("\n")

    start_time = time.time()
    contraflow_file = data_dir / "contraflow_evacuation.csv"

    try:
        metrics_contraflow, contraflow_info = run_contraflow_evacuation(
            num_agents=num_agents,
            num_contraflow_roads=num_contraflow_roads,
            contraflow_activation_time=0,
            seed=seed,
            output_file=contraflow_file
        )
        results['contraflow'] = metrics_contraflow
        results['contraflow_info'] = contraflow_info
        elapsed = time.time() - start_time
        print(f"\nScenario 3 completed in {elapsed:.1f} seconds")
    except Exception as e:
        print(f"\nERROR in Scenario 3: {e}")
        import traceback
        traceback.print_exc()
        results['contraflow'] = None

    # =========================================================================
    # SCENARIO 4: Staged + Contraflow
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# SCENARIO 4/4: STAGED EVACUATION + CONTRAFLOW")
    print("#" * 80)
    print("\n")

    start_time = time.time()
    staged_contraflow_file = data_dir / "staged_contraflow_evacuation.csv"

    try:
        metrics_staged_contraflow, staged_contraflow_info = run_staged_contraflow_evacuation(
            num_agents=num_agents,
            num_waves=num_waves,
            wave_interval_seconds=wave_interval,
            num_contraflow_roads=num_contraflow_roads,
            contraflow_activation_time=0,
            seed=seed,
            output_file=staged_contraflow_file
        )
        results['staged_contraflow'] = metrics_staged_contraflow
        results['staged_contraflow_info'] = staged_contraflow_info
        elapsed = time.time() - start_time
        print(f"\nScenario 4 completed in {elapsed:.1f} seconds")
    except Exception as e:
        print(f"\nERROR in Scenario 4: {e}")
        import traceback
        traceback.print_exc()
        results['staged_contraflow'] = None

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# COMPARATIVE ANALYSIS")
    print("#" * 80)
    print("\n")

    try:
        analysis_results = analyze_all_scenarios(
            simultaneous_file=simultaneous_file,
            staged_file=staged_file,
            contraflow_file=contraflow_file,
            num_agents=num_agents,
            output_dir=data_dir
        )
        results['analysis'] = analysis_results
    except Exception as e:
        print(f"\nERROR in Analysis: {e}")
        import traceback
        traceback.print_exc()
        results['analysis'] = None

    # =========================================================================
    # TOTAL EVACUATION TIME (TET) METRICS
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# TOTAL EVACUATION TIME (TET) CALCULATION")
    print("#" * 80)
    print("\n")

    try:
        tet_summary = create_tet_summary(
            data_dir=str(data_dir),
            output_file=str(data_dir / "total_evacuation_time_metrics.csv")
        )
        results['tet_summary'] = tet_summary
    except Exception as e:
        print(f"\nERROR in TET Calculation: {e}")
        import traceback
        traceback.print_exc()
        results['tet_summary'] = None

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n")
    print("#" * 80)
    print("# VISUALIZATION GENERATION")
    print("#" * 80)
    print("\n")

    try:
        # Generate standard comparison plots (time series)
        print("Generating time-series comparison plots...")
        comparison_file = data_dir / "intervention_effectiveness.csv"
        visualize_all_scenarios(
            simultaneous_file=simultaneous_file,
            staged_file=staged_file,
            contraflow_file=contraflow_file,
            comparison_file=comparison_file,
            output_dir=figures_dir
        )

        # Generate 4-scenario intervention effectiveness plot (absolute metrics)
        print("\nGenerating 4-scenario intervention effectiveness plot...")
        create_intervention_effectiveness_plot(
            simultaneous_csv=str(simultaneous_file),
            staged_csv=str(staged_file),
            contraflow_csv=str(contraflow_file),
            staged_contraflow_csv=str(staged_contraflow_file),
            output_path=str(figures_dir / "intervention_effectiveness_4scenarios.png")
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
    print(f"\nResults saved to: {base_dir}")
    print(f"\nGenerated files:")
    print(f"\n  Data files (output/evacuation/data/):")
    print(f"    - simultaneous_evacuation.csv")
    print(f"    - staged_evacuation.csv")
    print(f"    - contraflow_evacuation.csv")
    print(f"    - contraflow_evacuation_contraflow_roads.csv")
    print(f"    - staged_contraflow_evacuation.csv")
    print(f"    - staged_contraflow_evacuation_contraflow_roads.csv")
    print(f"    - scenario_comparison.csv")
    print(f"    - intervention_effectiveness.csv")
    print(f"    - total_evacuation_time_metrics.csv (TET per scenario)")
    print(f"\n  Figures (output/evacuation/figures/):")
    print(f"    - evacuation_progress.png")
    print(f"    - congestion_comparison.png")
    print(f"    - speed_comparison.png")
    print(f"    - intervention_effectiveness.png (old % comparison)")
    print(f"    - intervention_effectiveness_4scenarios.png (NEW: 4-scenario absolute metrics)")
    print(f"    - evacuation_dashboard.png")
    print("\n" + "=" * 80)

    return results


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run complete evacuation scenario study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (2000 agents)
  python run_evacuation_study.py

  # Run with custom agent count
  python run_evacuation_study.py --num-agents 1000

  # Quick test run (500 agents)
  python run_evacuation_study.py --quick

  # Custom seed for reproducibility
  python run_evacuation_study.py --seed 123
        """
    )

    parser.add_argument(
        '--num-agents',
        type=int,
        default=2000,
        help='Number of evacuees (default: 2000)'
    )

    parser.add_argument(
        '--num-waves',
        type=int,
        default=4,
        help='Number of waves for staged evacuation (default: 4)'
    )

    parser.add_argument(
        '--wave-interval',
        type=int,
        default=600,
        help='Time between waves in seconds (default: 600 = 10 minutes)'
    )

    parser.add_argument(
        '--num-contraflow',
        type=int,
        default=4,
        help='Number of contraflow roads (default: 4)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: <project_root>/output/evacuation)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with 500 agents'
    )

    args = parser.parse_args()

    # Override num_agents if quick mode
    num_agents = 500 if args.quick else args.num_agents

    # Run study
    results = run_complete_study(
        num_agents=num_agents,
        num_waves=args.num_waves,
        wave_interval=args.wave_interval,
        num_contraflow_roads=args.num_contraflow,
        seed=args.seed,
        output_dir=args.output_dir
    )

    return results


if __name__ == '__main__':
    results = main()
