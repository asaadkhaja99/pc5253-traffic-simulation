"""
Create focused comparison plots for evacuation scenarios.

Generates two separate visualizations:
1. Staged vs Simultaneous - Comparing departure timing strategies
2. Contraflow vs Non-Contraflow - Comparing infrastructure interventions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from ..core.plot_utils import setup_high_res_plot_style, COLORS, LINE_STYLES


def load_scenario_data(csv_path: str) -> pd.DataFrame:
    """Load scenario data from CSV file."""
    df = pd.read_csv(csv_path)
    df['time_minutes'] = df['time_step'] / 60
    return df


def plot_staged_vs_simultaneous(
    simultaneous_csv: str,
    staged_csv: str,
    output_path: str = "output/evacuation/comparison_staged_vs_simultaneous.png"
):
    """
    Create comparison plot: Staged vs Simultaneous evacuation.

    Focus: Does staggered departure reduce congestion?
    """
    print("\n" + "=" * 80)
    print("CREATING: STAGED VS SIMULTANEOUS COMPARISON")
    print("=" * 80)

    # Setup plot style
    setup_high_res_plot_style(fontsize=23, dpi=300)

    # Load data
    df_sim = load_scenario_data(simultaneous_csv)
    df_staged = load_scenario_data(staged_csv)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Departure Strategy Comparison: Staged vs Simultaneous Evacuation',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Colors (using consistent palette)
    color_sim = COLORS['simultaneous']
    color_staged = COLORS['staged']
    lstyle_sim = LINE_STYLES['simultaneous']
    lstyle_staged = LINE_STYLES['staged']

    # 1. Evacuation Progress
    ax1 = axes[0, 0]
    ax1.plot(df_sim['time_minutes'], df_sim['evacuated_pct'],
             color=color_sim, label='Simultaneous', linestyle=lstyle_sim)
    ax1.plot(df_staged['time_minutes'], df_staged['evacuated_pct'],
             color=color_staged, label='Staged (3 waves)', linestyle=lstyle_staged)

    ax1.axhline(100, color='gray', linestyle=':', alpha=0.5, label='100% Evacuation')
    ax1.set_xlabel('Time (minutes)', fontweight='bold')
    ax1.set_ylabel('Evacuated (%)', fontweight='bold')
    ax1.set_title('Evacuation Progress', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # 2. Network Congestion
    ax2 = axes[0, 1]
    ax2.plot(df_sim['time_minutes'], df_sim['congested_roads'],
             color=color_sim, label='Simultaneous', linestyle=lstyle_sim)
    ax2.plot(df_staged['time_minutes'], df_staged['congested_roads'],
             color=color_staged, label='Staged (3 waves)', linestyle=lstyle_staged)

    ax2.set_xlabel('Time (minutes)', fontweight='bold')
    ax2.set_ylabel('Congested Roads', fontweight='bold')
    ax2.set_title('Network Congestion (speed < 30%)', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Add annotation for peak congestion reduction
    peak_sim = df_sim['congested_roads'].max()
    peak_staged = df_staged['congested_roads'].max()
    reduction_pct = (peak_sim - peak_staged) / peak_sim * 100

    ax2.text(
        0.98, 0.98,
        f'Peak Congestion Reduction:\n{reduction_pct:.1f}%',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # 3. Mean Network Speed
    ax3 = axes[1, 0]
    ax3.plot(df_sim['time_minutes'], df_sim['mean_speed'],
             color=color_sim, label='Simultaneous', linestyle=lstyle_sim, alpha=0.7)
    ax3.plot(df_staged['time_minutes'], df_staged['mean_speed'],
             color=color_staged, label='Staged (3 waves)', linestyle=lstyle_staged, alpha=0.7)

    ax3.set_xlabel('Time (minutes)', fontweight='bold')
    ax3.set_ylabel('Mean Speed (km/h)', fontweight='bold')
    ax3.set_title('Mean Network Speed', fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Add annotation for average speed improvement
    avg_speed_sim = df_sim['mean_speed'].mean()
    avg_speed_staged = df_staged['mean_speed'].mean()
    speed_improvement = (avg_speed_staged - avg_speed_sim) / avg_speed_sim * 100

    ax3.text(
        0.98, 0.02,
        f'Avg Speed Improvement:\n+{speed_improvement:.1f}%',
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    )

    # 4. Network Throughput
    ax4 = axes[1, 1]
    ax4.plot(df_sim['time_minutes'], df_sim['network_flow'],
             color=color_sim, label='Simultaneous', linestyle=lstyle_sim, alpha=0.7)
    ax4.plot(df_staged['time_minutes'], df_staged['network_flow'],
             color=color_staged, label='Staged (3 waves)', linestyle=lstyle_staged, alpha=0.7)

    ax4.set_xlabel('Time (minutes)', fontweight='bold')
    ax4.set_ylabel('Evacuation Rate (vehicles/step)', fontweight='bold')
    ax4.set_title('Network Throughput', fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()


def plot_contraflow_comparison(
    simultaneous_csv: str,
    contraflow_csv: str,
    output_path: str = "output/evacuation/comparison_contraflow.png"
):
    """
    Create comparison plot: Simultaneous with vs without Contraflow.

    Focus: Does increasing highway capacity help?
    """
    print("\n" + "=" * 80)
    print("CREATING: CONTRAFLOW COMPARISON")
    print("=" * 80)

    # Setup plot style
    setup_high_res_plot_style(fontsize=23, dpi=300)

    # Load data
    df_baseline = load_scenario_data(simultaneous_csv)
    df_contraflow = load_scenario_data(contraflow_csv)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Infrastructure Intervention Comparison: Contraflow Lanes Effect\n(Both scenarios: All evacuees depart simultaneously)',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Colors (using consistent palette)
    color_baseline = COLORS['baseline']
    color_contraflow = COLORS['contraflow']
    lstyle_baseline = LINE_STYLES['baseline']
    lstyle_contraflow = LINE_STYLES['contraflow']

    # 1. Evacuation Progress
    ax1 = axes[0, 0]
    ax1.plot(df_baseline['time_minutes'], df_baseline['evacuated_pct'],
             color=color_baseline, label='Baseline (Normal Capacity)', linestyle=lstyle_baseline)
    ax1.plot(df_contraflow['time_minutes'], df_contraflow['evacuated_pct'],
             color=color_contraflow, label='Contraflow (+50% Highway Capacity)', linestyle=lstyle_contraflow)

    ax1.axhline(100, color='gray', linestyle=':', alpha=0.5, label='100% Evacuation')
    ax1.set_xlabel('Time (minutes)', fontweight='bold')
    ax1.set_ylabel('Evacuated (%)', fontweight='bold')
    ax1.set_title('Evacuation Progress', fontweight='bold')
    ax1.legend(loc='best', )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # 2. Network Congestion
    ax2 = axes[0, 1]
    ax2.plot(df_baseline['time_minutes'], df_baseline['congested_roads'],
             color=color_baseline, label='Baseline', linestyle=lstyle_baseline)
    ax2.plot(df_contraflow['time_minutes'], df_contraflow['congested_roads'],
             color=color_contraflow, label='Contraflow', linestyle=lstyle_contraflow)

    ax2.set_xlabel('Time (minutes)', fontweight='bold')
    ax2.set_ylabel('Congested Roads', fontweight='bold')
    ax2.set_title('Network Congestion (speed < 30%)', fontweight='bold')
    ax2.legend(loc='best', )
    ax2.grid(True, alpha=0.3)

    # Add annotation for congestion difference
    avg_cong_baseline = df_baseline['congested_roads'].mean()
    avg_cong_contraflow = df_contraflow['congested_roads'].mean()
    cong_change = (avg_cong_contraflow - avg_cong_baseline) / avg_cong_baseline * 100

    ax2.text(
        0.98, 0.98,
        f'Avg Congestion Change:\n{cong_change:+.1f}%',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # 3. Mean Network Speed
    ax3 = axes[1, 0]
    ax3.plot(df_baseline['time_minutes'], df_baseline['mean_speed'],
             color=color_baseline, label='Baseline', linestyle=lstyle_baseline, alpha=0.7)
    ax3.plot(df_contraflow['time_minutes'], df_contraflow['mean_speed'],
             color=color_contraflow, label='Contraflow', linestyle=lstyle_contraflow, alpha=0.7)

    ax3.set_xlabel('Time (minutes)', fontweight='bold')
    ax3.set_ylabel('Mean Speed (km/h)', fontweight='bold')
    ax3.set_title('Mean Network Speed', fontweight='bold')
    ax3.legend(loc='best', )
    ax3.grid(True, alpha=0.3)

    # Add annotation for speed improvement
    avg_speed_baseline = df_baseline['mean_speed'].mean()
    avg_speed_contraflow = df_contraflow['mean_speed'].mean()
    speed_improvement = (avg_speed_contraflow - avg_speed_baseline) / avg_speed_baseline * 100

    ax3.text(
        0.98, 0.02,
        f'Avg Speed Improvement:\n+{speed_improvement:.1f}%',
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    )

    # 4. Network Throughput
    ax4 = axes[1, 1]
    ax4.plot(df_baseline['time_minutes'], df_baseline['network_flow'],
             color=color_baseline, label='Baseline', linestyle=lstyle_baseline, alpha=0.7)
    ax4.plot(df_contraflow['time_minutes'], df_contraflow['network_flow'],
             color=color_contraflow, label='Contraflow', linestyle=lstyle_contraflow, alpha=0.7)

    ax4.set_xlabel('Time (minutes)', fontweight='bold')
    ax4.set_ylabel('Evacuation Rate (vehicles/step)', fontweight='bold')
    ax4.set_title('Network Throughput', fontweight='bold')
    ax4.legend(loc='best', )
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create focused comparison plots for evacuation scenarios'
    )

    parser.add_argument(
        '--simultaneous',
        type=str,
        default='output/evacuation/simultaneous_metrics.csv',
        help='Path to simultaneous scenario CSV'
    )

    parser.add_argument(
        '--staged',
        type=str,
        default='output/evacuation/staged_metrics.csv',
        help='Path to staged scenario CSV'
    )

    parser.add_argument(
        '--contraflow',
        type=str,
        default='output/evacuation/contraflow_metrics.csv',
        help='Path to contraflow scenario CSV'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/evacuation',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EVACUATION SCENARIO COMPARISON PLOTS")
    print("=" * 80)

    # Create both comparison plots
    plot_staged_vs_simultaneous(
        simultaneous_csv=args.simultaneous,
        staged_csv=args.staged,
        output_path=f"{args.output_dir}/comparison_staged_vs_simultaneous.png"
    )

    plot_contraflow_comparison(
        simultaneous_csv=args.simultaneous,
        contraflow_csv=args.contraflow,
        output_path=f"{args.output_dir}/comparison_contraflow.png"
    )

    print("\n" + "=" * 80)
    print("COMPARISON PLOTS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print("  1. comparison_staged_vs_simultaneous.png")
    print("  2. comparison_contraflow.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
