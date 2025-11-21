"""
Evacuation Scenario Visualization

Generates visualizations for evacuation scenario analysis:
1. Evacuation progress time series (all scenarios)
2. Congestion level comparison
3. Network speed comparison
4. Contraflow effectiveness
5. Spatial congestion heatmap (optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..core.plot_utils import setup_plot_style, COLORS, LINE_STYLES


def plot_evacuation_progress(
    scenarios_data,
    output_file
):
    """
    Plot evacuation progress over time for all scenarios.

    Args:
        scenarios_data: Dict of {scenario_name: DataFrame}
        output_file: Path to save plot

    Note: Uses actual agent count from each scenario's data (max evacuated_count)
          rather than a fixed parameter, since scenarios may spawn different numbers.
    """
    # Check if any data exists
    has_data = any(df is not None and len(df) > 0 for df in scenarios_data.values())
    if not has_data:
        print(f"Warning: No scenario data available for {output_file.name}")
        print("  Please run evacuation scenarios first:")
        print("  - python scenario_simultaneous.py")
        print("  - python scenario_staged.py")
        print("  - python contraflow_intervention.py")
        return

    # Apply consistent plot styling
    setup_plot_style(fontsize=23)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

    colors = {
        'Simultaneous': COLORS['simultaneous'],
        'Staged': COLORS['staged'],
        'Contraflow': COLORS['contraflow']
    }

    linestyles = {
        'Simultaneous': LINE_STYLES['simultaneous'],
        'Staged': LINE_STYLES['staged'],
        'Contraflow': LINE_STYLES['contraflow']
    }

    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            # Use actual number of agents in this scenario (max evacuated count)
            actual_agents = df['evacuated_count'].max()
            evac_pct = (df['evacuated_count'] / actual_agents) * 100

            ax.plot(
                time_min,
                evac_pct,
                label=scenario_name,
                color=colors.get(scenario_name, '#95a5a6'),
                linestyle=linestyles.get(scenario_name, '-'),
                linewidth=2.5
            )

    # 100% threshold line
    ax.axhline(100, color='red', linestyle=':', linewidth=2, alpha=0.7, label='100% Evacuation')

    ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evacuated (%)', fontsize=14, fontweight='bold')
    ax.set_title('Evacuation Progress Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_congestion_comparison(
    scenarios_data,
    output_file
):
    """
    Plot congestion levels over time for all scenarios.

    Args:
        scenarios_data: Dict of {scenario_name: DataFrame}
        output_file: Path to save plot
    """
    # Check if any data exists
    has_data = any(df is not None and len(df) > 0 for df in scenarios_data.values())
    if not has_data:
        print(f"Warning: No scenario data available for {output_file.name}")
        return

    # Apply consistent plot styling
    setup_plot_style(fontsize=23)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

    colors = {
        'Simultaneous': COLORS['simultaneous'],
        'Staged': COLORS['staged'],
        'Contraflow': COLORS['contraflow']
    }

    linestyles = {
        'Simultaneous': LINE_STYLES['simultaneous'],
        'Staged': LINE_STYLES['staged'],
        'Contraflow': LINE_STYLES['contraflow']
    }

    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            congested = df['congested_roads']

            ax.plot(
                time_min,
                congested,
                label=scenario_name,
                color=colors.get(scenario_name, '#95a5a6'),
                linestyle=linestyles.get(scenario_name, '-'),
                linewidth=2.5
            )

    ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Congested Roads (count)', fontsize=14, fontweight='bold')
    ax.set_title('Network Congestion Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_speed_comparison(
    scenarios_data,
    output_file
):
    """
    Plot mean network speed over time for all scenarios.

    Args:
        scenarios_data: Dict of {scenario_name: DataFrame}
        output_file: Path to save plot
    """
    # Check if any data exists
    has_data = any(df is not None and len(df) > 0 for df in scenarios_data.values())
    if not has_data:
        print(f"Warning: No scenario data available for {output_file.name}")
        return

    # Apply consistent plot styling
    setup_plot_style(fontsize=23)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

    colors = {
        'Simultaneous': COLORS['simultaneous'],
        'Staged': COLORS['staged'],
        'Contraflow': COLORS['contraflow']
    }

    linestyles = {
        'Simultaneous': LINE_STYLES['simultaneous'],
        'Staged': LINE_STYLES['staged'],
        'Contraflow': LINE_STYLES['contraflow']
    }

    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            speed = df['mean_speed_kph']

            ax.plot(
                time_min,
                speed,
                label=scenario_name,
                color=colors.get(scenario_name, '#95a5a6'),
                linestyle=linestyles.get(scenario_name, '-'),
                linewidth=2.5
            )

    ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Network Speed (km/h)', fontsize=14, fontweight='bold')
    ax.set_title('Network Speed Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_intervention_effectiveness(
    comparison_df,
    output_file
):
    """
    Bar chart showing intervention effectiveness.

    Args:
        comparison_df: DataFrame with intervention comparisons
        output_file: Path to save plot
    """
    if comparison_df is None or len(comparison_df) == 0:
        print("Warning: No comparison data available for effectiveness plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

    interventions = comparison_df['intervention'].tolist()
    colors = ['#3498db', '#2ecc71']

    # T95 improvement
    ax1 = axes[0, 0]
    t95_improvements = comparison_df['t95_improvement_percentage'].fillna(0).tolist()
    bars1 = ax1.bar(interventions, t95_improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('T95 Evacuation Time Improvement', fontsize=13, fontweight='bold')
    ax1.axhline(0, color='black', linewidth=1, linestyle='-')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(bottom=min(0, min(t95_improvements) - 5))

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

    # Mean time improvement
    ax2 = axes[0, 1]
    mean_improvements = comparison_df['mean_time_improvement_percentage'].fillna(0).tolist()
    bars2 = ax2.bar(interventions, mean_improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Evacuation Time Improvement', fontsize=13, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=1, linestyle='-')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim(bottom=min(0, min(mean_improvements) - 5))

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

    # Congestion reduction
    ax3 = axes[1, 0]
    congestion_reductions = comparison_df['congestion_reduction_percentage'].fillna(0).tolist()
    bars3 = ax3.bar(interventions, congestion_reductions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Reduction (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Peak Congestion Reduction', fontsize=13, fontweight='bold')
    ax3.axhline(0, color='black', linewidth=1, linestyle='-')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.set_ylim(bottom=min(0, min(congestion_reductions) - 5))

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

    # Speed improvement
    ax4 = axes[1, 1]
    speed_improvements = comparison_df['speed_improvement_percentage'].fillna(0).tolist()
    bars4 = ax4.bar(interventions, speed_improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Network Speed Improvement', fontsize=13, fontweight='bold')
    ax4.axhline(0, color='black', linewidth=1, linestyle='-')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.set_ylim(bottom=min(0, min(speed_improvements) - 5))

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=11)

    plt.suptitle('Intervention Effectiveness vs. Baseline (Simultaneous)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def plot_combined_dashboard(
    scenarios_data,
    output_file
):
    """
    Create a combined dashboard with multiple metrics.

    Args:
        scenarios_data: Dict of {scenario_name: DataFrame}
        output_file: Path to save plot

    Note: Uses actual agent count from each scenario's data (max evacuated_count)
          rather than a fixed parameter, since scenarios may spawn different numbers.
    """
    # Check if any data exists
    has_data = any(df is not None and len(df) > 0 for df in scenarios_data.values())
    if not has_data:
        print(f"Warning: No scenario data available for {output_file.name}")
        return

    # Apply consistent plot styling
    setup_plot_style(fontsize=23)

    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    colors = {
        'Simultaneous': COLORS['simultaneous'],
        'Staged': COLORS['staged'],
        'Contraflow': COLORS['contraflow']
    }

    linestyles = {
        'Simultaneous': LINE_STYLES['simultaneous'],
        'Staged': LINE_STYLES['staged'],
        'Contraflow': LINE_STYLES['contraflow']
    }

    # 1. Evacuation progress
    ax1 = fig.add_subplot(gs[0, 0])
    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            # Use actual number of agents in this scenario (max evacuated count)
            actual_agents = df['evacuated_count'].max()
            evac_pct = (df['evacuated_count'] / actual_agents) * 100
            ax1.plot(time_min, evac_pct, label=scenario_name,
                    color=colors.get(scenario_name, '#95a5a6'),
                    linestyle=linestyles.get(scenario_name, '-'), linewidth=2.5)
    ax1.axhline(100, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Evacuated (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Evacuation Progress', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    ax1.set_ylim(0, 105)

    # 2. Congestion
    ax2 = fig.add_subplot(gs[0, 1])
    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            congested = df['congested_roads']
            ax2.plot(time_min, congested, label=scenario_name,
                    color=colors.get(scenario_name, '#95a5a6'),
                    linestyle=linestyles.get(scenario_name, '-'), linewidth=2.5)
    ax2.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Congested Roads', fontsize=11, fontweight='bold')
    ax2.set_title('Network Congestion', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)

    # 3. Network speed
    ax3 = fig.add_subplot(gs[1, 0])
    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            speed = df['mean_speed_kph']
            ax3.plot(time_min, speed, label=scenario_name,
                    color=colors.get(scenario_name, '#95a5a6'),
                    linestyle=linestyles.get(scenario_name, '-'), linewidth=2.5)
    ax3.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Speed (km/h)', fontsize=11, fontweight='bold')
    ax3.set_title('Mean Network Speed', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)

    # 4. Network flow (evacuation rate)
    ax4 = fig.add_subplot(gs[1, 1])
    for scenario_name, df in scenarios_data.items():
        if df is not None and len(df) > 0:
            time_min = df['time_step'] / 60
            # Smooth flow with rolling average
            if len(df) > 10:
                flow_smooth = df['network_flow'].rolling(window=10, min_periods=1).mean()
            else:
                flow_smooth = df['network_flow']
            ax4.plot(time_min, flow_smooth, label=scenario_name,
                    color=colors.get(scenario_name, '#95a5a6'),
                    linestyle=linestyles.get(scenario_name, '-'), linewidth=2.5)
    ax4.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Evacuation Rate (veh/step)', fontsize=11, fontweight='bold')
    ax4.set_title('Network Throughput', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(left=0)

    # plt.suptitle('Evacuation Metrics', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    plt.close()


def visualize_all_scenarios(
    simultaneous_file,
    staged_file,
    contraflow_file,
    comparison_file,
    output_dir
):
    """
    Generate all visualizations for evacuation scenarios.

    Args:
        simultaneous_file: Path to simultaneous scenario CSV
        staged_file: Path to staged scenario CSV
        contraflow_file: Path to contraflow scenario CSV
        comparison_file: Path to comparison CSV
        output_dir: Directory to save plots

    Note: Agent count is automatically determined from each scenario's data
          (max evacuated_count) to handle scenarios with different spawn numbers.
    """
    print("=" * 80)
    print("EVACUATION SCENARIO VISUALIZATION")
    print("=" * 80)

    # Load data
    print("\n[1/2] Loading scenario data...")
    scenarios_data = {}

    if Path(simultaneous_file).exists():
        df = pd.read_csv(simultaneous_file)
        if len(df) > 0:
            scenarios_data['Simultaneous'] = df
            print(f"  ✓ Loaded: {simultaneous_file} ({len(df)} rows)")
        else:
            scenarios_data['Simultaneous'] = None
            print(f"  ✗ Empty: {simultaneous_file}")
    else:
        scenarios_data['Simultaneous'] = None
        print(f"  ✗ Not found: {simultaneous_file}")

    if Path(staged_file).exists():
        df = pd.read_csv(staged_file)
        if len(df) > 0:
            scenarios_data['Staged'] = df
            print(f"  ✓ Loaded: {staged_file} ({len(df)} rows)")
        else:
            scenarios_data['Staged'] = None
            print(f"  ✗ Empty: {staged_file}")
    else:
        scenarios_data['Staged'] = None
        print(f"  ✗ Not found: {staged_file}")

    if Path(contraflow_file).exists():
        df = pd.read_csv(contraflow_file)
        if len(df) > 0:
            scenarios_data['Contraflow'] = df
            print(f"  ✓ Loaded: {contraflow_file} ({len(df)} rows)")
        else:
            scenarios_data['Contraflow'] = None
            print(f"  ✗ Empty: {contraflow_file}")
    else:
        scenarios_data['Contraflow'] = None
        print(f"  ✗ Not found: {contraflow_file}")

    comparison_df = None
    if Path(comparison_file).exists():
        comparison_df = pd.read_csv(comparison_file)
        print(f"  ✓ Loaded: {comparison_file}")

    # Check if we have any data at all
    has_any_data = any(df is not None and len(df) > 0 for df in scenarios_data.values())
    if not has_any_data:
        print("\n" + "!" * 80)
        print("ERROR: No scenario data found!")
        print("!" * 80)
        print("\nYou must run the evacuation scenarios first:")
        print("  1. cd evacuation_model")
        print("  2. Run one of:")
        print("     - uv run python run_evacuation_study.py --quick  (fast test)")
        print("     - uv run python run_evacuation_study.py          (full study)")
        print("\nOr run scenarios individually:")
        print("     - uv run python scenario_simultaneous.py")
        print("     - uv run python scenario_staged.py")
        print("     - uv run python contraflow_intervention.py")
        print("\n" + "!" * 80)
        return

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n[2/2] Generating visualizations...")

    plot_evacuation_progress(
        scenarios_data,
        output_dir / "evacuation_progress.png"
    )

    plot_congestion_comparison(
        scenarios_data,
        output_dir / "congestion_comparison.png"
    )

    plot_speed_comparison(
        scenarios_data,
        output_dir / "speed_comparison.png"
    )

    if comparison_df is not None:
        plot_intervention_effectiveness(
            comparison_df,
            output_dir / "intervention_effectiveness.png"
        )

    plot_combined_dashboard(
        scenarios_data,
        output_dir / "evacuation_dashboard.png"
    )

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    # Default paths
    NUM_AGENTS = 2000
    BASE_DIR = Path(__file__).parent.parent.parent / "output" / "evacuation"
    DATA_DIR = BASE_DIR / "data"
    FIGURES_DIR = BASE_DIR / "figures"

    SIMULTANEOUS_FILE = DATA_DIR / "simultaneous_evacuation.csv"
    STAGED_FILE = DATA_DIR / "staged_evacuation.csv"
    CONTRAFLOW_FILE = DATA_DIR / "contraflow_evacuation.csv"
    COMPARISON_FILE = DATA_DIR / "intervention_effectiveness.csv"

    # Generate visualizations (saves to figures dir)
    visualize_all_scenarios(
        simultaneous_file=SIMULTANEOUS_FILE,
        staged_file=STAGED_FILE,
        contraflow_file=CONTRAFLOW_FILE,
        comparison_file=COMPARISON_FILE,
        num_agents=NUM_AGENTS,
        output_dir=FIGURES_DIR
    )
