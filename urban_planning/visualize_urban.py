"""
Urban Planning Visualization

Generates visualizations for urban planning scenarios:
1. Paya Lebar: Queue spillback, congestion comparison
2. PIE: Rat-running analysis, residential road usage
3. Comparative plots for both scenarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def setup_plot_style(fontsize=23):
    """Apply consistent plot styling parameters."""
    plt.rcParams.update({
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'savefig.dpi': 300,  # High resolution for publications
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'legend.fontsize': fontsize * 0.9,
        'legend.labelspacing': .3,
        'legend.columnspacing': .3,
        'legend.handletextpad': .1,
        'text.usetex': False,
        'mathtext.fontset': 'stix',
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    })


def plot_paya_lebar_comparison(
    baseline_file,
    closure_file,
    output_dir
):
    """
    Plot Paya Lebar lane closure impact.

    Args:
        baseline_file: Path to baseline CSV
        closure_file: Path to closure CSV
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("PAYA LEBAR VISUALIZATIONS")
    print("=" * 80)

    # Apply consistent plot styling
    setup_plot_style(fontsize=23)

    # Load data
    print("\nLoading data...")
    df_baseline = pd.read_csv(baseline_file)
    df_closure = pd.read_csv(closure_file)
    print(f"  ✓ Baseline: {len(df_baseline)} rows")
    print(f"  ✓ Closure: {len(df_closure)} rows")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), facecolor='white')

    # Convert time to minutes
    time_baseline = df_baseline['time_step'] / 60
    time_closure = df_closure['time_step'] / 60

    # Plot 1: Completed Trips
    ax1 = axes[0]
    ax1.plot(time_baseline, df_baseline['completed_trips'],
             label='Baseline (No Incident)', color='#2ecc71', linewidth=2.5)
    ax1.plot(time_closure, df_closure['completed_trips'],
             label='With Incident (Gaussian Bottleneck)', color='#e74c3c', linewidth=2.5, linestyle='--')
    ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Completed Trips', fontsize=12, fontweight='bold')
    ax1.set_title('Trip Completion: Baseline vs Incident', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)

    # Plot 2: Queue Length Over Time (NEW - IMPORTANT!)
    ax2 = axes[1]
    ax2.plot(time_baseline, df_baseline['total_queue_length'],
             label='Baseline (No Incident)', color='#2ecc71', linewidth=2.5)
    ax2.plot(time_closure, df_closure['total_queue_length'],
             label='With Incident', color='#e74c3c', linewidth=2.5, linestyle='--')
    ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Queue Length (vehicles)', fontsize=12, fontweight='bold')
    ax2.set_title('Queue Size Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.fill_between(time_closure, 0, df_closure['total_queue_length'],
                      color='#e74c3c', alpha=0.2, label='_nolegend_')

    # Plot 3: Network Congestion
    ax3 = axes[2]
    ax3.plot(time_baseline, df_baseline['congested_roads'],
             label='Baseline', color='#2ecc71', linewidth=2.5)
    ax3.plot(time_closure, df_closure['congested_roads'],
             label='With Incident', color='#e74c3c', linewidth=2.5, linestyle='--')
    ax3.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Congested Roads (count)', fontsize=12, fontweight='bold')
    ax3.set_title('Congested Roads Over Time', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=11, loc='upper right')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(left=0)

    # Plot 4: Mean Network Speed
    ax4 = axes[3]
    ax4.plot(time_baseline, df_baseline['mean_speed_kph'],
             label='Baseline', color='#2ecc71', linewidth=2.5)
    ax4.plot(time_closure, df_closure['mean_speed_kph'],
             label='With Incident', color='#e74c3c', linewidth=2.5, linestyle='--')
    ax4.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean Speed (km/h)', fontsize=12, fontweight='bold')
    ax4.set_title('Network Speed Degradation', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(left=0)

    plt.suptitle('Paya Lebar Localized Incident: IDM with Gaussian Bottleneck Impact',
                 fontsize=16, fontweight='bold', y=0.997)
    plt.tight_layout()

    output_file = output_dir / 'paya_lebar_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {output_file}")
    plt.close()

    # Create impact summary with delay comparison (NEW - IMPORTANT!)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

    # LEFT: Delay Comparison Bar Graph
    baseline_total_delay = df_baseline['total_delay'].iloc[-1]
    closure_total_delay = df_closure['total_delay'].iloc[-1]

    delays = [baseline_total_delay / 60.0, closure_total_delay / 60.0]  # Convert to minutes
    labels = ['Baseline\n(No Incident)', 'With Incident\n(Gaussian Bottleneck)']
    colors_delay = ['#2ecc71', '#e74c3c']

    bars_delay = ax1.bar(labels, delays, color=colors_delay, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Total Delay (vehicle-minutes)', fontsize=13, fontweight='bold')
    ax1.set_title('Delay Comparison', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(axis='both', labelsize=10)

    # Add value labels on bars (without "min" suffix)
    for bar, delay in zip(bars_delay, delays):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{delay:.1f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Add percentage increase annotation
    if baseline_total_delay > 0:
        delay_increase = ((closure_total_delay - baseline_total_delay) / baseline_total_delay) * 100
        ax1.text(0.5, max(delays) * 0.95, f'+{delay_increase:.1f}% increase',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                transform=ax1.transData)

    # RIGHT: Peak Queue Length Comparison (Grouped Bar Chart)
    baseline_peak_queue = df_baseline['total_queue_length'].max()
    closure_peak_queue = df_closure['total_queue_length'].max()

    # Prepare data for grouped bar chart (only Peak Queue Length)
    labels = ['Peak Queue Length']
    baseline_values = [baseline_peak_queue]
    incident_values = [closure_peak_queue]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline',
                    color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, incident_values, width, label='With Incident',
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax2.set_ylabel('Queue Length (vehicles)', fontsize=13, fontweight='bold')
    ax2.set_title('Peak Queue Length Comparison', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.tick_params(axis='y', labelsize=10)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_file = output_dir / 'paya_lebar_impact.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_file}")
    plt.close()


def visualize_all_urban_scenarios(
    paya_lebar_baseline,
    paya_lebar_closure,
    pie_baseline=None,  # Deprecated, PIE scenario removed
    pie_closure=None,
    pie_baseline_ratrun=None,
    pie_closure_ratrun=None,
    output_dir="output/urban_planning"
):
    """
    Generate Paya Lebar urban planning visualizations.

    Args:
        paya_lebar_baseline: Path to Paya Lebar baseline CSV
        paya_lebar_closure: Path to Paya Lebar closure CSV
        pie_baseline: Deprecated (PIE scenario removed)
        pie_closure: Deprecated (PIE scenario removed)
        pie_baseline_ratrun: Deprecated (PIE scenario removed)
        pie_closure_ratrun: Deprecated (PIE scenario removed)
        output_dir: Directory to save plots
    """
    print("=" * 80)
    print("URBAN PLANNING VISUALIZATION SUITE")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paya Lebar visualizations
    if Path(paya_lebar_baseline).exists() and Path(paya_lebar_closure).exists():
        plot_paya_lebar_comparison(
            paya_lebar_baseline,
            paya_lebar_closure,
            output_dir
        )
    else:
        print("\nWarning: Paya Lebar data files not found. Skipping Paya Lebar plots.")

    # PIE scenario removed - skip

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    # Default paths
    OUTPUT_DIR = Path(__file__).parent.parent / "output" / "urban_planning"

    PAYA_LEBAR_BASELINE = OUTPUT_DIR / "paya_lebar_baseline.csv"
    PAYA_LEBAR_CLOSURE = OUTPUT_DIR / "paya_lebar_closure.csv"

    PIE_BASELINE = OUTPUT_DIR / "pie_baseline.csv"
    PIE_CLOSURE = OUTPUT_DIR / "pie_closure.csv"
    PIE_BASELINE_RATRUN = OUTPUT_DIR / "pie_baseline_ratrunning.csv"
    PIE_CLOSURE_RATRUN = OUTPUT_DIR / "pie_closure_ratrunning.csv"

    # Generate visualizations
    visualize_all_urban_scenarios(
        paya_lebar_baseline=PAYA_LEBAR_BASELINE,
        paya_lebar_closure=PAYA_LEBAR_CLOSURE,
        pie_baseline=PIE_BASELINE,
        pie_closure=PIE_CLOSURE,
        pie_baseline_ratrun=PIE_BASELINE_RATRUN,
        pie_closure_ratrun=PIE_CLOSURE_RATRUN,
        output_dir=OUTPUT_DIR
    )
