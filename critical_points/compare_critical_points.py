"""
Master Critical Point Comparison Script

Compares critical points across all traffic models (Observed, Bando, IDM, NaSch)
and generates academic-quality results for research publication.

Usage:
    python compare_critical_points.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]

Outputs:
    - comparison_table.csv: Numerical comparison of critical points
    - fundamental_diagrams.png: Side-by-side FD plots
    - critical_analysis.md: Markdown report
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

from critical_point_analysis import (
    load_osm_network,
    load_tsb_data,
    match_tsb_to_osm,
    compute_observed_fd,
    run_bando_sweep,
    run_idm_sweep,
    run_nasch_sweep,
    analyze_fd,
    create_comparison_table
)

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Density ranges for each model
BANDO_DENSITIES = np.linspace(0.02, 0.25, 30)  # veh/m
IDM_DENSITIES = np.linspace(0.02, 0.25, 30)    # veh/m
NASCH_DENSITIES = np.linspace(0.02, 0.8, 30)   # veh/cell

# Model parameters (highway-specific)
BANDO_PARAMS = {
    'L': 1000.0,
    'alpha': 1.0,
    'v0': 30.0,   # m/s (~108 km/h highway speed)
    'h0': 25.0,   # m
    'delta': 8.0,
    'dt': 0.2,
    'T': 600.0,
    'warm': 200.0
}

IDM_PARAMS = {
    'L': 1000.0,
    'v0': 30.0,      # m/s
    's0': 2.0,       # m
    'T': 1.5,        # s
    'a_max': 1.0,    # m/s²
    'b': 2.0,        # m/s²
    'delta': 4.0,
    'dt': 0.1,
    'sim_time': 600.0,
    'warm': 200.0
}

NASCH_PARAMS = {
    'L': 600,
    'v_max': 5,
    'p_slow': 0.3,
    'steps': 2000,
    'warm': 500,
    'n_seeds': 5
}

# Plot styling
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_complete_analysis(data_dir: str, output_dir: str):
    """
    Run complete critical point analysis for all models.

    Args:
        data_dir: Directory containing TSB data
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("CRITICAL POINT ANALYSIS - Traffic Flow Models")
    print("=" * 80)

    # ========================================================================
    # 1. Load and Process Observed Data
    # ========================================================================
    print("\n[1/4] Loading observed data (LTA Traffic Speed Bands)...")

    try:
        edges = load_osm_network("Singapore", "drive")
        print(f"  - Loaded {len(edges):,} OSM edges")

        tsb = load_tsb_data(data_dir, aggregate=True)
        print(f"  - Loaded {len(tsb):,} TSB observations (aggregated)")

        edges_obs = match_tsb_to_osm(edges, tsb, buffer_m=25)
        matched = edges_obs.dropna(subset=['rho_hat', 'chi'])
        print(f"  - Matched {len(matched):,} edges with observations")

        fd_obs = compute_observed_fd(edges_obs, bins=24)
        print(f"  - Computed FD with {len(fd_obs.density)} density bins")

    except Exception as e:
        print(f"  ⚠ Warning: Could not load observed data: {e}")
        print("  - Continuing with simulations only...")
        fd_obs = None

    # ========================================================================
    # 2. Run Bando/OVM Simulation
    # ========================================================================
    print("\n[2/4] Running Bando/OVM simulations...")
    print(f"  - Density range: {BANDO_DENSITIES.min():.3f} - {BANDO_DENSITIES.max():.3f} veh/m")
    print(f"  - Points: {len(BANDO_DENSITIES)}")

    fd_bando = run_bando_sweep(BANDO_DENSITIES, **BANDO_PARAMS)
    print(f"  ✓ Completed {len(fd_bando.density)} simulations")

    # ========================================================================
    # 3. Run IDM Simulation
    # ========================================================================
    print("\n[3/4] Running IDM simulations...")
    print(f"  - Density range: {IDM_DENSITIES.min():.3f} - {IDM_DENSITIES.max():.3f} veh/m")
    print(f"  - Points: {len(IDM_DENSITIES)}")

    fd_idm = run_idm_sweep(IDM_DENSITIES, **IDM_PARAMS)
    print(f"  ✓ Completed {len(fd_idm.density)} simulations")

    # ========================================================================
    # 4. Run NaSch CA Simulation
    # ========================================================================
    print("\n[4/4] Running NaSch CA simulations...")
    print(f"  - Density range: {NASCH_DENSITIES.min():.3f} - {NASCH_DENSITIES.max():.3f} veh/cell")
    print(f"  - Points: {len(NASCH_DENSITIES)}")
    print(f"  - Seeds per point: {NASCH_PARAMS['n_seeds']}")

    fd_nasch = run_nasch_sweep(NASCH_DENSITIES, **NASCH_PARAMS)
    print(f"  ✓ Completed {len(fd_nasch.density)} simulations")

    # ========================================================================
    # 5. Analyze All Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("CRITICAL POINT DETECTION")
    print("=" * 80)

    analyses = []

    if fd_obs is not None and len(fd_obs.density) > 0:
        print("\nObserved (LTA):")
        analysis_obs = analyze_fd(fd_obs)
        analyses.append(analysis_obs)
        if analysis_obs['breakpoint']:
            print(f"  - {analysis_obs['breakpoint']}")
        if analysis_obs['susceptibility']:
            print(f"  - {analysis_obs['susceptibility']}")
        print(f"  - Capacity: {analysis_obs['capacity']:.4f} {analysis_obs['unit_flow']}")

    print("\nBando/OVM:")
    analysis_bando = analyze_fd(fd_bando)
    analyses.append(analysis_bando)
    if analysis_bando['breakpoint']:
        print(f"  - {analysis_bando['breakpoint']}")
    if analysis_bando['susceptibility']:
        print(f"  - {analysis_bando['susceptibility']}")
    print(f"  - Capacity: {analysis_bando['capacity']:.4f} {analysis_bando['unit_flow']}")

    print("\nIDM:")
    analysis_idm = analyze_fd(fd_idm)
    analyses.append(analysis_idm)
    if analysis_idm['breakpoint']:
        print(f"  - {analysis_idm['breakpoint']}")
    if analysis_idm['susceptibility']:
        print(f"  - {analysis_idm['susceptibility']}")
    print(f"  - Capacity: {analysis_idm['capacity']:.4f} {analysis_idm['unit_flow']}")

    print("\nNaSch CA:")
    analysis_nasch = analyze_fd(fd_nasch)
    analyses.append(analysis_nasch)
    if analysis_nasch['breakpoint']:
        print(f"  - {analysis_nasch['breakpoint']}")
    if analysis_nasch['susceptibility']:
        print(f"  - {analysis_nasch['susceptibility']}")
    print(f"  - Capacity: {analysis_nasch['capacity']:.4f} {analysis_nasch['unit_flow']}")

    # ========================================================================
    # 6. Generate Comparison Table
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    comparison_df = create_comparison_table(analyses)
    print("\n" + comparison_df.to_string(index=False))

    # Save to CSV
    csv_path = output_path / "comparison_table.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to: {csv_path}")

    # ========================================================================
    # 7. Generate Plots
    # ========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Observed (if available)
    if fd_obs is not None and len(fd_obs.density) > 0:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(fd_obs.density, fd_obs.flow, 'o-', color='#1f77b4', label='Observed FD')
        if analysis_obs['breakpoint']:
            ax1.axvline(analysis_obs['breakpoint'].x_star, linestyle='--',
                       color='red', label=f"Breakpoint: {analysis_obs['breakpoint'].x_star:.4f}")
        if analysis_obs['susceptibility']:
            ax1.axvline(analysis_obs['susceptibility'].x_peak, linestyle=':',
                       color='orange', label=f"Var peak: {analysis_obs['susceptibility'].x_peak:.4f}")
        ax1.set_xlabel(f"Density ({fd_obs.unit_density})")
        ax1.set_ylabel(f"Flow ({fd_obs.unit_flow})")
        ax1.set_title(f"{fd_obs.model_name} - Fundamental Diagram")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Bando/OVM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fd_bando.density, fd_bando.flow, 'o-', color='#2ca02c', label='Bando FD')
    if analysis_bando['breakpoint']:
        ax2.axvline(analysis_bando['breakpoint'].x_star, linestyle='--',
                   color='red', label=f"Breakpoint: {analysis_bando['breakpoint'].x_star:.4f}")
    if analysis_bando['susceptibility']:
        ax2.axvline(analysis_bando['susceptibility'].x_peak, linestyle=':',
                   color='orange', label=f"Var peak: {analysis_bando['susceptibility'].x_peak:.4f}")
    ax2.set_xlabel(f"Density ({fd_bando.unit_density})")
    ax2.set_ylabel(f"Flow ({fd_bando.unit_flow})")
    ax2.set_title(f"{fd_bando.model_name} - Fundamental Diagram")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: IDM
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(fd_idm.density, fd_idm.flow, 'o-', color='#ff7f0e', label='IDM FD')
    if analysis_idm['breakpoint']:
        ax3.axvline(analysis_idm['breakpoint'].x_star, linestyle='--',
                   color='red', label=f"Breakpoint: {analysis_idm['breakpoint'].x_star:.4f}")
    if analysis_idm['susceptibility']:
        ax3.axvline(analysis_idm['susceptibility'].x_peak, linestyle=':',
                   color='orange', label=f"Var peak: {analysis_idm['susceptibility'].x_peak:.4f}")
    ax3.set_xlabel(f"Density ({fd_idm.unit_density})")
    ax3.set_ylabel(f"Flow ({fd_idm.unit_flow})")
    ax3.set_title(f"{fd_idm.model_name} - Fundamental Diagram")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: NaSch CA
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(fd_nasch.density, fd_nasch.flow, 'o-', color='#d62728', label='NaSch FD')
    if analysis_nasch['breakpoint']:
        ax4.axvline(analysis_nasch['breakpoint'].x_star, linestyle='--',
                   color='red', label=f"Breakpoint: {analysis_nasch['breakpoint'].x_star:.4f}")
    if analysis_nasch['susceptibility']:
        ax4.axvline(analysis_nasch['susceptibility'].x_peak, linestyle=':',
                   color='orange', label=f"Var peak: {analysis_nasch['susceptibility'].x_peak:.4f}")
    ax4.set_xlabel(f"Density ({fd_nasch.unit_density})")
    ax4.set_ylabel(f"Flow ({fd_nasch.unit_flow})")
    ax4.set_title(f"{fd_nasch.model_name} - Fundamental Diagram")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Critical Point Analysis - All Traffic Models", fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    fig_path = output_path / "fundamental_diagrams.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plots to: {fig_path}")



# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Critical Point Analysis for Traffic Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default data directory
    python compare_critical_points.py

    # Specify custom data directory
    python compare_critical_points.py --data-dir /path/to/tsb/data

    # Specify output directory
    python compare_critical_points.py --output-dir results/
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/',
        help='Directory containing TSB CSV files (default: data/)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/critical_analysis/',
        help='Output directory for results (default: output/critical_analysis/)'
    )

    args = parser.parse_args()

    run_complete_analysis(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
