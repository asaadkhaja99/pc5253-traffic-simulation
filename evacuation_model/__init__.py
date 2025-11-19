"""
Evacuation Model Package

Agent-based evacuation simulation using NaSch traffic dynamics on real road networks.

Organized structure:
- core: Base model components and utilities
- scenarios: Different evacuation strategies
- analysis: Metrics and analysis functions
- visualization: Plotting and visualization
- runners: Orchestration scripts
- docs: Documentation
"""

# Core components
from .core import (
    EvacuationConfig,
    EvacuationMetrics,
    EvacueeAgent,
    EvacuationRoadAgent,
    EvacuationModel,
    nasch_step,
    get_origin_nodes,
    setup_plot_style,
    setup_high_res_plot_style,
    COLORS,
    LINE_STYLES,
)

# Scenarios
from .scenarios import (
    run_simultaneous_evacuation,
    run_staged_evacuation,
    run_contraflow_evacuation,
)

# Analysis
from .analysis import analyze_all_scenarios

# Visualization
from .visualization import (
    visualize_all_scenarios,
    plot_staged_vs_simultaneous,
    plot_contraflow_comparison,
    create_intervention_effectiveness_plot,
    create_evacuation_zone_map,
)

# Runners
from .runners import run_complete_study

__all__ = [
    # Core
    'EvacuationConfig',
    'EvacuationMetrics',
    'EvacueeAgent',
    'EvacuationRoadAgent',
    'EvacuationModel',
    'nasch_step',
    'get_origin_nodes',
    'setup_plot_style',
    'setup_high_res_plot_style',
    'COLORS',
    'LINE_STYLES',
    # Scenarios
    'run_simultaneous_evacuation',
    'run_staged_evacuation',
    'run_contraflow_evacuation',
    # Analysis
    'analyze_all_scenarios',
    # Visualization
    'visualize_all_scenarios',
    'plot_staged_vs_simultaneous',
    'plot_contraflow_comparison',
    'create_intervention_effectiveness_plot',
    'create_evacuation_zone_map',
    # Runners
    'run_complete_study',
]

__version__ = '1.0.0'
