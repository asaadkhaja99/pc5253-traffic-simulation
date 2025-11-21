"""
Critical Point Analysis Module

This module provides tools for analyzing critical transitions in traffic flow models
by comparing fundamental diagrams, detecting breakpoints, and computing susceptibility
metrics across different traffic models (Bando/OVM, IDM, NaSch) and actual data.

Academic reference:
- Breakpoint detection: Piecewise linear regression
- Susceptibility: Variance peak as early warning signal
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import osmnx as ox


@dataclass
class BreakpointResult:
    """Results from breakpoint detection analysis."""
    x_star: float          # Critical point location
    i_star: int            # Index of critical point
    slopes: tuple          # (slope_before, slope_after)
    intercepts: tuple      # (intercept_before, intercept_after)
    sse: float             # Sum of squared errors

    def __str__(self):
        return f"Breakpoint: x* = {self.x_star:.4f} (index {self.i_star})"


@dataclass
class SusceptibilityResult:
    """Results from susceptibility (variance peak) analysis."""
    x_peak: float          # Location of maximum variance
    variance_max: float    # Maximum variance value
    variance_series: np.ndarray  # Full variance series

    def __str__(self):
        return f"Variance peak: x = {self.x_peak:.4f}, χ² = {self.variance_max:.4f}"


@dataclass
class FundamentalDiagram:
    """Fundamental diagram data structure."""
    density: np.ndarray    # Density values (ρ)
    flow: np.ndarray       # Flow values (q)
    speed: np.ndarray      # Speed values (v)
    variance: np.ndarray   # Variance at each density
    model_name: str        # Name of traffic model
    unit_density: str      # Unit for density (e.g., "veh/m", "veh/cell")
    unit_flow: str         # Unit for flow
    unit_speed: str        # Unit for speed


# ============================================================================
# Core Analysis Functions
# ============================================================================

def estimate_breakpoint(x: np.ndarray, y: np.ndarray, min_frac: float = 0.15) -> Optional[BreakpointResult]:
    """
    Estimate breakpoint in data using piecewise linear regression.

    Finds the point that minimizes SSE when fitting two linear segments.

    Args:
        x: Independent variable (sorted)
        y: Dependent variable
        min_frac: Minimum fraction of data in each segment (default 0.15)

    Returns:
        BreakpointResult or None if estimation fails
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)

    if n < 6:
        return None

    kmin = max(2, int(min_frac * n))
    kmax = min(n - 2, n - kmin)

    best = None
    for k in range(kmin, kmax + 1):
        x1, y1 = x[:k], y[:k]
        x2, y2 = x[k:], y[k:]

        # Fit two linear segments
        A1 = np.vstack([np.ones_like(x1), x1]).T
        A2 = np.vstack([np.ones_like(x2), x2]).T

        try:
            # Segment 1: y = b2 + b1*x
            sol1 = np.linalg.lstsq(A1, y1, rcond=None)
            b2, b1 = sol1[0][0], sol1[0][1]

            # Segment 2: y = c2 + c1*x
            sol2 = np.linalg.lstsq(A2, y2, rcond=None)
            c2, c1 = sol2[0][0], sol2[0][1]

        except Exception:
            continue

        # Calculate SSE
        y1hat = b2 + b1 * x1
        y2hat = c2 + c1 * x2
        sse = ((y1 - y1hat)**2).sum() + ((y2 - y2hat)**2).sum()

        if (best is None) or (sse < best.sse):
            best = BreakpointResult(
                x_star=float(x[k]),
                i_star=k,
                slopes=(float(b1), float(c1)),
                intercepts=(float(b2), float(c2)),
                sse=float(sse)
            )

    return best


def susceptibility_peak(x: np.ndarray, y: np.ndarray, window: int = 3) -> SusceptibilityResult:
    """
    Find variance peak (susceptibility) as early warning signal.

    Computes sliding window variance and identifies the maximum,
    which indicates heightened sensitivity near critical transition.

    Args:
        x: Independent variable
        y: Dependent variable
        window: Half-width of sliding window (default 3)

    Returns:
        SusceptibilityResult with peak location and variance series
    """
    x = np.asarray(x)
    y = np.asarray(y)
    varv = []

    for i in range(len(y)):
        lo = max(0, i - window)
        hi = min(len(y), i + window + 1)
        if hi - lo > 2:
            varv.append(np.var(y[lo:hi], ddof=1))
        else:
            varv.append(0.0)

    varv = np.asarray(varv)
    i_star = int(np.nanargmax(varv)) if len(varv) > 0 else 0

    return SusceptibilityResult(
        x_peak=float(x[i_star]) if len(x) > 0 else np.nan,
        variance_max=float(varv[i_star]) if len(varv) > 0 else np.nan,
        variance_series=varv
    )


def bin_xy(x: np.ndarray, y: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin data into quantiles for smoother fundamental diagram.

    Args:
        x: Independent variable
        y: Dependent variable
        bins: Number of bins (default 20)

    Returns:
        Tuple of (x_centers, y_means, y_variances)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x = x[ok]
    y = y[ok]

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    qs = np.quantile(x, np.linspace(0, 1, bins + 1))
    xc = []
    yc = []
    yv = []

    for i in range(bins):
        if i == bins - 1:
            mask = (x >= qs[i]) & (x <= qs[i + 1])
        else:
            mask = (x >= qs[i]) & (x < qs[i + 1])

        if mask.sum() >= 5:
            xc.append(x[mask].mean())
            yc.append(y[mask].mean())
            yv.append(y[mask].var(ddof=1))

    return np.array(xc), np.array(yc), np.array(yv)


# ============================================================================
# Observed Data Processing (LTA Traffic Speed Bands)
# ============================================================================

def load_osm_network(place: str = "Singapore", network_type: str = "drive") -> gpd.GeoDataFrame:
    """
    Load OpenStreetMap road network using OSMnx.

    Args:
        place: Place name for OSMnx query
        network_type: Type of network (default "drive")

    Returns:
        GeoDataFrame of edges with speed estimates
    """
    ox.settings.use_cache = True
    G = ox.graph_from_place(place, network_type=network_type)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    edges_gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    if not {'u', 'v', 'key'}.issubset(edges_gdf.columns):
        edges_gdf = edges_gdf.reset_index()

    edge_cols = [c for c in ['u', 'v', 'key', 'geometry', 'length', 'highway', 'speed_kph']
                 if c in edges_gdf.columns]
    edges = edges_gdf[edge_cols].copy().set_crs(4326)

    return edges


def load_tsb_data(data_path: str, aggregate: bool = True) -> pd.DataFrame:
    """
    Load LTA Traffic Speed Band (TSB) data from CSV archive.

    Args:
        data_path: Path to TSB CSV file or directory
        aggregate: If True, aggregate multiple files by averaging

    Returns:
        DataFrame with TSB data
    """
    data_path = Path(data_path)

    if data_path.is_file():
        # Single file
        tsb = pd.read_csv(data_path)
    elif data_path.is_dir():
        # Multiple files - aggregate
        files = sorted(data_path.glob("tsb_archive_*.csv"))
        if not files:
            raise ValueError(f"No TSB files found in {data_path}")

        dfs = []
        for f in files:
            df = pd.read_csv(f)
            dfs.append(df)

        if aggregate:
            # Average speeds across all timestamps for each link
            tsb = pd.concat(dfs, ignore_index=True)
            tsb = tsb.groupby('LinkID', as_index=False).agg({
                'RoadName': 'first',
                'RoadCategory': 'first',
                'SpeedBand': 'mean',
                'MinimumSpeed': 'mean',
                'MaximumSpeed': 'mean',
                'StartLon': 'first',
                'StartLat': 'first',
                'EndLon': 'first',
                'EndLat': 'first',
                'speed_kph_obs': 'mean'
            })
        else:
            tsb = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Invalid path: {data_path}")

    # Ensure numeric columns
    for c in ['StartLon', 'StartLat', 'EndLon', 'EndLat', 'speed_kph_obs']:
        if c in tsb.columns:
            tsb[c] = pd.to_numeric(tsb[c], errors='coerce')

    return tsb


def match_tsb_to_osm(edges: gpd.GeoDataFrame, tsb: pd.DataFrame, buffer_m: float = 25) -> gpd.GeoDataFrame:
    """
    Match TSB observations to OSM edges using spatial join.

    Args:
        edges: OSM edges GeoDataFrame
        tsb: TSB data DataFrame
        buffer_m: Buffer distance in meters for matching

    Returns:
        Edges with observed speeds matched
    """
    from shapely.geometry import LineString

    # Create TSB geometries
    tsb = tsb.dropna(subset=['StartLon', 'StartLat', 'EndLon', 'EndLat']).copy()
    tsb['geometry'] = tsb.apply(
        lambda r: LineString([(r['StartLon'], r['StartLat']), (r['EndLon'], r['EndLat'])]),
        axis=1
    )
    tsb_g = gpd.GeoDataFrame(tsb, geometry='geometry', crs='EPSG:4326')

    # Project to metric CRS for buffering
    edges_3414 = edges.to_crs(3414)
    tsb_buf = tsb_g.to_crs(3414).copy()
    tsb_buf['geometry'] = tsb_buf.geometry.buffer(buffer_m)

    # Spatial join
    hit = gpd.sjoin(edges_3414, tsb_buf[['geometry', 'speed_kph_obs']],
                    how='inner', predicate='intersects')
    edge_speed = hit.groupby(['u', 'v', 'key'], as_index=False)['speed_kph_obs'].mean()

    edges_obs = edges_3414.merge(edge_speed, on=['u', 'v', 'key'], how='left').to_crs(4326)

    # Calculate density proxy: rho_hat = 1 - v_obs/v_ff
    edges_obs['chi'] = edges_obs['speed_kph_obs'] / edges_obs['speed_kph']
    edges_obs['rho_hat'] = (1 - edges_obs['chi']).clip(0, 1)

    return edges_obs


def compute_observed_fd(edges_obs: gpd.GeoDataFrame, bins: int = 24) -> FundamentalDiagram:
    """
    Compute fundamental diagram from observed data.

    Args:
        edges_obs: Edges with observed speeds
        bins: Number of bins for aggregation

    Returns:
        FundamentalDiagram object
    """
    observed = edges_obs.dropna(subset=['rho_hat', 'chi']).copy()

    if len(observed) == 0:
        return FundamentalDiagram(
            density=np.array([]),
            flow=np.array([]),
            speed=np.array([]),
            variance=np.array([]),
            model_name="Observed (LTA)",
            unit_density="proxy ρ̂",
            unit_flow="proxy χ",
            unit_speed="km/h"
        )

    x_obs, y_obs, v_obs = bin_xy(observed['rho_hat'].values, observed['chi'].values, bins=bins)

    return FundamentalDiagram(
        density=x_obs,
        flow=y_obs,  # Using chi as flow proxy
        speed=y_obs,  # Using chi as speed proxy
        variance=v_obs,
        model_name="Observed (LTA)",
        unit_density="proxy ρ̂",
        unit_flow="proxy χ",
        unit_speed="normalized"
    )


# ============================================================================
# Simulation Model Runners
# ============================================================================

def run_bando_sweep(
    densities: np.ndarray,
    L: float = 1000.0,
    alpha: float = 1.0,
    v0: float = 30.0,
    h0: float = 25.0,
    delta: float = 8.0,
    dt: float = 0.2,
    T: float = 600.0,
    warm: float = 200.0,
    seed: int = 1
) -> FundamentalDiagram:
    """
    Run density sweep for Bando/OVM model.

    Args:
        densities: Array of densities to test (vehicles/meter)
        L: Ring length (meters)
        alpha: Sensitivity parameter
        v0: Free-flow velocity (m/s)
        h0: Headway parameter (m)
        delta: Transition smoothness (m)
        dt: Time step (s)
        T: Total simulation time (s)
        warm: Warmup time (s)
        seed: Random seed

    Returns:
        FundamentalDiagram with Bando results
    """
    rhos = []
    flows = []
    speeds = []

    for rho in densities:
        N = int(round(rho * L))
        if N < 2:
            continue

        rng = np.random.default_rng(seed)
        x = np.linspace(0, L * (N - 1) / N, N)
        v = v0 * np.ones(N)
        x += rng.normal(0, 0.5, size=N)

        steps = int(T / dt)
        warm_steps = int(warm / dt)

        def V(h):
            return v0 * (np.tanh((h - h0) / delta) + np.tanh(h0 / delta))

        flow_samples = []
        for t in range(steps):
            order = np.argsort(x)
            x = x[order]
            v = v[order]
            h = np.diff(np.r_[x, x[0] + L])
            a = alpha * (V(h) - v)
            v = np.clip(v + a * dt, 0, v0 * 1.5)
            x = (x + v * dt) % L

            if t >= warm_steps:
                flow_samples.append(v.mean())

        vbar = np.mean(flow_samples) if flow_samples else v.mean()
        q = rho * vbar

        rhos.append(rho)
        flows.append(q)
        speeds.append(vbar)

    return FundamentalDiagram(
        density=np.array(rhos),
        flow=np.array(flows),
        speed=np.array(speeds),
        variance=np.zeros_like(flows),  # Single run, no variance
        model_name="Bando/OVM",
        unit_density="veh/m",
        unit_flow="veh/s",
        unit_speed="m/s"
    )


def run_idm_sweep(
    densities: np.ndarray,
    L: float = 1000.0,
    v0: float = 30.0,
    s0: float = 2.0,
    T: float = 1.5,
    a_max: float = 1.0,
    b: float = 2.0,
    delta: float = 4.0,
    dt: float = 0.1,
    sim_time: float = 600.0,
    warm: float = 200.0,
    seed: int = 1
) -> FundamentalDiagram:
    """
    Run density sweep for IDM (Intelligent Driver Model).

    Args:
        densities: Array of densities to test (vehicles/meter)
        L: Ring length (meters)
        v0: Desired velocity (m/s)
        s0: Minimum gap (m)
        T: Safe time headway (s)
        a_max: Maximum acceleration (m/s²)
        b: Comfortable deceleration (m/s²)
        delta: Acceleration exponent
        dt: Time step (s)
        sim_time: Total simulation time (s)
        warm: Warmup time (s)
        seed: Random seed

    Returns:
        FundamentalDiagram with IDM results
    """
    rhos = []
    flows = []
    speeds = []

    for rho in densities:
        N = int(round(rho * L))
        if N < 2:
            continue

        rng = np.random.default_rng(seed)
        x = np.linspace(0, L * (N - 1) / N, N)
        v = v0 * np.ones(N) * 0.8  # Start at 80% free-flow
        x += rng.normal(0, 0.5, size=N)

        steps = int(sim_time / dt)
        warm_steps = int(warm / dt)

        flow_samples = []
        for t in range(steps):
            order = np.argsort(x)
            x = x[order]
            v = v[order]

            # Calculate gaps
            x_lead = np.roll(x, -1)
            x_lead[-1] += L
            s = x_lead - x

            v_lead = np.roll(v, -1)
            dv = v - v_lead

            # IDM acceleration
            s_star = s0 + v * T + (v * dv) / (2 * np.sqrt(a_max * b))
            a = a_max * (1 - (v / v0)**delta - (s_star / s)**2)

            v = np.clip(v + a * dt, 0, v0 * 1.2)
            x = (x + v * dt) % L

            if t >= warm_steps:
                flow_samples.append(v.mean())

        vbar = np.mean(flow_samples) if flow_samples else v.mean()
        q = rho * vbar

        rhos.append(rho)
        flows.append(q)
        speeds.append(vbar)

    return FundamentalDiagram(
        density=np.array(rhos),
        flow=np.array(flows),
        speed=np.array(speeds),
        variance=np.zeros_like(flows),
        model_name="IDM",
        unit_density="veh/m",
        unit_flow="veh/s",
        unit_speed="m/s"
    )


def run_nasch_sweep(
    densities: np.ndarray,
    L: int = 600,
    v_max: int = 5,
    p_slow: float = 0.3,
    steps: int = 2000,
    warm: int = 500,
    n_seeds: int = 5
) -> FundamentalDiagram:
    """
    Run density sweep for NaSch cellular automaton.

    Args:
        densities: Array of densities to test (vehicles/cell)
        L: Ring length (cells)
        v_max: Maximum velocity (cells/tick)
        p_slow: Randomization probability
        steps: Total simulation steps
        warm: Warmup steps
        n_seeds: Number of random seeds for averaging

    Returns:
        FundamentalDiagram with NaSch results
    """
    def nasch_ring(L, rho, vmax, p, steps, warm, seed):
        """Single NaSch simulation on ring."""
        rng = np.random.default_rng(seed)
        n = int(round(rho * L))
        road = -np.ones(L, dtype=int)
        pos = rng.choice(L, size=n, replace=False)
        road[pos] = rng.integers(0, vmax + 1, size=n)

        flows = []
        for t in range(steps):
            # Acceleration
            road[road >= 0] = np.minimum(road[road >= 0] + 1, vmax)

            # Braking
            occ = np.where(road >= 0)[0]
            if len(occ) == 0:
                flows.append(0)
                continue

            next_pos = np.roll(occ, -1)
            gaps = (next_pos - occ - 1) % L
            for i, posi in enumerate(occ):
                road[posi] = min(road[posi], gaps[i])

            # Randomization
            mask = (road >= 1) & (rng.random(L) < p)
            road[mask] -= 1

            # Movement
            new = -np.ones(L, dtype=int)
            occ = np.where(road >= 0)[0]
            newpos = (occ + road[occ]) % L
            new[newpos] = road[occ]
            passed = ((occ + road[occ]) % L < occ).sum()
            flows.append(passed)
            road = new

        flows = np.array(flows)
        q = flows[warm:].mean()
        vbar = q / rho if rho > 0 else 0.0
        return q, vbar

    rhos = []
    q_means = []
    q_vars = []
    v_means = []

    for rho in densities:
        qs = []
        vs = []
        for s in range(n_seeds):
            q, v = nasch_ring(L, rho, v_max, p_slow, steps, warm, seed=100 + s)
            qs.append(q)
            vs.append(v)

        qs = np.array(qs)
        vs = np.array(vs)
        rhos.append(rho)
        q_means.append(qs.mean())
        q_vars.append(qs.var(ddof=1))
        v_means.append(vs.mean())

    return FundamentalDiagram(
        density=np.array(rhos),
        flow=np.array(q_means),
        speed=np.array(v_means),
        variance=np.array(q_vars),
        model_name="NaSch CA",
        unit_density="veh/cell",
        unit_flow="veh/tick",
        unit_speed="cells/tick"
    )


# ============================================================================
# Comparison and Reporting
# ============================================================================

def analyze_fd(fd: FundamentalDiagram) -> Dict:
    """
    Perform complete analysis on fundamental diagram.

    Args:
        fd: FundamentalDiagram to analyze

    Returns:
        Dictionary with breakpoint, susceptibility, and capacity metrics
    """
    if len(fd.density) < 6:
        return {
            'model': fd.model_name,
            'breakpoint': None,
            'susceptibility': None,
            'capacity': None,
            'critical_density': None
        }

    # Breakpoint detection
    bp = estimate_breakpoint(fd.density, fd.flow)

    # Susceptibility (variance peak)
    sus = susceptibility_peak(fd.density, fd.flow)

    # Capacity (max flow)
    i_max = np.argmax(fd.flow)
    capacity = fd.flow[i_max]
    critical_density = fd.density[i_max]

    return {
        'model': fd.model_name,
        'breakpoint': bp,
        'susceptibility': sus,
        'capacity': float(capacity),
        'critical_density': float(critical_density),
        'unit_density': fd.unit_density,
        'unit_flow': fd.unit_flow
    }


def create_comparison_table(analyses: List[Dict]) -> pd.DataFrame:
    """
    Create comparison table from multiple analyses.

    Args:
        analyses: List of analysis dictionaries from analyze_fd()

    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    for a in analyses:
        row = {
            'Model': a['model'],
            'Critical ρ (capacity)': f"{a['critical_density']:.4f}" if a['critical_density'] else "N/A",
            'Breakpoint ρ*': f"{a['breakpoint'].x_star:.4f}" if a['breakpoint'] else "N/A",
            'Variance peak ρ': f"{a['susceptibility'].x_peak:.4f}" if a['susceptibility'] else "N/A",
            'Max flow': f"{a['capacity']:.4f}" if a['capacity'] else "N/A",
            'Unit (ρ)': a['unit_density'],
            'Unit (q)': a['unit_flow']
        }
        rows.append(row)

    return pd.DataFrame(rows)
