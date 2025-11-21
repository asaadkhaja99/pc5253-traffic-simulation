"""
Urban Planning Simulation Base Module

Extends the evacuation model framework for urban planning scenarios:
1. Localized incidents (lane closures, accidents)
"""

import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
import mesa
import mesa_geo as mg
from shapely.geometry import Point, LineString
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import deque
import sys
from pathlib import Path

# Import from evacuation model (for NaSch fallback)
sys.path.insert(0, str(Path(__file__).parent.parent / 'evacuation_model' / 'core'))
from evacuation_base import (
    nasch_step,
    EvacueeAgent,
    EvacuationRoadAgent,
    EvacuationMetrics
)

# Import IDM road agent (imported here to avoid circular dependency issues)
# Will be imported in _initialize_roads when needed


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UrbanPlanningConfig:
    """Configuration for urban planning scenarios."""
    # Network bounds
    bbox_north: float = 1.3700  # Expanded to cover Paya Lebar
    bbox_south: float = 1.2800
    bbox_east: float = 103.9000
    bbox_west: float = 103.8200

    # Simulation parameters
    num_vehicles: int = 1000  # Normal traffic flow
    delta_t: float = 1.0
    max_steps: int = 7200  # 2 hour simulation (ensures all vehicles complete even with incidents)

    # Traffic demand
    demand_level: float = 0.5  # 0-1 scale, moderate traffic

    # Traffic model selection
    use_idm: bool = True  # Use IDM instead of NaSch (recommended for urban planning)

    # NaSch parameters (legacy)
    cell_length_m: float = 7.5

    # Random seed
    seed: int = 42


@dataclass
class DisruptionConfig:
    """Configuration for network disruption."""
    disruption_type: str  # 'lane_closure', 'road_closure', 'accident'
    affected_edges: List[Tuple[int, int, int]]  # List of (u, v, key) tuples
    start_time: int = 0  # When disruption starts
    duration: int = None  # None = permanent
    capacity_reduction: float = 0.8  # For lane closures (0-1, NaSch only)

    # For localized incidents
    location_name: str = ""  # e.g., "Paya Lebar"
    location_coords: Tuple[float, float] = None  # (lat, lon)

    # IDM bottleneck parameters (Gaussian bottleneck factor, Eq. 17)
    bottleneck_epsilon: float = 0.8  # Strength (0-1), 0.8 = 80% capacity reduction
    bottleneck_sigma: float = 50.0  # Spatial spread (meters)
    incident_position_m: float = None  # Position along road (m), None = midpoint


# =============================================================================
# Extended Road Agent with Disruption Support
# =============================================================================

class UrbanRoadAgent(EvacuationRoadAgent):
    """
    Road agent with support for disruptions (lane closures, road closures).

    Extends EvacuationRoadAgent with:
    - Capacity reduction for lane closures
    - Complete blockage for road closures
    - Queue tracking for spillback analysis
    - Vehicle counting for rat-running detection
    """

    def __init__(self, model, geometry, crs, edge_tuple, length_m, speed_kph, highway_type):
        super().__init__(model, geometry, crs, edge_tuple, length_m, speed_kph, highway_type)

        # Disruption state
        self.is_disrupted = False
        self.is_blocked = False  # Complete closure
        self.original_capacity = self.num_cells
        self.disruption_start = None
        self.disruption_end = None

        # Traffic metrics
        self.queue_length = 0  # Number of queued vehicles
        self.total_vehicles_passed = 0  # Cumulative throughput
        self.is_residential = highway_type in ['residential', 'tertiary', 'unclassified']
        self.baseline_flow = 0  # Normal traffic flow (for rat-run detection)

    def apply_lane_closure(self, capacity_reduction=0.5, start_time=0, duration=None):
        """
        Apply lane closure (partial capacity reduction).

        Args:
            capacity_reduction: Fraction of capacity to remove (0-1)
            start_time: Simulation step when closure starts
            duration: Duration in steps (None = permanent)
        """
        self.is_disrupted = True
        self.disruption_start = start_time
        self.disruption_end = start_time + duration if duration else None

        # Reduce capacity
        new_capacity = int(self.original_capacity * (1 - capacity_reduction))
        new_capacity = max(1, new_capacity)  # At least 1 cell

        # Shrink road array
        if len(self.road) > new_capacity:
            self.road = self.road[:new_capacity]
        self.num_cells = new_capacity

    def apply_road_closure(self, start_time=0, duration=None):
        """
        Apply complete road closure (no vehicles can pass).

        Args:
            start_time: Simulation step when closure starts
            duration: Duration in steps (None = permanent)
        """
        self.is_disrupted = True
        self.is_blocked = True
        self.disruption_start = start_time
        self.disruption_end = start_time + duration if duration else None

        # Prevent all entry
        self.num_cells = 0
        self.road = np.array([], dtype=np.int16)

    def remove_disruption(self):
        """Remove disruption and restore normal capacity."""
        self.is_disrupted = False
        self.is_blocked = False

        # Restore original capacity
        self.num_cells = self.original_capacity
        self.road = -np.ones(self.num_cells, dtype=np.int16)

    def try_insert_from_queue(self):
        """Override to handle blocked roads."""
        if self.is_blocked:
            # No vehicles can enter
            return

        # Normal insertion logic
        super().try_insert_from_queue()

    def step(self):
        """Update road state and check for disruption timing."""
        # Check if disruption should end
        if self.is_disrupted and self.disruption_end is not None:
            if self.model.step_count >= self.disruption_end:
                self.remove_disruption()

        # Update queue length
        self.queue_length = len(self.entry_queue)

        # Normal step
        super().step()


# =============================================================================
# Urban Traffic Agent with Origin-Destination
# =============================================================================

class UrbanVehicleAgent(EvacueeAgent):
    """
    Vehicle agent for normal urban traffic.

    Similar to EvacueeAgent but represents daily commuters with:
    - Origin-destination pairs
    - Route choice based on shortest path
    - Ability to reroute when encountering congestion
    """

    def __init__(self, model, geometry, crs, agent_id, origin_node, destination_node, route):
        super().__init__(model, geometry, crs, agent_id, destination_node, route)
        self.origin_node = origin_node

        # Trip completion (proper urban traffic terminology)
        self._trip_completed = False
        self._completion_time = None
        self.trip_time = None

        # Legacy evacuation compatibility (synced via properties)
        self.evacuated = False
        self.evacuation_time = None

        self.rerouted_count = 0

    @property
    def trip_completed(self):
        return self._trip_completed

    @trip_completed.setter
    def trip_completed(self, value):
        self._trip_completed = value
        self.evacuated = value  # Keep legacy attribute synced

    @property
    def completion_time(self):
        return self._completion_time

    @completion_time.setter
    def completion_time(self, value):
        self._completion_time = value
        self.evacuation_time = value  # Keep legacy attribute synced
        if value is not None and hasattr(self, 'route_index'):
            self.trip_time = value

    def step(self):
        """Override step - IDM roads handle vehicle movement internally."""
        # When using IDM roads, the road agent manages vehicle movement
        # The parent agent just needs to track state
        # Check if current road is IDM by checking for IDM-specific attributes
        if self.current_road and hasattr(self.current_road, 'num_cells'):
            # NaSch road - use parent step
            super().step()
        # Otherwise, IDM road handles everything internally

    def check_reroute(self):
        """
        Check if agent should reroute due to congestion.

        Simple rule: If current road has speed_ratio < 0.2 (severe congestion),
        attempt to find alternative route.
        """
        if self.current_road and self.current_road.speed_ratio < 0.2:
            # Severe congestion - try to reroute
            if self.rerouted_count < 3:  # Limit rerouting attempts
                return True
        return False

    def reroute(self):
        """Compute new route from current location to destination."""
        if not self.current_road:
            return

        current_node = self.current_road.v  # End node of current road

        try:
            new_route_nodes = nx.shortest_path(
                self.model.G,
                current_node,
                self.destination_node,
                weight='length'
            )

            # Convert to edge route
            new_route = []
            for i in range(len(new_route_nodes) - 1):
                u, v = new_route_nodes[i], new_route_nodes[i + 1]
                edge_data = self.model.G.get_edge_data(u, v)
                if edge_data:
                    key = list(edge_data.keys())[0]
                    new_route.append((u, v, key))

            if len(new_route) > 0:
                self.route = new_route
                self.route_index = 0
                self.rerouted_count += 1

        except nx.NetworkXNoPath:
            # No alternative route
            pass


# =============================================================================
# Urban Planning Model
# =============================================================================

class UrbanPlanningModel(mesa.Model):
    """
    Mesa model for urban planning scenarios.

    Features:
    - Normal traffic flow (not evacuation)
    - Network disruptions (lane/road closures)
    - Origin-destination pairs for vehicles
    - Rat-running detection
    - Queue spillback tracking
    """

    def __init__(self, config: UrbanPlanningConfig, disruption: DisruptionConfig = None):
        super().__init__()
        self.config = config
        self.disruption = disruption
        self.rng = np.random.default_rng(config.seed)
        self.space = mg.GeoSpace(warn_crs_conversion=False)

        self.step_count = 0
        self.metrics = EvacuationMetrics()  # Reuse same structure

        # Add urban-specific metrics
        self.metrics.total_queue_length = []  # Sum of all queued vehicles across network
        self.metrics.total_delay = []  # Total delay in vehicle-seconds

        # Network data
        self.G = None
        self.edges_gdf = None
        self.nodes_gdf = None

        # Agent tracking
        self.vehicles = []
        self.road_agents = {}  # Map (u, v, key) -> UrbanRoadAgent

        # Major nodes for OD pairs
        self.major_nodes = []  # High-degree intersections

        # Load network
        self._load_network()
        self._identify_major_nodes()
        self._initialize_roads()

        # Apply disruption if specified
        if self.disruption:
            self._apply_disruption()

    def _load_network(self):
        """Load Singapore road network from OSM."""
        print(f"Loading road network (bbox: {self.config.bbox_south:.4f}, {self.config.bbox_north:.4f}, "
              f"{self.config.bbox_west:.4f}, {self.config.bbox_east:.4f})...")

        bbox = (self.config.bbox_west, self.config.bbox_south,
                self.config.bbox_east, self.config.bbox_north)

        self.G = ox.graph_from_bbox(
            bbox=bbox,
            network_type='drive',
            simplify=True
        )

        self.G = ox.add_edge_speeds(self.G)
        self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G, nodes=True, edges=True, fill_edge_geometry=True)

        if not {'u', 'v', 'key'}.issubset(self.edges_gdf.columns):
            self.edges_gdf = self.edges_gdf.reset_index()

        print(f"  Loaded {len(self.edges_gdf)} road segments, {len(self.nodes_gdf)} nodes")

    def _identify_major_nodes(self):
        """Identify major intersections (high degree nodes) for OD pairs."""
        degrees = dict(self.G.degree())

        # Select nodes with degree >= 3 (junctions, not dead ends)
        major = [node for node, degree in degrees.items() if degree >= 3]

        # Sample subset for computational efficiency
        if len(major) > 100:
            major = self.rng.choice(major, size=100, replace=False).tolist()

        self.major_nodes = major
        print(f"  Identified {len(self.major_nodes)} major nodes for OD pairs")

    def _initialize_roads(self):
        """Create road agents for all edges."""
        model_type = "IDM" if self.config.use_idm else "NaSch"
        print(f"  Using {model_type} traffic model")

        # Import IDM when needed to avoid circular dependency
        if self.config.use_idm:
            from urban_road_idm import UrbanRoadIDM

        for idx, row in self.edges_gdf.iterrows():
            edge_tuple = (row['u'], row['v'], row['key'])

            if self.config.use_idm:
                # Use IDM road agent
                road_agent = UrbanRoadIDM(
                    model=self,
                    geometry=row.geometry,
                    crs=self.edges_gdf.crs,
                    edge_tuple=edge_tuple,
                    length_m=row.get('length', 100),
                    speed_kph=row.get('speed_kph', 50),
                    highway_type=row.get('highway', 'residential')
                )
            else:
                # Use NaSch road agent (legacy)
                road_agent = UrbanRoadAgent(
                    model=self,
                    geometry=row.geometry,
                    crs=self.edges_gdf.crs,
                    edge_tuple=edge_tuple,
                    length_m=row.get('length', 100),
                    speed_kph=row.get('speed_kph', 50),
                    highway_type=row.get('highway', 'residential')
                )

            self.road_agents[edge_tuple] = road_agent
            self.space.add_agents(road_agent)

        print(f"  Initialized {len(self.road_agents)} road agents")

    def _apply_disruption(self):
        """Apply network disruption based on configuration."""
        print(f"\nApplying disruption: {self.disruption.disruption_type}")
        print(f"  Location: {self.disruption.location_name}")
        print(f"  Affected edges: {len(self.disruption.affected_edges)}")

        for edge_tuple in self.disruption.affected_edges:
            road = self.road_agents.get(edge_tuple)
            if road:
                if self.config.use_idm:
                    # IDM: Use Gaussian bottleneck factor
                    if self.disruption.disruption_type == 'lane_closure':
                        # Determine incident position
                        incident_pos = self.disruption.incident_position_m
                        if incident_pos is None:
                            incident_pos = road.length_m / 2.0  # Midpoint

                        road.apply_incident(
                            position_m=incident_pos,
                            epsilon=self.disruption.bottleneck_epsilon,
                            sigma=self.disruption.bottleneck_sigma
                        )
                        print(f"    IDM bottleneck on edge {edge_tuple}:")
                        print(f"      Position: {incident_pos:.1f}m (ε={self.disruption.bottleneck_epsilon}, σ={self.disruption.bottleneck_sigma}m)")

                    elif self.disruption.disruption_type == 'road_closure':
                        # Complete closure: ε = 1.0 (100% reduction)
                        incident_pos = self.disruption.incident_position_m or road.length_m / 2.0
                        road.apply_incident(position_m=incident_pos, epsilon=1.0, sigma=10.0)
                        print(f"    IDM complete blockage on edge {edge_tuple}")
                else:
                    # NaSch: Use capacity reduction (legacy)
                    if self.disruption.disruption_type == 'lane_closure':
                        road.apply_lane_closure(
                            capacity_reduction=self.disruption.capacity_reduction,
                            start_time=self.disruption.start_time,
                            duration=self.disruption.duration
                        )
                        print(f"    NaSch lane closure on edge {edge_tuple}: capacity reduced by {self.disruption.capacity_reduction*100:.0f}%")

                    elif self.disruption.disruption_type == 'road_closure':
                        road.apply_road_closure(
                            start_time=self.disruption.start_time,
                            duration=self.disruption.duration
                        )
                        print(f"    NaSch road closure on edge {edge_tuple}: complete blockage")

    def spawn_vehicle(self, origin_node, destination_node):
        """Spawn a vehicle with OD pair."""
        try:
            route_nodes = nx.shortest_path(self.G, origin_node, destination_node, weight='length')
        except nx.NetworkXNoPath:
            return None

        route = []
        for i in range(len(route_nodes) - 1):
            u, v = route_nodes[i], route_nodes[i + 1]
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                key = list(edge_data.keys())[0]
                route.append((u, v, key))

        if len(route) == 0:
            return None

        origin_point = self.nodes_gdf.loc[origin_node].geometry
        agent_id = len(self.vehicles)

        vehicle = UrbanVehicleAgent(
            model=self,
            geometry=origin_point,
            crs=self.edges_gdf.crs,
            agent_id=agent_id,
            origin_node=origin_node,
            destination_node=destination_node,
            route=route
        )

        self.vehicles.append(vehicle)
        self.space.add_agents(vehicle)

        first_road = self.road_agents.get(route[0])
        if first_road:
            first_road.add_vehicle(vehicle)

        return vehicle

    def get_road_agent(self, edge_tuple):
        """Get road agent by edge tuple."""
        return self.road_agents.get(edge_tuple, None)

    def collect_metrics(self):
        """Collect simulation metrics."""
        # Filter vehicles: ONLY those tracked as bottleneck probes
        # This requires the spawn logic to set 'passed_through_bottleneck' = True
        relevant_vehicles = [
            v for v in self.vehicles
            if getattr(v, 'passed_through_bottleneck', False)
        ]

        # Fallback if attribute not set (e.g. running legacy code)
        if not relevant_vehicles:
            relevant_vehicles = self.vehicles

        completed = sum(1 for v in relevant_vehicles if v.evacuated)

        if len(self.metrics.evacuated_count) > 0:
            flow = completed - self.metrics.evacuated_count[-1]
        else:
            flow = completed

        speeds = [r.mean_speed_kph for r in self.road_agents.values() if r.mean_speed_kph > 0]
        mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0

        congested = sum(1 for r in self.road_agents.values() if r.speed_ratio < 0.3 and len(r.vehicles) > 0)

        # Calculate total queue length (Othman et al. methodology: v < 5 km/h)
        # For IDM roads, use their measure_queue_length() method
        total_queue = 0
        for road in self.road_agents.values():
            if hasattr(road, 'queue_length'):
                total_queue += road.queue_length

        # Calculate total delay (T_actual - T_freeflow) for BOTTLENECK vehicles only
        # For completed vehicles, calculate delay based on trip time vs freeflow time
        total_delay = 0.0
        for vehicle in relevant_vehicles:
            if vehicle.evacuated and hasattr(vehicle, 'trip_time') and vehicle.trip_time is not None:
                # Calculate freeflow time for the route
                if hasattr(vehicle, 'route') and vehicle.route:
                    freeflow_time = 0.0
                    for edge_tuple in vehicle.route:
                        road = self.road_agents.get(edge_tuple)
                        if road and hasattr(road, 'length_m') and hasattr(road, 'speed_kph'):
                            # Freeflow time = length / speed
                            freeflow_time += (road.length_m / 1000.0) / (road.speed_kph / 3600.0)  # in seconds

                    # Delay = actual - freeflow
                    if freeflow_time > 0:
                        delay = vehicle.trip_time - freeflow_time
                        total_delay += max(0, delay)  # Only positive delays

        self.metrics.time_steps.append(self.step_count)
        self.metrics.evacuated_count.append(completed)
        self.metrics.network_flow.append(flow)
        self.metrics.mean_speed.append(mean_speed)
        self.metrics.congested_roads.append(congested)
        self.metrics.total_queue_length.append(total_queue)
        self.metrics.total_delay.append(total_delay)

    def step(self):
        """Advance simulation by one time step."""
        for road in self.road_agents.values():
            road.step()

        for vehicle in self.vehicles:
            if not vehicle.evacuated:
                vehicle.step()

        self.collect_metrics()
        self.step_count += 1

    def run(self, max_steps=None):
        """Run simulation until completion or max steps."""
        if max_steps is None:
            max_steps = self.config.max_steps

        print(f"\nRunning urban planning simulation ({len(self.vehicles)} vehicles)...")

        while self.step_count < max_steps:
            self.step()

            if self.step_count % 600 == 0:  # Report every 10 minutes
                completed = sum(1 for v in self.vehicles if v.evacuated)
                print(f"  Step {self.step_count}/{max_steps} - "
                      f"Completed: {completed}/{len(self.vehicles)} - "
                      f"Congested roads: {self.metrics.congested_roads[-1] if self.metrics.congested_roads else 0}")

        completed = sum(1 for v in self.vehicles if v.evacuated)
        print(f"Simulation complete: {completed}/{len(self.vehicles)} vehicles completed trips")

        return self.metrics
