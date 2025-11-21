"""
Base Evacuation Model using NaSch Traffic Dynamics

This module provides the core components for simulating evacuation scenarios
on real road networks using the Nagel-Schreckenberg (NaSch) cellular automaton
traffic model with routing capabilities.

Key Components:
- EvacueeAgent: Vehicle agent with destination and routing
- EvacuationRoadAgent: Road segment with NaSch dynamics and queue management
- EvacuationModel: Mesa model coordinating evacuation simulation
"""

import numpy as np
import networkx as nx
import osmnx as ox
import geopandas as gpd
import mesa
import mesa_geo as mg
from shapely.geometry import Point, LineString
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque


# =============================================================================
# Configuration and Data Structures
# =============================================================================

@dataclass
class EvacuationConfig:
    """Configuration for evacuation simulation."""
    # Network bounds (Singapore Central region - Marina Bay area)
    bbox_north: float = 1.295
    bbox_south: float = 1.260
    bbox_east: float = 103.880
    bbox_west: float = 103.840

    # Evacuation zones
    evacuation_center: Tuple[float, float] = (1.2775, 103.860)  # Marina Bay center
    safe_zone_radius_km: float = 2.0  # Safe zones beyond this radius

    # Simulation parameters
    num_agents: int = 2000
    delta_t: float = 1.0  # Time step in seconds
    max_steps: int = 7200  # 2 hours simulation time

    # NaSch parameters
    cell_length_m: float = 7.5
    default_v_max: int = 5
    default_p_slow: float = 0.3

    # Random seed
    seed: int = 42


@dataclass
class EvacuationMetrics:
    """Metrics collected during evacuation simulation."""
    time_steps: List[int]
    evacuated_count: List[int]
    network_flow: List[float]
    mean_speed: List[float]
    congested_roads: List[int]

    # Total Evacuation Time (TET) components
    first_departure_time: Optional[int] = None
    last_arrival_time: Optional[int] = None
    total_evacuation_time: Optional[int] = None  # TET in seconds

    def __init__(self):
        self.time_steps = []
        self.evacuated_count = []
        self.network_flow = []
        self.mean_speed = []
        self.congested_roads = []
        self.first_departure_time = None
        self.last_arrival_time = None
        self.total_evacuation_time = None


# =============================================================================
# NaSch Core Functions
# =============================================================================

def nasch_step(road, v_max=5, p_slow=0.3, rng=None):
    """
    Single step of Nagel-Schreckenberg model.

    Args:
        road: Array where road[i] = velocity of car at cell i, or -1 if empty
        v_max: Maximum velocity
        p_slow: Randomization probability
        rng: Random number generator

    Returns:
        Updated road state
    """
    if rng is None:
        rng = np.random.default_rng()

    num_cells = len(road)
    new_road = -np.ones(num_cells, dtype=np.int16)
    car_positions = np.where(road >= 0)[0]

    if len(car_positions) == 0:
        return new_road

    # Calculate gaps to next car
    next_positions = np.roll(car_positions, -1)
    gaps = (next_positions - car_positions - 1) % num_cells

    # Update cars in random order to avoid artifacts
    update_order = rng.permutation(len(car_positions))
    for idx in update_order:
        pos = car_positions[idx]
        v = road[pos]

        # NaSch rules:
        # 1. Acceleration: v -> min(v+1, v_max)
        v = min(v + 1, v_max)

        # 2. Deceleration: v -> min(v, gap)
        v = min(v, gaps[idx])

        # 3. Randomization: if v > 0, with prob p_slow: v -> v-1
        if v > 0 and rng.random() < p_slow:
            v -= 1

        # 4. Movement: move v cells forward
        new_pos = (pos + v) % num_cells
        new_road[new_pos] = v

    return new_road


# =============================================================================
# Agent Classes
# =============================================================================

class EvacueeAgent(mg.GeoAgent):
    """
    Vehicle agent evacuating to a safe zone.

    Attributes:
        destination_node: Target node ID in network
        route: List of edge tuples (u, v, key) to destination
        current_road: EvacuationRoadAgent the vehicle is on
        position: Normalized position on current road (0-1)
        velocity: Current velocity (normalized 0-1)
        cell_index: Index in road's NaSch array
        evacuated: Whether agent has reached safe zone
        evacuation_time: Time step when agent evacuated
    """

    def __init__(self, model, geometry, crs, agent_id, destination_node, route):
        super().__init__(model, geometry, crs)
        self.agent_id = agent_id
        self.destination_node = destination_node
        self.route = route  # List of (u, v, key) tuples
        self.current_road = None
        self.position = 0.0
        self.velocity = 0.0
        self.cell_index = -1
        self.evacuated = False
        self.evacuation_time = None
        self.route_index = 0

    def assign_to_road(self, road_agent):
        """Assign agent to a road segment."""
        self.current_road = road_agent
        self.position = 0.0
        self.update_geometry()

    def update_geometry(self):
        """Update agent's geometry based on position on road."""
        if self.current_road is not None:
            road_geom = self.current_road.geometry
            point_on_road = road_geom.interpolate(self.position, normalized=True)
            self.geometry = point_on_road

    def transition_to_next_road(self):
        """Move to next road segment in route."""
        self.route_index += 1

        if self.route_index >= len(self.route):
            # Reached destination
            self.evacuated = True
            self.evacuation_time = self.model.step_count
            if self.current_road:
                self.current_road.remove_vehicle(self)
            return

        # Get next road from route
        next_edge = self.route[self.route_index]
        next_road = self.model.get_road_agent(next_edge)

        if next_road:
            if self.current_road:
                self.current_road.remove_vehicle(self)
            next_road.add_vehicle(self)

    def step(self):
        """Update agent position and check for road transitions."""
        if self.evacuated:
            return

        # Position updated by road agent's NaSch dynamics
        # Check if reached end of current road
        # Use cell-based check to avoid issues with short roads
        if self.current_road and self.cell_index >= self.current_road.num_cells - 2:
            # At or near last cell - transition to next road
            self.transition_to_next_road()
        else:
            self.update_geometry()


class EvacuationRoadAgent(mg.GeoAgent):
    """
    Road segment agent managing NaSch CA and vehicle flow.

    Extends NaSch model with:
    - Vehicle entry/exit queue management
    - Routing integration (vehicles transitioning between roads)
    - Contraflow capability
    - Performance metrics (throughput, congestion)
    """

    def __init__(self, model, geometry, crs, edge_tuple, length_m, speed_kph, highway_type):
        super().__init__(model, geometry, crs)
        self.edge_tuple = edge_tuple  # (u, v, key)
        self.u, self.v, self.key = edge_tuple
        self.length_m = length_m
        self.speed_kph = speed_kph
        self.highway_type = highway_type

        # NaSch parameters
        self.set_nasch_params()

        # Cellular automaton state
        self.cell_length_m = model.config.cell_length_m
        self.num_cells = max(5, int(self.length_m / self.cell_length_m))
        self.road = -np.ones(self.num_cells, dtype=np.int16)

        # Vehicle tracking
        self.vehicles = []  # List of EvacueeAgent
        self.cell_to_vehicle = {}  # Map cell index -> EvacueeAgent

        # Entry queue (vehicles waiting to enter from previous road)
        self.entry_queue = deque()

        # Metrics
        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0
        self.throughput = 0  # Vehicles exited this road

        # Contraflow
        self.contraflow_enabled = False
        self.base_num_cells = self.num_cells

    def set_nasch_params(self):
        """Set NaSch parameters based on highway type."""
        highway_params = {
            'motorway': {'v_max': 7, 'p_slow': 0.2, 'lanes': 4},
            'trunk': {'v_max': 6, 'p_slow': 0.25, 'lanes': 3},
            'primary': {'v_max': 5, 'p_slow': 0.30, 'lanes': 2},
            'secondary': {'v_max': 4, 'p_slow': 0.35, 'lanes': 2},
            'tertiary': {'v_max': 3, 'p_slow': 0.40, 'lanes': 1},
            'residential': {'v_max': 2, 'p_slow': 0.45, 'lanes': 1}
        }

        highway_type = self.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'residential'
        elif not isinstance(highway_type, str):
            highway_type = str(highway_type) if highway_type else 'residential'

        params = highway_params.get(highway_type, highway_params['residential'])
        self.v_max = params['v_max']
        self.p_slow = params['p_slow']
        self.num_lanes = params['lanes']

    def enable_contraflow(self):
        """
        Enable contraflow - increase road capacity by reversing lanes.

        Contraflow increases the physical capacity (num_cells) of the road by ~50%
        to simulate using opposite-direction lanes for evacuation. This does NOT
        affect routing weights - routes are still based on physical distance.
        """
        if not self.contraflow_enabled and self.num_lanes > 1:
            self.contraflow_enabled = True
            # Increase capacity by ~50% (simulating reversed lanes)
            capacity_multiplier = 1.5
            new_num_cells = int(self.num_cells * capacity_multiplier)

            # Expand road array
            old_road = self.road.copy()
            self.road = -np.ones(new_num_cells, dtype=np.int16)
            self.road[:len(old_road)] = old_road
            self.num_cells = new_num_cells

    def disable_contraflow(self):
        """Disable contraflow - restore original capacity."""
        if self.contraflow_enabled:
            self.contraflow_enabled = False
            self.road = self.road[:self.base_num_cells]
            self.num_cells = self.base_num_cells

    def add_vehicle(self, vehicle):
        """Add vehicle to entry queue."""
        self.entry_queue.append(vehicle)

    def remove_vehicle(self, vehicle):
        """Remove vehicle from road."""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
            if vehicle.cell_index >= 0 and vehicle.cell_index < self.num_cells:
                self.road[vehicle.cell_index] = -1
                if vehicle.cell_index in self.cell_to_vehicle:
                    del self.cell_to_vehicle[vehicle.cell_index]
            vehicle.current_road = None
            vehicle.cell_index = -1
            self.throughput += 1

    def try_insert_from_queue(self):
        """Try to insert vehicles from entry queue into road."""
        while self.entry_queue and self.road[0] == -1:
            vehicle = self.entry_queue.popleft()
            vehicle.current_road = self
            vehicle.position = 0.0
            vehicle.cell_index = 0
            vehicle.velocity = min(1, self.v_max) / self.v_max if self.v_max > 0 else 0
            self.road[0] = min(1, self.v_max)
            self.vehicles.append(vehicle)
            self.cell_to_vehicle[0] = vehicle

    def update_vehicle_positions(self):
        """Update vehicle agents based on NaSch CA state."""
        new_cell_to_vehicle = {}
        car_positions = np.where(self.road >= 0)[0]

        # Match vehicles to cells
        for cell_idx in car_positions:
            # Find which vehicle moved to this cell
            vehicle = None
            for v in self.vehicles:
                if v.cell_index >= 0:
                    # Check if this vehicle could have moved to cell_idx
                    old_pos = v.cell_index
                    velocity = self.road[cell_idx]
                    expected_old = (cell_idx - velocity) % self.num_cells
                    if old_pos == expected_old or old_pos == cell_idx:
                        vehicle = v
                        break

            if vehicle:
                vehicle.cell_index = cell_idx
                vehicle.position = cell_idx / self.num_cells
                vehicle.velocity = self.road[cell_idx] / self.v_max if self.v_max > 0 else 0
                new_cell_to_vehicle[cell_idx] = vehicle

        self.cell_to_vehicle = new_cell_to_vehicle

        # Update speed metrics
        self._update_speed_metrics()

    def _update_speed_metrics(self):
        """Update speed metrics from current road state."""
        moving_vehicles = self.road[self.road >= 0]

        if len(moving_vehicles) > 0:
            mean_v_cells = moving_vehicles.mean()
            mean_speed_kph = mean_v_cells * (self.cell_length_m / 1000) * 3600 / self.model.config.delta_t
            speed_ratio = mean_speed_kph / self.speed_kph if self.speed_kph > 0 else 1.0
            speed_ratio = np.clip(speed_ratio, 0.0, 1.2)

            self.mean_speed_kph = mean_speed_kph
            self.speed_ratio = speed_ratio
        else:
            self.mean_speed_kph = 0.0
            self.speed_ratio = 0.0

    def step(self):
        """Update road state and vehicles."""
        # Try to insert waiting vehicles
        self.try_insert_from_queue()

        # Run NaSch CA step
        self.road = nasch_step(self.road, self.v_max, self.p_slow, self.model.rng)

        # Update vehicle positions
        self.update_vehicle_positions()


# =============================================================================
# Evacuation Model
# =============================================================================

class EvacuationModel(mesa.Model):
    """
    Mesa model for evacuation simulation.

    Manages:
    - Road network loading and routing
    - Agent spawning and destination assignment
    - Simulation stepping
    - Metrics collection
    """

    def __init__(self, config: EvacuationConfig):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.space = mg.GeoSpace(warn_crs_conversion=False)

        self.step_count = 0
        self.metrics = EvacuationMetrics()

        # Network data
        self.G = None  # NetworkX graph
        self.edges_gdf = None  # GeoDataFrame of edges
        self.nodes_gdf = None  # GeoDataFrame of nodes

        # Agent tracking
        self.evacuees = []
        self.road_agents = {}  # Map (u, v, key) -> EvacuationRoadAgent

        # Safe zone nodes
        self.safe_zone_nodes = []
        self.evacuation_zone_center = Point(config.evacuation_center[1], config.evacuation_center[0])

        # Load network
        self._load_network()
        self._identify_safe_zones()
        self._initialize_roads()

    def _load_network(self):
        """Load Singapore road network from OSM."""
        print(f"Loading road network (bbox: {self.config.bbox_south:.4f}, {self.config.bbox_north:.4f}, "
              f"{self.config.bbox_west:.4f}, {self.config.bbox_east:.4f})...")

        # Load network
        # bbox format: (west, south, east, north) or (left, bottom, right, top)
        bbox = (self.config.bbox_west, self.config.bbox_south,
                self.config.bbox_east, self.config.bbox_north)

        self.G = ox.graph_from_bbox(
            bbox=bbox,
            network_type='drive',
            simplify=True
        )

        # Add speeds
        self.G = ox.add_edge_speeds(self.G)

        # Convert to GeoDataFrames
        self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.G, nodes=True, edges=True, fill_edge_geometry=True)

        if not {'u', 'v', 'key'}.issubset(self.edges_gdf.columns):
            self.edges_gdf = self.edges_gdf.reset_index()

        print(f"  Loaded {len(self.edges_gdf)} road segments, {len(self.nodes_gdf)} nodes")

    def _identify_safe_zones(self):
        """Identify nodes outside evacuation zone as safe destinations."""
        safe_radius_m = self.config.safe_zone_radius_km * 1000

        # Use graph nodes directly to ensure compatibility
        for node_id in self.G.nodes():
            if node_id not in self.nodes_gdf.index:
                continue

            node_data = self.nodes_gdf.loc[node_id]
            node_point = node_data.geometry
            distance_m = self.evacuation_zone_center.distance(node_point) * 111139  # Approx meters

            if distance_m >= safe_radius_m:
                self.safe_zone_nodes.append(node_id)

        print(f"  Identified {len(self.safe_zone_nodes)} safe zone nodes")

    def _initialize_roads(self):
        """Create road agents for all edges in network."""
        for idx, row in self.edges_gdf.iterrows():
            edge_tuple = (row['u'], row['v'], row['key'])

            road_agent = EvacuationRoadAgent(
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

    def spawn_evacuee(self, origin_node, destination_node):
        """
        Spawn an evacuee agent.

        Args:
            origin_node: Node ID where agent starts
            destination_node: Node ID of safe zone

        Returns:
            EvacueeAgent instance
        """
        # Try to find a route to the requested destination
        # If that fails, try alternative safe zone destinations
        destinations_to_try = [destination_node] + [
            d for d in self.safe_zone_nodes[:20] if d != destination_node
        ]

        route_nodes = None
        final_destination = None

        for dest in destinations_to_try:
            try:
                route_nodes = nx.shortest_path(self.G, origin_node, dest, weight='length')
                final_destination = dest
                break
            except nx.NetworkXNoPath:
                continue

        if route_nodes is None:
            # No path to any safe zone - this origin is isolated
            return None

        # Convert to edge route
        route = []
        for i in range(len(route_nodes) - 1):
            u, v = route_nodes[i], route_nodes[i + 1]
            # Get edge key (may be multiple edges between u and v)
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                key = list(edge_data.keys())[0]
                route.append((u, v, key))

        if len(route) == 0:
            return None

        # Create agent at origin
        origin_point = self.nodes_gdf.loc[origin_node].geometry
        agent_id = len(self.evacuees)

        evacuee = EvacueeAgent(
            model=self,
            geometry=origin_point,
            crs=self.edges_gdf.crs,
            agent_id=agent_id,
            destination_node=final_destination,  # Use the destination we found a route to
            route=route
        )

        self.evacuees.append(evacuee)
        self.space.add_agents(evacuee)

        # Add to first road
        first_road = self.get_road_agent(route[0])
        if first_road:
            first_road.add_vehicle(evacuee)

        return evacuee

    def get_road_agent(self, edge_tuple):
        """Get road agent by edge tuple."""
        return self.road_agents.get(edge_tuple, None)

    def collect_metrics(self):
        """Collect simulation metrics."""
        evacuated_count = sum(1 for e in self.evacuees if e.evacuated)

        # Network flow (vehicles that completed evacuation this step)
        if len(self.metrics.evacuated_count) > 0:
            flow = evacuated_count - self.metrics.evacuated_count[-1]
        else:
            flow = evacuated_count

        # Mean speed across network
        speeds = [r.mean_speed_kph for r in self.road_agents.values() if r.mean_speed_kph > 0]
        mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0

        # Congested roads (chi < 0.3)
        congested = sum(1 for r in self.road_agents.values() if r.speed_ratio < 0.3 and len(r.vehicles) > 0)

        # Track Total Evacuation Time (TET) components
        # First departure time: when first vehicle enters network (usually t=0 for simultaneous)
        if self.metrics.first_departure_time is None and len(self.evacuees) > 0:
            # Check if any vehicle has started (is on a road)
            if any(e.current_road is not None for e in self.evacuees):
                self.metrics.first_departure_time = self.step_count

        # Last arrival time: when last vehicle reaches destination
        if evacuated_count == len(self.evacuees) and evacuated_count > 0:
            if self.metrics.last_arrival_time is None:
                self.metrics.last_arrival_time = self.step_count
                # Calculate TET
                if self.metrics.first_departure_time is not None:
                    self.metrics.total_evacuation_time = self.metrics.last_arrival_time - self.metrics.first_departure_time

        self.metrics.time_steps.append(self.step_count)
        self.metrics.evacuated_count.append(evacuated_count)
        self.metrics.network_flow.append(flow)
        self.metrics.mean_speed.append(mean_speed)
        self.metrics.congested_roads.append(congested)

    def step(self):
        """Advance simulation by one time step."""
        # Update all roads
        for road in self.road_agents.values():
            road.step()

        # Update all evacuees
        for evacuee in self.evacuees:
            if not evacuee.evacuated:
                evacuee.step()

        # Collect metrics
        self.collect_metrics()

        self.step_count += 1

    def run(self, max_steps=None):
        """
        Run simulation until completion or max steps.

        Args:
            max_steps: Maximum time steps (default: config.max_steps)

        Returns:
            EvacuationMetrics
        """
        if max_steps is None:
            max_steps = self.config.max_steps

        print(f"\nRunning evacuation simulation ({len(self.evacuees)} agents)...")

        while self.step_count < max_steps:
            self.step()

            # Check evacuation progress
            evacuated_count = sum(1 for e in self.evacuees if e.evacuated)
            evacuation_pct = evacuated_count / len(self.evacuees) * 100 if len(self.evacuees) > 0 else 0

            if self.step_count % 300 == 0:  # Every 5 minutes
                print(f"  Step {self.step_count}/{max_steps} - "
                      f"Evacuated: {evacuated_count}/{len(self.evacuees)} ({evacuation_pct:.1f}%) - "
                      f"Congested roads: {self.metrics.congested_roads[-1]}")

            # Stop if 95% evacuated
            if evacuation_pct >= 95.0:
                print(f"  95% evacuation threshold reached at step {self.step_count}")
                break

        print(f"Simulation complete: {evacuated_count}/{len(self.evacuees)} evacuated")
        return self.metrics


# =============================================================================
# Helper Functions
# =============================================================================

def get_origin_nodes(model, num_agents):
    """
    Select origin nodes within evacuation zone.

    Args:
        model: EvacuationModel instance
        num_agents: Number of agents to spawn

    Returns:
        List of node IDs
    """
    # Select nodes close to evacuation zone center
    evacuation_radius_m = model.config.safe_zone_radius_km * 0.5 * 1000  # Half of safe zone radius
    origin_candidates = []

    for node_id, node_data in model.nodes_gdf.iterrows():
        node_point = node_data.geometry
        distance_m = model.evacuation_zone_center.distance(node_point) * 111139

        if distance_m <= evacuation_radius_m:
            origin_candidates.append(node_id)

    if len(origin_candidates) == 0:
        # Fallback: use all nodes
        origin_candidates = list(model.nodes_gdf.index)

    # Sample with replacement
    origins = model.rng.choice(origin_candidates, size=num_agents, replace=True)
    return list(origins)


def assign_destinations(model, origin_nodes):
    """
    Assign destination nodes to evacuees.

    Args:
        model: EvacuationModel instance
        origin_nodes: List of origin node IDs

    Returns:
        List of (origin, destination) tuples
    """
    destinations = []

    for origin in origin_nodes:
        # Assign nearest safe zone node
        origin_point = model.nodes_gdf.loc[origin].geometry
        min_dist = float('inf')
        best_dest = None

        for safe_node in model.safe_zone_nodes[:50]:  # Check first 50 for efficiency
            safe_point = model.nodes_gdf.loc[safe_node].geometry
            dist = origin_point.distance(safe_point)
            if dist < min_dist:
                min_dist = dist
                best_dest = safe_node

        if best_dest is None:
            best_dest = model.rng.choice(model.safe_zone_nodes)

        destinations.append((origin, best_dest))

    return destinations
