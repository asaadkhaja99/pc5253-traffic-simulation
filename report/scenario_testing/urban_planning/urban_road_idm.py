"""
IDM-Based Road Agent for Urban Planning Scenarios

Implements the Intelligent Driver Model (IDM) with:
1. Virtual Leader / Ghost Vehicle concept for traffic lights and obstacles
2. Gaussian Bottleneck Factor (Eq. 16/17 from methodology)
3. Queue measurement and delay tracking

This allows proper modeling of:
- Traffic signals (red light = stopped virtual car)
- Lane closures/incidents (construction cone = stopped virtual car)
- Smooth deceleration and queue formation
"""

import numpy as np
import mesa_geo as mg
from typing import Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# IDM Core Functions
# =============================================================================

def idm_acceleration(v, v_lead, gap, v0, s0, T, a_max, b, delta):
    """
    Calculate IDM acceleration for a single vehicle.

    Args:
        v: Current velocity (m/s)
        v_lead: Leader velocity (m/s)
        gap: Gap to leader (m)
        v0: Desired velocity (m/s)
        s0: Minimum gap (m)
        T: Time headway (s)
        a_max: Maximum acceleration (m/s²)
        b: Comfortable deceleration (m/s²)
        delta: Acceleration exponent

    Returns:
        Acceleration (m/s²)
    """
    # Desired dynamical distance
    dv = v - v_lead
    s_star = s0 + max(0, v * T + (v * dv) / (2 * np.sqrt(a_max * b)))

    # Free road term
    free_term = 1 - (v / v0)**delta if v0 > 0 else 0

    # Interaction term
    interaction_term = (s_star / max(gap, s0))**2

    acceleration = a_max * (free_term - interaction_term)

    return acceleration


def gaussian_bottleneck_factor(position_m, incident_position_m, epsilon, sigma):
    """
    Gaussian bottleneck factor (Eq. 17 from methodology).

    Reduces desired velocity near incident location.

    Args:
        position_m: Vehicle position (m)
        incident_position_m: Incident location (m)
        epsilon: Strength of bottleneck (0-1)
        sigma: Spatial spread (m)

    Returns:
        Bottleneck factor B(x) in range [1-epsilon, 1]
    """
    distance_sq = (position_m - incident_position_m)**2
    B = 1.0 - epsilon * np.exp(-distance_sq / (2 * sigma**2))
    return B


# =============================================================================
# Virtual Leader / Ghost Vehicle Implementation
# =============================================================================

@dataclass
class VirtualLeader:
    """Represents a virtual leader (traffic light, obstacle)."""
    position_m: float  # Position on road
    velocity: float = 0.0  # Always 0 for obstacles
    active: bool = True  # For traffic lights: True when RED


class IDMVehicle:
    """
    Single vehicle following IDM dynamics.

    Attributes:
        position_m: Position along road (meters from start)
        velocity: Current velocity (m/s)
        vehicle_id: Unique identifier
    """

    def __init__(self, position_m: float, velocity: float, vehicle_id: int):
        self.position_m = position_m
        self.velocity = velocity
        self.vehicle_id = vehicle_id
        self.acceleration = 0.0

        # Trip metrics
        self.entry_time = None  # When vehicle entered road
        self.exit_time = None  # When vehicle exited road
        self.travel_time = None  # Actual travel time (seconds)


class UrbanRoadIDM(mg.GeoAgent):
    """
    Road segment agent using IDM dynamics with virtual leaders.

    This agent manages a collection of vehicles following IDM dynamics
    and handles virtual leaders for traffic control and incidents.
    """

    def __init__(self, model, geometry, crs, edge_tuple, length_m, speed_kph, highway_type):
        super().__init__(model, geometry, crs)

        # Road properties
        self.edge_tuple = edge_tuple  # (u, v, key)
        self.u, self.v, self.key = edge_tuple
        self.length_m = length_m
        self.speed_kph = speed_kph
        self.highway_type = highway_type

        # IDM parameters
        self._set_idm_params()

        # Vehicle management
        self.vehicles: List[IDMVehicle] = []
        self.entry_queue = []  # Vehicles waiting to enter
        # Note: We lookup agents directly from model.vehicles list using vehicle_id as index

        # Virtual leaders (traffic lights, incidents)
        self.virtual_leaders: List[VirtualLeader] = []

        # Disruption state
        self.is_disrupted = False
        self.incident_position_m = None  # Where incident is located
        self.bottleneck_epsilon = 0.0  # Strength of bottleneck
        self.bottleneck_sigma = 50.0  # Spatial spread (m)

        # Metrics
        self.queue_length = 0  # Number of queued vehicles
        self.queue_length_m = 0.0  # Queue length in meters
        self.total_vehicles_passed = 0
        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0
        self.is_residential = highway_type in ['residential', 'tertiary', 'unclassified']

        # Time step
        self.dt = 0.5  # IDM sub-timestep (seconds)

    def _set_idm_params(self):
        """Set IDM parameters based on highway type."""
        highway_params = {
            'motorway': {'v0_kph': 110, 's0': 2.0, 'T': 1.2, 'a_max': 2.0, 'b': 3.0, 'delta': 4},
            'trunk': {'v0_kph': 90, 's0': 2.0, 'T': 1.3, 'a_max': 1.8, 'b': 2.8, 'delta': 4},
            'primary': {'v0_kph': 70, 's0': 2.0, 'T': 1.4, 'a_max': 1.6, 'b': 2.5, 'delta': 4},
            'secondary': {'v0_kph': 60, 's0': 2.0, 'T': 1.5, 'a_max': 1.4, 'b': 2.2, 'delta': 4},
            'tertiary': {'v0_kph': 50, 's0': 1.5, 'T': 1.6, 'a_max': 1.2, 'b': 2.0, 'delta': 4},
            'residential': {'v0_kph': 40, 's0': 1.5, 'T': 1.8, 'a_max': 1.0, 'b': 1.8, 'delta': 4}
        }

        highway_type = self.highway_type
        if isinstance(highway_type, list):
            highway_type = highway_type[0] if highway_type else 'residential'
        elif not isinstance(highway_type, str):
            highway_type = str(highway_type) if highway_type else 'residential'

        params = highway_params.get(highway_type, highway_params['residential'])

        # Convert to m/s
        self.v0 = params['v0_kph'] / 3.6  # m/s
        self.s0 = params['s0']  # m
        self.T = params['T']  # s
        self.a_max = params['a_max']  # m/s²
        self.b = params['b']  # m/s²
        self.delta = params['delta']

    def apply_incident(self, position_m: float, epsilon: float = 0.8, sigma: float = 50.0):
        """
        Apply incident/lane closure at specific position using Gaussian bottleneck factor.

        Args:
            position_m: Position of incident along road (meters from start)
            epsilon: Bottleneck strength (0-1), 0.8 = 80% reduction
            sigma: Spatial spread of bottleneck effect (meters)

        Note: We use ONLY the Gaussian bottleneck factor (Eq. 17) to reduce desired velocity.
        We do NOT add a virtual leader because that would create a permanent roadblock.
        Virtual leaders are for traffic lights (temporary stops), not lane closures (reduced capacity).
        """
        self.is_disrupted = True
        self.incident_position_m = position_m
        self.bottleneck_epsilon = epsilon
        self.bottleneck_sigma = sigma

        # Do NOT add virtual leader - Gaussian bottleneck handles capacity reduction

    def remove_incident(self):
        """Remove incident and restore normal operation."""
        self.is_disrupted = False
        self.incident_position_m = None
        self.bottleneck_epsilon = 0.0
        self.virtual_leaders = []

    def get_leader_info(self, vehicle: IDMVehicle) -> Tuple[float, float]:
        """
        Get leader information for a vehicle (implements Virtual Leader concept).

        Returns (gap, v_lead) where:
        - gap: Distance to next obstacle/vehicle (m)
        - v_lead: Velocity of leader (m/s)

        This method checks:
        1. Virtual leaders (traffic lights, incidents)
        2. Actual physical vehicles ahead

        Returns whichever is closer.
        """
        # Find actual vehicle ahead
        vehicles_ahead = [v for v in self.vehicles
                         if v.position_m > vehicle.position_m]

        if len(vehicles_ahead) > 0:
            # Sort by position, get closest
            vehicles_ahead.sort(key=lambda v: v.position_m)
            actual_leader = vehicles_ahead[0]
            actual_gap = actual_leader.position_m - vehicle.position_m
            actual_v_lead = actual_leader.velocity
        else:
            # No vehicle ahead - use road length
            actual_gap = self.length_m - vehicle.position_m
            actual_v_lead = self.v0  # Free flow

        # Check for virtual leaders (closer than actual vehicle)
        min_gap = actual_gap
        leader_velocity = actual_v_lead

        for virtual in self.virtual_leaders:
            if not virtual.active:
                continue

            # Only consider virtual leaders AHEAD of vehicle
            if virtual.position_m > vehicle.position_m:
                virtual_gap = virtual.position_m - vehicle.position_m

                if virtual_gap < min_gap:
                    min_gap = virtual_gap
                    leader_velocity = virtual.velocity  # Usually 0.0

        return min_gap, leader_velocity

    def get_effective_desired_velocity(self, position_m: float) -> float:
        """
        Get effective desired velocity at position (includes bottleneck effect).

        Applies Gaussian bottleneck factor near incidents.
        """
        v0_base = self.v0

        if self.is_disrupted and self.incident_position_m is not None:
            B = gaussian_bottleneck_factor(
                position_m,
                self.incident_position_m,
                self.bottleneck_epsilon,
                self.bottleneck_sigma
            )
            v0_effective = v0_base * B
        else:
            v0_effective = v0_base

        return v0_effective

    def add_vehicle(self, vehicle_agent):
        """Add vehicle from upstream road."""
        # Create IDM vehicle
        idm_vehicle = IDMVehicle(
            position_m=0.0,  # Start of road
            velocity=self.v0 * 0.5,  # Half of desired speed
            vehicle_id=vehicle_agent.agent_id  # Use agent_id not id
        )
        idm_vehicle.entry_time = self.model.step_count

        self.vehicles.append(idm_vehicle)

        # Update agent state
        vehicle_agent.current_road = self
        vehicle_agent.position = 0.0


    def _transfer_agent_to_next_road(self, agent):
        """
        Helper to move an agent to the next road in its route.
        """
        # Check if route is finished
        if agent.route_index >= len(agent.route) - 1:
            # Trip complete!
            agent.trip_completed = True
            agent.completion_time = self.model.step_count
            agent.current_road = None
            return

        # Increment route index
        agent.route_index += 1
        next_edge = agent.route[agent.route_index]

        # Find the next road agent
        next_road_agent = self.model.get_road_agent(next_edge)

        if next_road_agent:
            # Add vehicle to the NEXT road
            # We pass the agent object; the road will create a new IDMVehicle struct
            next_road_agent.add_vehicle(agent)

            # Update agent's current tracking
            agent.current_road = next_road_agent
        else:
            # Edge case: Route points to non-existent road (graph mismatch)
            # Just mark finished to prevent crashing
            agent.trip_completed = True
            agent.completion_time = self.model.step_count
            agent.current_road = None

    def measure_queue_length(self) -> Tuple[int, float]:
        """
        Measure queue length (Othman et al. methodology).

        Queue defined as: vehicles with speed < 5 km/h (1.4 m/s)
        upstream of bottleneck.

        Returns:
            (queue_count, queue_length_m)
        """
        if not self.is_disrupted or self.incident_position_m is None:
            return 0, 0.0

        queue_threshold_mps = 1.4  # 5 km/h

        queued_vehicles = [
            v for v in self.vehicles
            if v.position_m < self.incident_position_m  # Upstream
            and v.velocity < queue_threshold_mps  # Slow/stopped
        ]

        if len(queued_vehicles) == 0:
            return 0, 0.0

        # Find furthest upstream queued vehicle
        positions = [v.position_m for v in queued_vehicles]
        queue_start = min(positions)
        queue_end = self.incident_position_m
        queue_length_m = queue_end - queue_start

        return len(queued_vehicles), queue_length_m

    def step(self):
        """Update vehicle positions and velocities using IDM."""
        if len(self.vehicles) == 0:
            self.mean_speed_kph = self.speed_kph
            self.speed_ratio = 1.0
            return

        # Calculate accelerations for all vehicles (store v0_eff for velocity capping)
        vehicle_v0_eff = {}
        for vehicle in self.vehicles:
            gap, v_lead = self.get_leader_info(vehicle)
            v0_eff = self.get_effective_desired_velocity(vehicle.position_m)
            vehicle_v0_eff[id(vehicle)] = v0_eff

            acceleration = idm_acceleration(
                v=vehicle.velocity,
                v_lead=v_lead,
                gap=gap,
                v0=v0_eff,
                s0=self.s0,
                T=self.T,
                a_max=self.a_max,
                b=self.b,
                delta=self.delta
            )

            vehicle.acceleration = acceleration

        # Update velocities and positions
        for vehicle in self.vehicles:
            # Update velocity
            vehicle.velocity += vehicle.acceleration * self.dt
            vehicle.velocity = max(0.0, vehicle.velocity)  # No negative speeds
            # Cap at EFFECTIVE max speed (respects Gaussian bottleneck)
            v0_eff = vehicle_v0_eff[id(vehicle)]
            vehicle.velocity = min(vehicle.velocity, v0_eff)

            # Update position
            vehicle.position_m += vehicle.velocity * self.dt

        # 3. Handle Exits - THE FIX from user
        # Identify vehicles that reached the end
        exited_idm_vehicles = [v for v in self.vehicles if v.position_m >= self.length_m]

        # Remove them from THIS road immediately
        self.vehicles = [v for v in self.vehicles if v.position_m < self.length_m]

        for v_idm in exited_idm_vehicles:
            # Update stats for this road
            v_idm.exit_time = self.model.step_count
            v_idm.travel_time = v_idm.exit_time - v_idm.entry_time if v_idm.entry_time else None
            self.total_vehicles_passed += 1

            # --- HANDOVER LOGIC START ---
            # Find the actual agent in the main model
            # (Assuming agent_id matches vehicle_id)
            if 0 <= v_idm.vehicle_id < len(self.model.vehicles):
                agent = self.model.vehicles[v_idm.vehicle_id]

                # Verify this is actually the agent (double check ID)
                if agent.agent_id == v_idm.vehicle_id:
                    # Mark if this vehicle passed through the bottleneck road
                    if hasattr(self.model, 'bottleneck_road_edge'):
                        if self.edge_tuple == self.model.bottleneck_road_edge:
                            agent.passed_through_bottleneck = True

                    # Move agent to next step in route
                    self._transfer_agent_to_next_road(agent)
            # --- HANDOVER LOGIC END ---

        # Update metrics
        if len(self.vehicles) > 0:
            speeds_kph = [v.velocity * 3.6 for v in self.vehicles]
            self.mean_speed_kph = np.mean(speeds_kph)
            self.speed_ratio = self.mean_speed_kph / self.speed_kph if self.speed_kph > 0 else 0
        else:
            self.mean_speed_kph = self.speed_kph
            self.speed_ratio = 1.0

        # Measure queue
        self.queue_length, self.queue_length_m = self.measure_queue_length()
