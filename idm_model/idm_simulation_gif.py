"""
IDM (Intelligent Driver Model) Traffic Simulation GIF Generator

This script generates an animated GIF showing traffic evolution over time
using the Intelligent Driver Model with Mesa-Geo framework.
"""

# Geospatial libraries
import osmnx as ox
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Data analysis
import pandas as pd

# Mesa framework
import mesa
import mesa_geo as mg

# Visualization
import folium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io
import time
import os

# Configuration
ox.settings.use_cache = True


# =============================================================================
# IDM Model Functions
# =============================================================================

def idm_acceleration(x, v, N, v0, s0, T, a_max, b, delta, L):
    """
    Calculate IDM acceleration for all vehicles.

    Parameters:
    -----------
    x : ndarray
        Vehicle positions
    v : ndarray
        Vehicle velocities
    N : int
        Number of vehicles
    v0 : float
        Desired velocity
    s0 : float
        Minimum spacing
    T : float
        Safe time headway
    a_max : float
        Maximum acceleration
    b : float
        Comfortable deceleration
    delta : float
        Acceleration exponent
    L : float
        Road length (for periodic boundary)

    Returns:
    --------
    ndarray : Accelerations for all vehicles
    """
    # Periodic boundary conditions
    x_lead = np.append(x[1:], x[0] + L)
    v_lead = np.append(v[1:], v[0])

    # Calculate spacing (distance to leading vehicle)
    s = x_lead - x
    s = np.where(s < 0, s + L, s)  # Handle wraparound

    # Calculate desired spacing
    dv = v - v_lead  # Approaching rate
    s_star = s0 + v * T + (v * dv) / (2 * np.sqrt(a_max * b))

    # IDM acceleration formula
    # a = a_max * [1 - (v/v0)^delta - (s*/s)^2]
    free_road_term = 1 - (v / v0)**delta
    interaction_term = (s_star / np.maximum(s, s0))**2  # Avoid division by zero

    acceleration = a_max * (free_road_term - interaction_term)

    return acceleration


# =============================================================================
# Agent Classes
# =============================================================================

class VehicleAgent(mg.GeoAgent):
    """Vehicle agent following IDM dynamics."""

    def __init__(self, model, geometry, crs, road_segment, position, velocity, vehicle_index):
        super().__init__(model, geometry, crs)
        self.road_segment = road_segment
        self.position = position
        self.velocity = velocity
        self.vehicle_index = vehicle_index
        self.acceleration = 0.0

    def update_position_on_road(self):
        """Update geometry based on normalized position along road."""
        road_geom = self.road_segment.geometry
        point_on_road = road_geom.interpolate(self.position, normalized=True)
        self.geometry = point_on_road

    def step(self):
        """Update vehicle position based on IDM dynamics."""
        self.update_position_on_road()


class RoadAgent(mg.GeoAgent):
    """Road segment agent managing IDM vehicle flow."""

    def __init__(self, model, geometry, crs, edge_id, u, v, key,
                 length_m, speed_kph, highway_type, num_vehicles):
        super().__init__(model, geometry, crs)
        self.edge_id = edge_id
        self.u = u
        self.v = v
        self.key = key
        self.length_m = length_m
        self.speed_kph = speed_kph
        self.highway_type = highway_type
        self.num_vehicles = num_vehicles

        self.set_idm_params()

        self.positions = None
        self.velocities = None
        self.vehicles = []

        self.velocity_history = []
        self.internal_step_count = 0
        self.warmup_steps = 0

        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0

    def set_idm_params(self):
        """Set IDM parameters based on highway type."""
        highway_params = {
            'motorway': {
                'v0': 1.0, 's0': 0.05, 'T': 1.2,
                'a_max': 2.0, 'b': 3.0, 'delta': 4,
                'congestion_factor': 0.70
            },
            'trunk': {
                'v0': 0.9, 's0': 0.04, 'T': 1.3,
                'a_max': 1.8, 'b': 2.8, 'delta': 4,
                'congestion_factor': 0.75
            },
            'primary': {
                'v0': 0.85, 's0': 0.04, 'T': 1.4,
                'a_max': 1.6, 'b': 2.5, 'delta': 4,
                'congestion_factor': 0.80
            },
            'secondary': {
                'v0': 0.8, 's0': 0.03, 'T': 1.5,
                'a_max': 1.4, 'b': 2.2, 'delta': 4,
                'congestion_factor': 0.85
            },
            'tertiary': {
                'v0': 0.75, 's0': 0.03, 'T': 1.6,
                'a_max': 1.2, 'b': 2.0, 'delta': 4,
                'congestion_factor': 0.90
            },
            'residential': {
                'v0': 0.7, 's0': 0.025, 'T': 1.8,
                'a_max': 1.0, 'b': 1.8, 'delta': 4,
                'congestion_factor': 0.95
            }
        }

        params = highway_params.get(self.highway_type, highway_params['residential'])
        self.v0 = params['v0']
        self.s0 = params['s0']
        self.T = params['T']
        self.a_max = params['a_max']
        self.b = params['b']
        self.delta = params['delta']
        self.congestion_factor = params['congestion_factor']
        self.L = 1.0  # Normalized length

    def initialize_vehicles(self):
        """Initialize vehicle positions and velocities for IDM."""
        N = self.num_vehicles

        # Uniform spacing initially
        self.positions = np.linspace(0, self.L, N, endpoint=False)

        # Add small perturbations
        perturbation = self.model.rng.normal(0, 0.02, N)
        self.positions = (self.positions + perturbation) % self.L

        # Sort positions
        self.positions = np.sort(self.positions)

        # Start at 80% of desired speed with small variation
        self.velocities = np.ones(N) * self.v0 * 0.8
        perturbation_v = self.model.rng.normal(0, self.v0 * 0.1, N)
        self.velocities = np.clip(self.velocities + perturbation_v, 0, self.v0)

        self._run_warmup_simulation()

    def _run_warmup_simulation(self, time_end=30.0, delta_t=0.01):
        """Run warmup simulation to reach equilibrium (IDM needs longer warmup)."""
        total_steps = int(time_end / delta_t)
        self.warmup_steps = int(0.70 * total_steps)  # Use last 30% for steady-state

        velocity_history = []

        for step in range(total_steps):
            a = idm_acceleration(self.positions, self.velocities, self.num_vehicles,
                               self.v0, self.s0, self.T, self.a_max, self.b,
                               self.delta, self.L)

            # Euler update
            self.positions = self.positions + delta_t * self.velocities
            self.velocities = self.velocities + delta_t * a

            # Periodic boundary conditions
            self.positions = self.positions % self.L
            self.velocities = np.clip(self.velocities, 0, self.v0 * 1.2)  # Allow slight overspeed

            # Sort to maintain order
            order = np.argsort(self.positions)
            self.positions = self.positions[order]
            self.velocities = self.velocities[order]

            if step >= self.warmup_steps:
                velocity_history.append(self.velocities.copy())

        if len(velocity_history) > 0:
            v_steady = np.array(velocity_history)
            mean_v_normalized = np.mean(v_steady)

            # Convert to speed ratio
            idm_influence = (mean_v_normalized / self.v0) * 0.4
            speed_ratio = self.congestion_factor + idm_influence - 0.2

            # Add traffic variation
            traffic_noise = self.model.rng.normal(0, 0.03)
            speed_ratio += traffic_noise
            speed_ratio = np.clip(speed_ratio, 0.3, 1.1)

            self.speed_ratio = speed_ratio
            self.mean_speed_kph = self.speed_kph * speed_ratio

    def update_idm_dynamics(self, delta_t=0.01):
        """Update vehicle dynamics using IDM (Euler method)."""
        if self.num_vehicles == 0:
            return

        a = idm_acceleration(self.positions, self.velocities, self.num_vehicles,
                           self.v0, self.s0, self.T, self.a_max, self.b,
                           self.delta, self.L)

        # Euler update
        self.positions = self.positions + delta_t * self.velocities
        self.velocities = self.velocities + delta_t * a

        # Periodic boundary conditions
        self.positions = self.positions % self.L
        self.velocities = np.clip(self.velocities, 0, self.v0 * 1.2)

        # Sort to maintain order
        order = np.argsort(self.positions)
        self.positions = self.positions[order]
        self.velocities = self.velocities[order]

        self.velocity_history.append(self.velocities.copy())
        self.internal_step_count += 1

        # Update metrics periodically
        if self.internal_step_count % 50 == 0 and len(self.velocity_history) > 10:
            recent_v = np.array(self.velocity_history[-10:])
            mean_v_normalized = np.mean(recent_v)

            idm_influence = (mean_v_normalized / self.v0) * 0.4
            speed_ratio = self.congestion_factor + idm_influence - 0.2
            speed_ratio = np.clip(speed_ratio, 0.3, 1.1)

            self.speed_ratio = speed_ratio
            self.mean_speed_kph = self.speed_kph * speed_ratio

    def step(self):
        """Update road and vehicle dynamics."""
        self.update_idm_dynamics(delta_t=self.model.delta_t)

        for i, vehicle in enumerate(self.vehicles):
            if i < len(self.positions):
                vehicle.position = self.positions[i]
                vehicle.velocity = self.velocities[i]


class IDMTrafficModel(mesa.Model):
    """Mesa-Geo model for IDM traffic simulation."""

    def __init__(self, edges_gdf, num_roads=50, vehicles_per_road=10,
                 delta_t=0.01, min_road_length=200):
        super().__init__()

        self.num_roads = num_roads
        self.vehicles_per_road = vehicles_per_road
        self.delta_t = delta_t
        self.min_road_length = min_road_length

        self.space = mg.GeoSpace(warn_crs_conversion=False)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Mean Speed (km/h)": lambda m: np.mean([r.mean_speed_kph for r in m.agents_by_type[RoadAgent]]) if m.agents_by_type[RoadAgent] else 0,
                "Mean Speed Ratio": lambda m: np.mean([r.speed_ratio for r in m.agents_by_type[RoadAgent]]) if m.agents_by_type[RoadAgent] else 0,
                "Total Vehicles": lambda m: sum([r.num_vehicles for r in m.agents_by_type[RoadAgent]]) if m.agents_by_type[RoadAgent] else 0,
            }
        )

        self.setup_roads(edges_gdf)
        self.setup_vehicles()

        self.running = True

    def setup_roads(self, edges_gdf):
        """Create road agents from edges GeoDataFrame."""
        filtered_edges = edges_gdf[edges_gdf['length'] >= self.min_road_length].copy()

        if len(filtered_edges) > self.num_roads:
            sampled_edges = filtered_edges.sample(self.num_roads, random_state=42)
        else:
            sampled_edges = filtered_edges

        for idx, row in sampled_edges.iterrows():
            if isinstance(idx, tuple):
                u, v, key = idx
            else:
                u = row.get('u', idx)
                v = row.get('v', None)
                key = row.get('key', 0)

            highway_raw = row.get('highway', 'residential')
            if isinstance(highway_raw, list):
                highway_type = highway_raw[0] if highway_raw else 'residential'
            else:
                highway_type = str(highway_raw) if highway_raw else 'residential'

            highway_vehicles = {
                'motorway': 25,
                'trunk': 20,
                'primary': 15,
                'secondary': 12,
                'tertiary': 10,
                'residential': 8
            }
            num_vehicles = highway_vehicles.get(highway_type, self.vehicles_per_road)

            road_agent = RoadAgent(
                model=self,
                geometry=row['geometry'],
                crs=self.space.crs,
                edge_id=f"{u}_{v}_{key}",
                u=u,
                v=v,
                key=key,
                length_m=row['length'],
                speed_kph=row.get('speed_kph', 50),
                highway_type=highway_type,
                num_vehicles=num_vehicles
            )

            road_agent.initialize_vehicles()
            self.space.add_agents(road_agent)

    def setup_vehicles(self):
        """Create vehicle agents on roads."""
        for road in self.agents_by_type[RoadAgent]:
            for i in range(road.num_vehicles):
                position_normalized = road.positions[i]
                point_on_road = road.geometry.interpolate(position_normalized, normalized=True)

                vehicle = VehicleAgent(
                    model=self,
                    geometry=point_on_road,
                    crs=self.space.crs,
                    road_segment=road,
                    position=position_normalized,
                    velocity=road.velocities[i],
                    vehicle_index=i
                )

                road.vehicles.append(vehicle)
                self.space.add_agents(vehicle)

    def step(self):
        """Run one step of the model."""
        self.agents_by_type[RoadAgent].do("step")
        self.agents_by_type[VehicleAgent].do("step")
        self.datacollector.collect(self)


# =============================================================================
# Utility Functions
# =============================================================================

def speed_color(speed_kph):
    """Color function matching notebook visualization."""
    if speed_kph is None or np.isnan(speed_kph):
        return '#999999'

    # Colors matching notebook visualization
    if speed_kph < 20:     return '#d73027'    # Red: Very slow
    elif speed_kph < 35:   return '#fc8d59'    # Orange: Slow
    elif speed_kph < 50:   return '#fee08b'    # Yellow: Moderate
    elif speed_kph < 70:   return '#91cf60'    # Light green: Fast
    else:                  return '#1a9850'    # Dark green: Very fast


def create_folium_frame(roads_data, step, num_steps, center=(1.3521, 103.8198),
                        zoom=12, width=1200, height=900, basemap_style='light'):
    """Create a single Folium map frame."""

    if basemap_style == 'dark':
        tiles = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        attr = '&copy; OpenStreetMap contributors &copy; CARTO'
    elif basemap_style == 'light':
        tiles = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
        attr = '&copy; OpenStreetMap contributors &copy; CARTO'
    elif basemap_style == 'satellite':
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        attr = 'Tiles &copy; Esri'
    else:  # none
        tiles = None
        attr = ''

    if tiles:
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=tiles,
            attr=attr,
            width=width,
            height=height,
            prefer_canvas=True,
            zoom_control=False,
            attributionControl=False
        )
    else:
        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles=None,
            width=width,
            height=height,
            prefer_canvas=True,
            zoom_control=False,
            attributionControl=False
        )

    # Add roads
    for road_data in roads_data:
        folium.GeoJson(
            road_data['geometry'].__geo_interface__,
            style_function=lambda x, speed=road_data['speed']: {
                'color': speed_color(speed),
                'weight': 4,
                'opacity': 0.9
            }
        ).add_to(m)

    # Add title
    title_html = f'''
    <div style="position: fixed;
                top: 10px; left: 50%; transform: translateX(-50%);
                width: 600px; height: 50px;
                background-color: white; border:2px solid grey;
                z-index:9999; font-size:20px; text-align:center;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
    <b>IDM Traffic Evolution - Step {step}/{num_steps}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Add legend
    legend_html = '''
    <div style="position: fixed;
                bottom: 20px; left: 20px; width: 180px; height: 140px;
                background-color: rgba(255, 255, 255, 0.95);
                border:1px solid #666; z-index:9999;
                font-size:12px; padding: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.3);
                border-radius: 4px;">
    <b style="font-size:13px;">Traffic Speed (km/h)</b><br><br>
    <div style="margin: 4px 0;"><span style="display:inline-block; width:15px; height:15px; background-color:#d73027; margin-right:5px; border:1px solid #666;"></span>&lt; 20 Very Slow</div>
    <div style="margin: 4px 0;"><span style="display:inline-block; width:15px; height:15px; background-color:#fc8d59; margin-right:5px; border:1px solid #666;"></span>20-35 Slow</div>
    <div style="margin: 4px 0;"><span style="display:inline-block; width:15px; height:15px; background-color:#fee08b; margin-right:5px; border:1px solid #666;"></span>35-50 Moderate</div>
    <div style="margin: 4px 0;"><span style="display:inline-block; width:15px; height:15px; background-color:#91cf60; margin-right:5px; border:1px solid #666;"></span>50-70 Fast</div>
    <div style="margin: 4px 0;"><span style="display:inline-block; width:15px; height:15px; background-color:#1a9850; margin-right:5px; border:1px solid #666;"></span>&gt; 70 Very Fast</div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def map_to_image(folium_map, width=1200, height=900):
    """Convert Folium map to PIL Image using selenium."""
    temp_html = 'temp_map.html'
    folium_map.save(temp_html)

    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument(f'--window-size={width},{height}')

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(f'file://{os.path.abspath(temp_html)}')
    time.sleep(2)

    png = driver.get_screenshot_as_png()
    driver.quit()

    os.remove(temp_html)

    img = Image.open(io.BytesIO(png))
    return img


# =============================================================================
# Animation Function
# =============================================================================

def create_idm_animation_folium(edges_gdf, num_roads=100, num_steps=50,
                                frames_to_capture=10,
                                zoom=12, width=1920, height=1080,
                                fps=5, delta_t=0.01,
                                basemap_style='light',
                                output_path='idm_traffic_evolution.gif'):
    """
    Create an animated GIF of IDM traffic evolution using Folium maps.

    Parameters:
    -----------
    edges_gdf : GeoDataFrame
        Road network data
    num_roads : int
        Number of road segments to simulate
    num_steps : int
        Total simulation steps
    frames_to_capture : int
        Number of frames to include in the GIF
    zoom : int
        Map zoom level (10-14 recommended for Singapore)
    width : int
        Map width in pixels
    height : int
        Map height in pixels
    fps : int
        Frames per second in output GIF
    delta_t : float
        Time step for simulation (seconds)
    basemap_style : str
        Basemap style: 'light', 'dark', 'satellite', 'none'
    output_path : str
        Path to save GIF

    Returns:
    --------
    str : Path to the generated GIF file
    """
    print(f"Initializing IDM model with {num_roads} roads...")

    # Create model
    model = IDMTrafficModel(edges_gdf, num_roads=num_roads,
                           vehicles_per_road=10, min_road_length=200,
                           delta_t=delta_t)

    # Calculate map center from roads
    all_roads = list(model.agents_by_type[RoadAgent])
    all_coords = []
    for road in all_roads:
        coords = list(road.geometry.coords)
        all_coords.extend(coords)

    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]
    center = (np.mean(lats), np.mean(lons))

    print(f"Map center: {center}, zoom: {zoom}")

    # Determine which steps to capture
    capture_steps = np.linspace(0, num_steps-1, frames_to_capture, dtype=int)
    frames_data = []

    print(f"Running simulation for {num_steps} steps, capturing {frames_to_capture} frames...")
    print(f"Using delta_t={delta_t}s for realistic IDM dynamics")

    # Run simulation and capture frames
    for step in range(num_steps):
        if step in capture_steps:
            print(f"  Capturing frame at step {step}...")

            # Collect current road speeds
            frame_data = []
            for road in model.agents_by_type[RoadAgent]:
                frame_data.append({
                    'geometry': road.geometry,
                    'speed': road.mean_speed_kph,
                    'highway_type': road.highway_type
                })
            frames_data.append((step, frame_data))

        model.step()

    print(f"Generating map images from Folium (basemap: {basemap_style})...")
    images = []

    for i, (step, roads_data) in enumerate(frames_data):
        print(f"  Rendering frame {i+1}/{len(frames_data)} (step {step})...")

        # Create Folium map with specified basemap style
        folium_map = create_folium_frame(roads_data, step, num_steps,
                                         center=center, zoom=zoom,
                                         width=width, height=height,
                                         basemap_style=basemap_style)

        # Convert to image
        img = map_to_image(folium_map, width=width, height=height)
        images.append(img)

    # Save as GIF
    print(f"Saving animation to {output_path}...")
    duration = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )

    print(f"Animation saved successfully to: {output_path}")
    return output_path


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("IDM Traffic Simulation GIF Generator")
    print("="*70)
    print()

    # =============================================================================
    # CONFIGURATION - Edit these parameters to customize your animation
    # =============================================================================

    # Simulation parameters
    NUM_ROADS = 10000            # Number of road segments to simulate
    NUM_STEPS = 1000             # Total simulation steps (IDM needs moderate length)
    FRAMES_TO_CAPTURE = 60       # Number of frames in the GIF
    DELTA_T = 0.01               # Time step (0.01s recommended for IDM stability)

    # Animation parameters
    FPS = 5                      # Frames per second (1-10 recommended)

    # Map parameters
    ZOOM = 12                    # Map zoom level (10-13 recommended for Singapore)
    BASEMAP_STYLE = 'light'      # Basemap: 'light', 'dark', 'satellite', 'none'

    # Image size (16:9 aspect ratio for Google Slides)
    WIDTH = 1920                 # Width in pixels (1920x1080 = Full HD 16:9)
    HEIGHT = 1080                # Height in pixels

    # Output
    OUTPUT_FILE = 'idm_traffic_evolution.gif'

    # =============================================================================

    # Load Singapore road network
    print("Loading Singapore road network from OpenStreetMap...")
    G = ox.graph_from_place("Singapore", network_type="drive")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Convert to GeoDataFrame
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)

    # Prepare edges DataFrame
    if not {"u", "v", "key"}.issubset(edges_gdf.columns):
        edges_gdf = edges_gdf.reset_index()

    # Select relevant columns
    edge_cols = [c for c in ["u", "v", "key", "geometry", "length", "highway",
                             "maxspeed", "speed_kph", "travel_time"]
                 if c in edges_gdf.columns]
    edges = edges_gdf[edge_cols].copy()
    edges = edges.set_crs(4326) if edges.crs is None else edges.to_crs(4326)

    print(f"Loaded {len(edges)} road segments")
    print()

    # Generate animation
    print(f"Using IDM method with '{BASEMAP_STYLE}' basemap...")
    gif_path = create_idm_animation_folium(
        edges,
        num_roads=NUM_ROADS,
        num_steps=NUM_STEPS,
        frames_to_capture=FRAMES_TO_CAPTURE,
        zoom=ZOOM,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        delta_t=DELTA_T,
        basemap_style=BASEMAP_STYLE,
        output_path=OUTPUT_FILE
    )

    print()
    print("="*70)
    print(f"SUCCESS! Animation saved to: {gif_path}")
    print(f"Total frames: {FRAMES_TO_CAPTURE}")
    print(f"Frame rate: {FPS} fps")
    print(f"Animation duration: ~{FRAMES_TO_CAPTURE/FPS:.1f} seconds")
    print("="*70)
