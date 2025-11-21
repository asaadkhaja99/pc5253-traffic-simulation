"""
NaSch (Nagel-Schreckenberg) Traffic Simulation GIF Generator

This script generates an animated GIF showing traffic evolution over time
using the Nagel-Schreckenberg Cellular Automaton model with Mesa-Geo framework.

Note: AI Tools:
Claude code was used to assist with the implementation of this script. 
All final content was reviewed and edited to ensure accuracy.
"""

# Geospatial libraries
import osmnx as ox
import numpy as np
import geopandas as gpd

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
# NaSch Model Functions
# =============================================================================

def nasch_step(road, v_max=5, p_slow=0.3, rng=None):
    """
    Single step of Nagel-Schreckenberg model.

    Parameters:
    -----------
    road : ndarray
        Array where -1 = empty, >=0 = vehicle with that velocity
    v_max : int
        Maximum velocity
    p_slow : float
        Probability of random slowdown
    rng : numpy.random.Generator
        Random number generator

    Returns:
    --------
    ndarray : Updated road state
    """
    if rng is None:
        rng = np.random.default_rng()

    num_cells = len(road)
    new_road = -np.ones(num_cells, dtype=np.int16)
    car_positions = np.where(road >= 0)[0]

    if len(car_positions) == 0:
        return new_road

    # Calculate gaps to next vehicle
    next_positions = np.roll(car_positions, -1)
    gaps = (next_positions - car_positions - 1) % num_cells

    # Process each vehicle
    update_order = rng.permutation(len(car_positions))
    for idx in update_order:
        pos = car_positions[idx]
        v = road[pos]

        # Acceleration
        v = min(v + 1, v_max)

        # Slowing (due to other vehicles)
        v = min(v, gaps[idx])

        # Randomization
        if v > 0 and rng.random() < p_slow:
            v -= 1

        # Car movement
        new_pos = (pos + v) % num_cells
        new_road[new_pos] = v

    return new_road


# =============================================================================
# Agent Classes
# =============================================================================

class VehicleAgent(mg.GeoAgent):
    """Vehicle agent for visualization."""

    def __init__(self, model, geometry, crs, road_segment, position, velocity, vehicle_index):
        super().__init__(model, geometry, crs)
        self.road_segment = road_segment
        self.position = position
        self.velocity = velocity
        self.vehicle_index = vehicle_index

    def update_position_on_road(self):
        """Update geometry based on normalized position along road."""
        road_geom = self.road_segment.geometry
        point_on_road = road_geom.interpolate(self.position, normalized=True)
        self.geometry = point_on_road

    def step(self):
        """Update vehicle position."""
        self.update_position_on_road()


class RoadAgent(mg.GeoAgent):
    """Road segment agent managing NaSch CA."""

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

        self.set_nasch_params()

        # NaSch state
        self.road = None
        self.num_cells = None
        self.cell_length_m = 7.5  # Standard cell size
        self.vehicles = []

        # Performance metrics
        self.mean_speed_kph = speed_kph
        self.speed_ratio = 1.0

    def set_nasch_params(self):
        """Set NaSch parameters based on highway type."""
        highway_params = {
            'motorway': {'v_max': 7, 'p_slow': 0.2, 'density': 0.10, 'base_ratio': 0.75},
            'trunk': {'v_max': 6, 'p_slow': 0.25, 'density': 0.12, 'base_ratio': 0.80},
            'primary': {'v_max': 5, 'p_slow': 0.30, 'density': 0.15, 'base_ratio': 0.85},
            'secondary': {'v_max': 4, 'p_slow': 0.35, 'density': 0.20, 'base_ratio': 0.90},
            'tertiary': {'v_max': 3, 'p_slow': 0.40, 'density': 0.25, 'base_ratio': 0.95},
            'residential': {'v_max': 2, 'p_slow': 0.45, 'density': 0.30, 'base_ratio': 0.95}
        }

        params = highway_params.get(self.highway_type, highway_params['residential'])
        self.v_max = params['v_max']
        self.p_slow = params['p_slow']
        self.density = params['density']
        self.base_ratio = params['base_ratio']

    def initialize_vehicles(self):
        """Initialize NaSch cellular automaton."""
        # Discretize road into cells
        self.num_cells = max(5, int(self.length_m / self.cell_length_m))
        self.road = -np.ones(self.num_cells, dtype=np.int16)

        # Place vehicles
        num_cars = int(self.density * self.num_cells)
        num_cars = min(num_cars, self.num_vehicles)  # Respect requested vehicle count

        if num_cars > 0:
            car_positions = self.model.rng.choice(self.num_cells, size=num_cars, replace=False)
            self.road[car_positions] = self.model.rng.integers(0, self.v_max + 1, size=num_cars)

        # Run warmup
        self._run_warmup()

    def _run_warmup(self, steps=200):
        """Run warmup simulation to reach steady state."""
        for _ in range(steps):
            self.road = nasch_step(self.road, self.v_max, self.p_slow, self.model.rng)

        # Calculate initial speed
        self._update_speed_metrics()

    def _update_speed_metrics(self):
        """Update speed metrics from current road state."""
        moving_vehicles = self.road[self.road >= 0]

        if len(moving_vehicles) > 0:
            # Mean velocity in cells per step
            mean_v_cells = moving_vehicles.mean()

            # Convert to km/h: cells/step * cell_length_m * 3.6
            mean_speed_kph = mean_v_cells * (self.cell_length_m / 1000) * 3600 / self.model.delta_t

            # Calculate ratio
            speed_ratio = mean_speed_kph / self.speed_kph if self.speed_kph > 0 else 1.0
            speed_ratio = np.clip(speed_ratio, 0.3, 1.2)

            self.mean_speed_kph = mean_speed_kph
            self.speed_ratio = speed_ratio
        else:
            self.mean_speed_kph = 0.0
            self.speed_ratio = 0.0

    def step(self):
        """Update NaSch CA and vehicles."""
        # Update cellular automaton
        self.road = nasch_step(self.road, self.v_max, self.p_slow, self.model.rng)

        # Update metrics
        self._update_speed_metrics()

        # Update vehicle agent positions
        car_positions = np.where(self.road >= 0)[0]
        for i, vehicle in enumerate(self.vehicles):
            if i < len(car_positions):
                cell_pos = car_positions[i]
                vehicle.position = cell_pos / self.num_cells  # Normalized position
                vehicle.velocity = self.road[cell_pos] / self.v_max if self.v_max > 0 else 0


class NaSchTrafficModel(mesa.Model):
    """Mesa-Geo model for NaSch traffic simulation."""

    def __init__(self, edges_gdf, num_roads=50, vehicles_per_road=10,
                 delta_t=1.0, min_road_length=200):
        super().__init__()

        self.num_roads = num_roads
        self.vehicles_per_road = vehicles_per_road
        self.delta_t = delta_t  # Time per step in seconds
        self.min_road_length = min_road_length

        self.space = mg.GeoSpace(warn_crs_conversion=False)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Mean Speed (km/h)": lambda m: np.mean([r.mean_speed_kph for r in m.agents_by_type[RoadAgent]]) if m.agents_by_type[RoadAgent] else 0,
                "Mean Speed Ratio": lambda m: np.mean([r.speed_ratio for r in m.agents_by_type[RoadAgent]]) if m.agents_by_type[RoadAgent] else 0,
                "Total Vehicles": lambda m: sum([(r.road >= 0).sum() for r in m.agents_by_type[RoadAgent]]) if m.agents_by_type[RoadAgent] else 0,
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
                'motorway': 20,
                'trunk': 15,
                'primary': 12,
                'secondary': 10,
                'tertiary': 8,
                'residential': 6
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
            car_positions = np.where(road.road >= 0)[0]
            for i, cell_pos in enumerate(car_positions):
                position_normalized = cell_pos / road.num_cells
                point_on_road = road.geometry.interpolate(position_normalized, normalized=True)

                vehicle = VehicleAgent(
                    model=self,
                    geometry=point_on_road,
                    crs=self.space.crs,
                    road_segment=road,
                    position=position_normalized,
                    velocity=road.road[cell_pos] / road.v_max if road.v_max > 0 else 0,
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
    <b>NaSch Traffic Evolution - Step {step}/{num_steps}</b>
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

def create_nasch_animation_folium(edges_gdf, num_roads=100, num_steps=50,
                                  frames_to_capture=10,
                                  zoom=12, width=1920, height=1080,
                                  fps=5, delta_t=1.0,
                                  basemap_style='light',
                                  output_path='nasch_traffic_evolution.gif'):
    """
    Create an animated GIF of NaSch traffic evolution using Folium maps.

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
        Time step for simulation (seconds per CA step)
    basemap_style : str
        Basemap style: 'light', 'dark', 'satellite', 'none'
    output_path : str
        Path to save GIF

    Returns:
    --------
    str : Path to the generated GIF file
    """
    print(f"Initializing NaSch model with {num_roads} roads...")

    # Create model
    model = NaSchTrafficModel(edges_gdf, num_roads=num_roads,
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
    print("NaSch Traffic Simulation GIF Generator")
    print("="*70)
    print()

    # =============================================================================
    # CONFIGURATION - Edit these parameters to customize your animation
    # =============================================================================

    # Simulation parameters
    NUM_ROADS = 10000               # Number of road segments to simulate
    NUM_STEPS = 1000              # Total CA steps (NaSch needs more steps)
    FRAMES_TO_CAPTURE = 60       # Number of frames in the GIF
    DELTA_T = 1.0                # Time per CA step (seconds)

    # Animation parameters
    FPS = 5                      # Frames per second (1-10 recommended)

    # Map parameters
    ZOOM = 12                    # Map zoom level (10-13 recommended for Singapore)
    BASEMAP_STYLE = 'light'      # Basemap: 'light', 'dark', 'satellite', 'none'

    # Image size (16:9 aspect ratio for Google Slides)
    WIDTH = 1920                 # Width in pixels
    HEIGHT = 1080                # Height in pixels

    # Output
    OUTPUT_FILE = 'nasch_traffic_evolution.gif'

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
    print(f"Using NaSch CA method with '{BASEMAP_STYLE}' basemap...")
    gif_path = create_nasch_animation_folium(
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
