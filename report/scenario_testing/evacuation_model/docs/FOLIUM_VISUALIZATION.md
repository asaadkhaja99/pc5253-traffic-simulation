# Folium Interactive Evacuation Visualization

Interactive web-based visualization for evacuation scenarios using Folium with time slider.

## Features

### Interactive Map
- **OpenStreetMap basemap** - CartoDB Positron style background
- **Time slider** - Scrub through evacuation timeline
- **Click interactions** - Click any road to see detailed statistics
- **Pan & zoom** - Standard map controls

### Visual Encoding

**Road Colors** (by congestion level):
- **Green** (#2ecc71) - Free flow (speed_ratio ≥ 80%)
- **Yellow** (#f1c40f) - Light congestion (60-80%)
- **Orange** (#f39c12) - Moderate congestion (30-60%)
- **Red** (#e74c3c) - Severe congestion (< 30%)

**Line Width** - Road occupancy density (2-8 pixels)

### Statistics Overlay
- Scenario name
- Total evacuees
- Capture interval
- Current timestep

### Road Popups (on click)
- Congestion level
- Number of vehicles
- Occupancy percentage
- Current speed (km/h)
- Speed ratio
- Road length

## Usage

### Quick Test (500 agents, 1 hour)
```bash
python evacuation_model/run_folium_visualization.py --quick --scenario simultaneous
```

### All Scenarios (2000 agents, 2 hours)
```bash
python evacuation_model/run_folium_visualization.py
```

### Single Scenario
```bash
python evacuation_model/run_folium_visualization.py --scenario staged
```

### Custom Parameters
```bash
python evacuation_model/run_folium_visualization.py \
    --num-agents 1000 \
    --interval 30 \
    --max-steps 3600
```

## Output

HTML files saved to: `output/evacuation/folium/<scenario>/`

Each scenario produces:
- `<scenario>_interactive.html` - Interactive map with time slider

Open in any web browser to explore interactively!

## Architecture

### Files

**[visualize_folium.py](visualize_folium.py)**
- `FoliumEvacuationVisualizer` - Main visualization class
- `create_folium_animation()` - Simulation runner with visualization

**[run_folium_visualization.py](run_folium_visualization.py)**
- CLI entry point
- Scenario orchestration
- Comparative analysis

### Data Flow

1. **Simulation Loop**
   ```
   For each timestep:
     - Run model.step()
     - Capture road states every N seconds
     - Store: coords, color, weight, stats
   ```

2. **Map Generation**
   ```
   - Create Folium base map
   - Build GeoJSON FeatureCollection with timestamps
   - Add TimestampedGeoJson layer
   - Add overlays (stats, legend)
   - Save to HTML
   ```

3. **User Interaction**
   ```
   - Open HTML in browser
   - Use time slider to navigate
   - Click roads for details
   - Pan/zoom to explore
   ```

## Advantages Over Static Visualization

✅ **Interactive** - Pan, zoom, click
✅ **Temporal** - Time slider for animation
✅ **Web-based** - No dependencies to view
✅ **Performance** - Better for large datasets
✅ **Tooltips** - Rich contextual information
✅ **Native OSM** - No projection issues

## Technical Details

### Dependencies
- `folium` - Interactive maps
- `folium.plugins.TimestampedGeoJson` - Temporal animation
- `geopandas` - Spatial data handling
- `numpy`, `pandas` - Data processing

### Projection
- Uses WGS84 (EPSG:4326) natively
- No Web Mercator conversion needed
- Coordinates stored as [lon, lat]

### Time Format
- ISO 8601 timestamps (`YYYY-MM-DDTHH:MM:SS`)
- Base date: 2024-01-01 00:00:00
- Timestep seconds added to base

### Performance
- Optimized with `prefer_canvas=True`
- Efficient GeoJSON rendering
- Handles 1000+ road segments smoothly

## Customization

### Change Basemap
```python
# In visualize_folium.py
m = folium.Map(
    location=[self.center_lat, self.center_lon],
    zoom_start=13,
    tiles='OpenStreetMap',  # or 'Stamen Terrain', 'Stamen Toner'
    prefer_canvas=True
)
```

### Adjust Congestion Thresholds
```python
# In capture_timestep()
if speed_ratio < 0.3:      # Severe (red)
elif speed_ratio < 0.6:    # Moderate (orange)
elif speed_ratio < 0.8:    # Light (yellow)
else:                      # Free (green)
```

### Change Capture Interval
```bash
python evacuation_model/run_folium_visualization.py --interval 30  # Every 30s
```

## Example Output

After running, you'll see:
```
FOLIUM VISUALIZATION STUDY COMPLETE
================================================================================

SIMULTANEOUS:
  HTML: output/evacuation/folium/simultaneous/simultaneous_interactive.html
  T95: 2714.0s (45.2 min)
  Evacuated: 1900/2000

STAGED:
  HTML: output/evacuation/folium/staged/staged_interactive.html
  T95: 3168.0s (52.8 min)
  Evacuated: 1950/2000

CONTRAFLOW:
  HTML: output/evacuation/folium/contraflow/contraflow_interactive.html
  T95: 2316.0s (38.6 min)
  Evacuated: 1980/2000

================================================================================
Open the HTML files in your browser to explore the interactive maps!
================================================================================
```

## Browser Compatibility

Tested and working on:
- Chrome/Chromium
- Firefox
- Safari
- Edge

Requires JavaScript enabled.
