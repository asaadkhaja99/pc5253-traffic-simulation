import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class NagelSchreckenberg:
    def __init__(self, length=100, num_cars=35, v_max=5, p_slow=0.3):
        """
        Initialize Nagel-Schreckenberg traffic model
        
        Parameters:
        - length: road length (number of cells)
        - num_cars: number of cars on the road
        - v_max: maximum velocity
        - p_slow: probability of random slowdown
        """
        self.length = length
        self.num_cars = num_cars
        self.v_max = v_max
        self.p_slow = p_slow
        
        # Initialize positions and velocities
        self.positions = np.sort(np.random.choice(length, num_cars, replace=False))
        self.velocities = np.random.randint(0, v_max + 1, num_cars)
        
    def step(self):
        """Execute one time step of the model"""
        # Step 1: Acceleration
        self.velocities = np.minimum(self.velocities + 1, self.v_max)
        
        # Step 2: Slowing down (due to other cars)
        for i in range(self.num_cars):
            # Find distance to next car
            next_car = (i + 1) % self.num_cars
            if next_car == 0:
                gap = (self.positions[next_car] + self.length) - self.positions[i] - 1
            else:
                gap = self.positions[next_car] - self.positions[i] - 1
            
            self.velocities[i] = min(self.velocities[i], gap)
        
        # Step 3: Randomization (random slowing)
        random_slow = np.random.random(self.num_cars) < self.p_slow
        self.velocities[random_slow] = np.maximum(self.velocities[random_slow] - 1, 0)
        
        # Step 4: Car motion
        self.positions = (self.positions + self.velocities) % self.length
        
    def get_occupancy_map(self):
        """Get binary occupancy map of the road (1 = car, 0 = empty)"""
        road = np.zeros(self.length)
        for pos in self.positions:
            road[pos] = 1
        return road

# Simulation parameters
road_length = 100
num_cars = 40
v_max = 5
p_slow = 0.35
num_steps = 100

print("Running simulation...")
# Initialize model
model = NagelSchreckenberg(road_length, num_cars, v_max, p_slow)

# Store history for visualization
history = []
for step in range(num_steps + 1):  # +1 to include t=100
    history.append(model.get_occupancy_map().copy())
    if step < num_steps:
        model.step()
    if step % 20 == 0:
        print(f"  Step {step}/{num_steps}")

print("\n=== Creating Combined Figure ===")

# Create figure with two subplots
fig, axes = plt.subplots(
    1, 2, figsize=(10,4),
    gridspec_kw={'width_ratios': [1.5, 0.8]},  # left:narrow, right:wide
    constrained_layout=True
)

# Panel a) - Emergence snapshots
ax1 = axes[0]
time_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
num_snapshots = len(time_points)


cell_h   = 1.0          # each row's cell height
line_gap = 2.5         # vertical gap between rows (in cell units)

for idx, t in enumerate(time_points):
    road_data = history[t]
    # step size per row = cell height + gap
    y_offset = (num_snapshots - idx - 1) * (cell_h + line_gap)

    for pos in range(road_length):
        fc = 'black' if road_data[pos] > 0 else 'white'
        ax1.add_patch(Rectangle((pos, y_offset), 1, cell_h,
                                facecolor=fc, edgecolor='lightgray', linewidth=0.3))
    ax1.text(-0.5, y_offset + cell_h/2, f'{t}', color='red', fontsize=12, va='center', ha='right')

total_height = num_snapshots * (cell_h + line_gap) - line_gap
ax1.set_ylim(-0.5, total_height + 0.5)
ax1.set_xlim(-8, road_length + 1)

ax1.set_xlabel('Position x', color='blue', fontsize=14)
ax1.set_title('a) Traffic Jam Emergence', fontsize=14, fontweight='bold', loc='left')
ax1.set_yticks([])
# ax1.set_aspect('equal')

# Panel b) - Space-time diagram
ax2 = axes[1]
space_time = np.array(history)

ax2.imshow(space_time, aspect='auto', cmap='binary', origin='lower', 
          interpolation='nearest')

ax2.set_xlabel('Position x', fontsize=14, color='blue')
ax2.set_ylabel('Time t', fontsize=14, color='red')
ax2.set_title('b) Back-Propagation of Jam', fontsize=14, fontweight='bold', loc='left')

# Add arrow annotations
ax2.annotate('', xy=(road_length-5, -3), xytext=(5, -3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.annotate('', xy=(-5, num_steps-5), xytext=(-5, 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('traffic_combined_figure.png', dpi=150, bbox_inches='tight', 
            facecolor='white')
print("✓ Combined figure saved as 'traffic_combined_figure.png'")

print("\n" + "="*50)
print("VISUALIZATION COMPLETE!")
print("="*50)
print(f"\nGenerated file:")
print(f"  traffic_combined_figure.png - Two-panel figure")
print(f"\nSimulation parameters:")
print(f"  Road length: {road_length} cells")
print(f"  Number of cars: {num_cars}")
print(f"  Density: {num_cars/road_length:.1%}")
print(f"  Max velocity: {v_max}")
print(f"  Slowdown probability: {p_slow}")