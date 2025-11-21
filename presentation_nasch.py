import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
num_cars = 40  # Higher density to show jams better
v_max = 5
p_slow = 0.35
num_steps = 100

print("Running simulation...")
# Initialize model
model = NagelSchreckenberg(road_length, num_cars, v_max, p_slow)

# Store history for visualization
history = []
for step in range(num_steps):
    history.append(model.get_occupancy_map().copy())
    model.step()
    if step % 20 == 0:
        print(f"  Step {step}/{num_steps}")

print("\n=== Creating Space-Time Diagram ===")
# Create space-time diagram (like your reference image)
space_time = np.array(history)

fig, ax = plt.subplots(figsize=(4,4))
# Display as binary (black = car, white = empty)
ax.imshow(space_time, aspect='auto', cmap='binary', origin='lower', 
          interpolation='nearest')

# Add labels similar to reference image
ax.set_xlabel('Position x', fontsize=18, color='blue')
ax.set_ylabel('Time t', fontsize=18, color='red')
# ax.set_title('Space-Time Diagram: Back-Propagating Traffic Jams\n(Nagel-Schreckenberg Model)', 
#              fontsize=14, fontweight='bold')

# Add arrows and labels like in reference
# ax.text(0, -5, 'x=0', fontsize=16, color='blue', ha='left')
# ax.text(road_length, -5, f'x={road_length}', fontsize=16, color='blue', ha='right')
# ax.text(-8, 0, 't=0', fontsize=16, color='red', va='bottom')
# ax.text(-8, num_steps-1, f't={num_steps}', fontsize=16, color='red', va='top')

# Add arrow annotations
ax.annotate('', xy=(road_length-5, -3), xytext=(5, -3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=(-5, num_steps-5), xytext=(-5, 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('traffic_spacetime_diagram.png', dpi=150, bbox_inches='tight', 
            facecolor='white')
print("✓ Space-time diagram saved as 'traffic_spacetime_diagram.png'")
plt.close()

print("\n=== Creating Emergence Animation ===")
# Create emergence animation GIF
fig, ax = plt.subplots(figsize=(8, 2))

def animate(frame):
    ax.clear()
    
    # Draw road
    road_data = history[frame]
    
    # Create visualization
    for pos in range(road_length):
        if road_data[pos] > 0:
            # Draw car as black square
            ax.add_patch(plt.Rectangle((pos, 0), 1, 1, facecolor='black', 
                                       edgecolor='none'))
        else:
            # Draw empty space
            ax.add_patch(plt.Rectangle((pos, 0), 1, 1, facecolor='white', 
                                       edgecolor='lightgray', linewidth=0.3))
    
    ax.set_xlim(-1, road_length + 1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Position', fontsize=14)
    ax.set_title(f'Traffic Jam Emergence (t = {frame})', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Add density info
    density = np.sum(road_data) / road_length * 100
    # ax.text(0.02, 0.95, f'Density: {density:.1f}%\nCars: {num_cars}\nMax Speed: {v_max}', 
    #         transform=ax.transAxes, fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Create animation
print("Generating GIF... This may take a moment.")
anim = FuncAnimation(fig, animate, frames=num_steps, interval=100, repeat=True)

# Save as GIF
writer = PillowWriter(fps=10)
anim.save('traffic_jam_emergence.gif', writer=writer, dpi=100)
print("✓ Emergence animation saved as 'traffic_jam_emergence.gif'")

plt.close()

print("\n" + "="*50)
print("SIMULATION COMPLETE!")
print("="*50)
print(f"\nGenerated files:")
print(f"  1. traffic_spacetime_diagram.png - Shows back-propagating jams")
print(f"  2. traffic_jam_emergence.gif - Animated jam formation")
print(f"\nSimulation parameters:")
print(f"  Road length: {road_length} cells")
print(f"  Number of cars: {num_cars}")
print(f"  Density: {num_cars/road_length:.1%}")
print(f"  Max velocity: {v_max}")
print(f"  Slowdown probability: {p_slow}")
print(f"\nThe diagonal black bands in the space-time diagram show")
print(f"traffic jams propagating backwards (upstream) through traffic!")


