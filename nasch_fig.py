import numpy as np
import matplotlib.pyplot as plt

class NagelSchreckenberg:
    def __init__(self, length=100, num_cars=35, v_max=5, p_m=0.3):
        """
        Initialize Nagel-Schreckenberg traffic model following Algorithm 1
        
        Parameters:
        - length: road length (number of cells)
        - num_cars: number of cars on the road
        - v_max: maximum velocity
        - p_m: probability of random slowdown (p_m in the algorithm)
        """
        self.length = length
        self.num_cars = num_cars
        self.v_max = v_max
        self.p_m = p_m
        
        # Initialize positions and velocities
        self.positions = np.sort(np.random.choice(length, num_cars, replace=False))
        self.velocities = np.random.randint(0, v_max + 1, num_cars)
        
    def step(self):
        """
        Execute one time step following NaSch Algorithm 1
        Process vehicles from last to first
        """
        old_positions = self.positions.copy()
        new_velocities = self.velocities.copy()
        
        # Process from last to first vehicle
        for i in range(self.num_cars - 1, -1, -1):
            # Step 2: Acceleration - v_i^t = min[v_i^(t-1) + 1, v_max]
            new_velocities[i] = min(new_velocities[i] + 1, self.v_max)
            
            # Step 3: Calculate distance to next car - d_i^t = x_(i-1)^(t-1) - x_i^(t-1) - 1
            next_car = (i + 1) % self.num_cars
            if next_car == 0:  # Wrap around (periodic boundary)
                gap = (self.positions[next_car] + self.length) - self.positions[i] - 1
            else:
                gap = self.positions[next_car] - self.positions[i] - 1
            
            # Step 5-6: Slowing down due to other cars - if v_i^t > d_i^t then v_i^t = d_i^t
            if new_velocities[i] > gap:
                new_velocities[i] = gap
            
            # Step 8-9: Randomization - if p <= p_m and v_i^(t-1) > 0 then v_i^t = v_i^(t-1) - 1
            p = np.random.random()
            if p <= self.p_m and new_velocities[i] > 0:
                new_velocities[i] = new_velocities[i] - 1
        
        # Update velocities
        self.velocities = new_velocities
        
        # Step 11: Car motion - x_i^t = x_i^(t-1) + v_i^t
        self.positions = (self.positions + self.velocities) % self.length
        
        return old_positions, self.positions
    
    def measure_flow(self, measurement_point=0):
        """
        Measure flow: count cars passing a measurement point
        Returns: number of cars that crossed the measurement point
        """
        old_pos, new_pos = self.step()
        
        # Count cars that crossed the measurement point
        flow = 0
        for i in range(self.num_cars):
            # Check if car crossed the measurement point (accounting for periodic boundary)
            if old_pos[i] < measurement_point <= new_pos[i]:
                flow += 1
            elif old_pos[i] > new_pos[i]:  # Car wrapped around
                if old_pos[i] < measurement_point or new_pos[i] >= measurement_point:
                    flow += 1
        
        return flow

def generate_fundamental_diagram(road_length=100, v_max=5, p_m=0.3, 
                                 num_runs=5, warmup_steps=200, measure_steps=500):
    """
    Generate fundamental diagram by simulating at different densities
    
    Parameters:
    - road_length: length of the road
    - v_max: maximum velocity
    - p_m: probability of random slowdown
    - num_runs: number of runs to average over for each density
    - warmup_steps: steps to reach steady state
    - measure_steps: steps to measure flow
    
    Returns:
    - densities: array of densities
    - flows: array of average flows
    """
    # Test different numbers of cars (densities)
    car_numbers = range(1, road_length, 2)  # From 1 to road_length-1, step by 2
    
    densities = []
    flows = []
    
    for num_cars in car_numbers:
        density = num_cars / road_length
        
        # Run multiple simulations for this density
        flow_measurements = []
        for run in range(num_runs):
            model = NagelSchreckenberg(road_length, num_cars, v_max, p_m)
            
            # Warmup period to reach steady state
            for _ in range(warmup_steps):
                model.step()
            
            # Measurement period
            total_flow = 0
            for _ in range(measure_steps):
                flow = model.measure_flow()
                total_flow += flow
            
            # Average flow per time step
            avg_flow = total_flow / measure_steps
            flow_measurements.append(avg_flow)
        
        # Store results
        densities.append(density)
        flows.append(np.mean(flow_measurements))
    
    return np.array(densities), np.array(flows)


# =============================================================================
# MAIN EXECUTION - Generate fundamental diagrams for different p_m values
# =============================================================================

print("="*60)
print("GENERATING FUNDAMENTAL DIAGRAMS (NaSch Algorithm 1)")
print("="*60)

# Simulation parameters
road_length = 100
v_max = 5
p_m_values = [0.01, 0.1, 0.4, 0.5, 0.75]  # Different slowdown probabilities
colors = ['#C41E3A', '#808080', '#FF8C00', '#32CD32', '#1E90FF']  # Red, Gray, Orange, Green, Blue

# Create plot
fig, ax = plt.subplots(figsize=(8, 6))

# Generate and plot for each p_m value
for p_m, color in zip(p_m_values, colors):
    print(f"\nGenerating for p = {p_m}...")
    
    densities, flows = generate_fundamental_diagram(
        road_length=road_length, 
        v_max=v_max, 
        p_m=p_m,
        num_runs=3,
        warmup_steps=200,
        measure_steps=500
    )
    
    # Normalize flow by v_max to get J (dimensionless flow)
    normalized_flows = flows / v_max
    
    # Plot
    ax.plot(densities, normalized_flows, '-', linewidth=2.5, 
            color=color, label=f'p={p_m}')
    
    max_flow = np.max(normalized_flows)
    max_density = densities[np.argmax(normalized_flows)]
    print(f"  Max J = {max_flow:.3f} at ρ = {max_density:.3f}")

# Formatting
ax.set_xlabel('ρ', fontsize=20, fontweight='bold')
ax.set_ylabel('J', fontsize=20, fontweight='bold', rotation=0, labelpad=15)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(fontsize=14, loc='upper right', frameon=True, fancybox=False, 
          edgecolor='black')

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=12)

# Make the plot look cleaner
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('fundamental_diagram_nasch.png', dpi=150, bbox_inches='tight', 
            facecolor='white')
print("\n" + "="*60)
print("✓ Fundamental diagram saved as 'fundamental_diagram_nasch.png'")
print("="*60)
# plt.show()

print("\n" + "="*60)
print("ALGORITHM DETAILS")
print("="*60)
print("\nNaSch Algorithm implemented:")
print("1. Process vehicles from LAST to FIRST")
print("2. Acceleration: v_i^t = min[v_i^(t-1) + 1, v_max]")
print("3. Calculate gap: d_i^t = x_(i-1)^(t-1) - x_i^(t-1) - 1")
print("4. Generate random p ∈ [0, 1]")
print("5-6. Slowing down: if v_i^t > d_i^t then v_i^t = d_i^t")
print("7-9. Randomization: if p ≤ p_m and v_i^(t-1) > 0 then v_i^t = v_i^(t-1) - 1")
print("10. Position update: x_i^t = x_i^(t-1) + v_i^t")
print("\nKey observations:")
print("- Lower p_m (less randomness) → higher maximum flow")
print("- p_m=0.01 shows nearly deterministic behavior with sharp peak")
print("- Higher p_m (more randomness) → smoother curves, lower capacity")
print("- All curves show free flow → congested transition")