import numpy as np
import matplotlib.pyplot as plt

# Define the function
def V(s):
    return 16.8 * (np.tanh(0.0860 * (s - 25)) + 0.913)

# Define the range of s (spacing)
s = np.linspace(0, 80, 400)
v = V(s)
# Determine regions for highlighting
vmax = np.max(v)
low_speed_region = v < 3
high_speed_region = (vmax - v) < 3


# Plot
plt.figure(figsize=(6,4))
plt.plot(s, v, linewidth=2)
# Fill entire vertical region based on x condition
ymin, ymax = np.min(v), vmax + 5  # define fill range (slightly above curve for visibility)

# Red transparent region where V(s) < 3
plt.fill_between(s, ymin, ymax, where=low_speed_region, color='red', alpha=0.15,)
                #  label=r'$V(s) < 3$ m/s (Congested)')

# Green transparent region where Vmax - V(s) < 3
plt.fill_between(s, ymin, ymax, where=high_speed_region, color='green', alpha=0.15,)
                #  label=r'$V_{\max}-V(s) < 3$ m/s (Free flow)')
plt.axvline(x=25, color='red', linestyle='--', label='Safe distance (25 m)')
plt.axhline(y=16.8*1.913, color='green', linestyle='--', label=f'Max velocity ({16.8*1.913:.2f} m/s)')
plt.legend(loc='lower right')
plt.xlabel(r'Spacing $s$ (m)')
plt.ylabel(r'Optimal velocity $V(s)$ (m/s)')
plt.savefig('optimal_velocity_function.png', dpi=300)
