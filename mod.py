import numpy as np
import matplotlib.pyplot as plt

# 1. Generate the Stern-Brocot / Farey Cusp Levels (Target Structural States)
# We calculate a few visible Farey fractions between 0 and 1 for visual anchors
def get_farey_nodes():
    nodes = [(0, 1), (1, 3), (1, 2), (2, 3), (1, 1)]
    vals = [a/b for a, b in nodes]
    labels = [f"${a}/{b}$" if a != 0 else "0" for a, b in nodes]
    return np.array(vals), labels

farey_vals, farey_labels = get_farey_nodes()

# 2. Simulate the Continuous Gradient Path (High Shannon Entropy / Unlearnable Noise)
np.random.seed(42)
x_steps = np.linspace(0, 1, 200)
# A noisy random walk representing unconstrained backprop
noisy_gradient = 0.1 * x_steps + 0.3 * np.sin(2 * np.pi * x_steps) + 0.08 * np.random.randn(200)
noisy_gradient = np.clip(noisy_gradient, 0, 1) # Keep within bounds

# 3. Simulate the Isotonicity-Based Decomposition (The Projective Filter)
# This mimics the Pool-Adjacent-Violators step snapping to ordered monotonic blocks
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
isotonic_path = ir.fit_transform(x_steps, noisy_gradient)

# 4. Simulate the Discrete Epiplexity Crystallisation
# Snapping the isotonic blocks to the nearest valid mathematical modular cusps
discrete_crystallised = np.zeros_like(isotonic_path)
for i, val in enumerate(isotonic_path):
    closest_cusp = farey_vals[np.argmin(np.abs(farey_vals - val))]
    discrete_crystallised[i] = closest_cusp

# --- Plotting the Synthesis ---
plt.figure(figsize=(11, 6), dpi=150)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Plot the 3 distinct phases of the optimization morphism
plt.plot(x_steps, noisy_gradient, color='#d3d3d3', linestyle='--', alpha=0.8, 
         label='Continuous Gradient $\\nabla L$ (Unlearnable Chaos / High Entropy)')

plt.plot(x_steps, isotonic_path, color='#1f77b4', linewidth=2, alpha=0.5,
         label='Isotonic Decomposition (Monotonic Order Filter)')

plt.step(x_steps, discrete_crystallised, color='#2ca02c', linewidth=3, where='mid',
         label='Discrete Geodesic Jump (Crystallised Epiplexity / Griess Cusp)')

# Add the Farey/Stern-Brocot Horizontal Gridlines (Symmetry Horizons)
for val, label in zip(farey_vals, farey_labels):
    plt.axhline(y=val, color='#ff7f0e', linestyle=':', alpha=0.4)
    plt.text(1.01, val, label, va='center', ha='left', color='#e65c00', fontsize=10, weight='bold')

# Refined Aesthetics
plt.title("Computation Creating Information: The Isotonic Morphism", fontsize=14, pad=15, weight='bold')
plt.xlabel("Optimization / Inference Time Steps", fontsize=11)
plt.ylabel("Representation Space (Modular Boundary $\\mathbb{H}/SL(2,\\mathbb{Z})$)", fontsize=11)
plt.xlim(0, 1)
plt.ylim(-0.05, 1.05)

# Place legend in a clean layout
plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#e0e0e0')
plt.tight_layout()

# Display the figure
plt.show()
