import numpy as np
import matplotlib.pyplot as plt

# Create figure with 3 subplots horizontally
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define coordinate system
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Create grid for vector field (matching the density in reference image)
x_vec = np.linspace(-3, 3, 25)
y_vec = np.linspace(-3, 3, 25)
X_vec, Y_vec = np.meshgrid(x_vec, y_vec)

# 1. RBF Kernel - Exact replication
# Create the characteristic spiral/radial pattern
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
rbf_kernel = np.exp(-0.5 * r**2) * (1 + 0.3 * np.cos(3 * theta))

# Vector field for RBF - radial pattern pointing outward from center
r_vec = np.sqrt(X_vec**2 + Y_vec**2)
theta_vec = np.arctan2(Y_vec, X_vec)
U_rbf = -X_vec * np.exp(-0.3 * r_vec)
V_rbf = -Y_vec * np.exp(-0.3 * r_vec)

# Plot RBF Kernel
axes[0].contourf(X, Y, rbf_kernel, levels=50, cmap='viridis')
axes[0].quiver(X_vec, Y_vec, U_rbf, V_rbf, color='cyan', alpha=0.8, scale=15, width=0.002)
axes[0].set_title('RBF Kernel', fontsize=12, fontweight='bold')
axes[0].set_xlabel('x-axis')
axes[0].set_ylabel('y-axis')
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)

# 2. Wavelet Kernel - Exact replication
# Create the characteristic wavelet pattern with strong central red region
r = np.sqrt(X**2 + Y**2)
wavelet_kernel = np.exp(-0.8 * r**2) + 0.5 * np.exp(-0.2 * ((X-0.5)**2 + (Y-0.5)**2))

# Vector field radiating outward from center with varying intensity
r_vec = np.sqrt(X_vec**2 + Y_vec**2)
U_wavelet = X_vec * (2 - r_vec) * np.exp(-0.2 * r_vec)
V_wavelet = Y_vec * (2 - r_vec) * np.exp(-0.2 * r_vec)

# Plot Wavelet Kernel
axes[1].contourf(X, Y, wavelet_kernel, levels=50, cmap='hot')
axes[1].quiver(X_vec, Y_vec, U_wavelet, V_wavelet, color='darkblue', alpha=0.7, scale=20, width=0.002)
axes[1].set_title('Wavelet Kernel', fontsize=12, fontweight='bold')
axes[1].set_xlabel('x-axis')
axes[1].set_ylabel('y-axis')
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)

# 3. Polynomial Kernel - Exact replication
# Create the characteristic polynomial pattern with radial structure
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
poly_kernel = (1 + 0.3 * (X**2 + Y**2))**2 * np.exp(-0.3 * r) * (1 + 0.2 * np.sin(4 * theta))

# Vector field with radial and tangential components
r_vec = np.sqrt(X_vec**2 + Y_vec**2)
theta_vec = np.arctan2(Y_vec, X_vec)
U_poly = X_vec * np.cos(2 * theta_vec) - Y_vec * np.sin(2 * theta_vec)
V_poly = X_vec * np.sin(2 * theta_vec) + Y_vec * np.cos(2 * theta_vec)

# Plot Polynomial Kernel
axes[2].contourf(X, Y, poly_kernel, levels=50, cmap='RdBu_r')
axes[2].quiver(X_vec, Y_vec, U_poly, V_poly, color='darkred', alpha=0.7, scale=25, width=0.002)
axes[2].set_title('Polynomial Kernel', fontsize=12, fontweight='bold')
axes[2].set_xlabel('x-axis')
axes[2].set_ylabel('y-axis')
axes[2].set_xlim(-3, 3)
axes[2].set_ylim(-3, 3)

# Adjust layout and styling to match reference
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)

# Make background white and adjust overall appearance
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_aspect('equal')
    ax.grid(False)

plt.show()

# Optional: Save the figure
# plt.savefig('kernel_visualizations_exact.png', dpi=300, bbox_inches='tight', facecolor='white')