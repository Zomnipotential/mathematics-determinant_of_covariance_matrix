import matplotlib.pyplot as plt
import numpy as np

# Function to generate data given covariance matrix
def generate_data(cov, n=400):
    mean = [0, 0]
    return np.random.multivariate_normal(mean, cov, n)

# Define three covariance matrices
cov_large = np.array([[3, 0.5],
                      [0.5, 2]])  # Larger spread, larger determinant
cov_small = np.array([[1, 0.9],
                      [0.9, 1]])  # Highly correlated, small determinant
cov_zero = np.array([[1, 1],
                     [1, 1]])     # Perfect correlation, determinant = 0

# Generate datasets
data_large = generate_data(cov_large)
data_small = generate_data(cov_small)
data_zero = np.dot(np.random.randn(400, 1), np.array([[1, 1]]))  # Collapse to line

# Plot
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].scatter(data_large[:, 0], data_large[:, 1], alpha=0.6, color='green')
axs[0].set_title(f"Large det(Σ): {np.linalg.det(cov_large):.2f}")
axs[0].set_xlim(-6, 6); axs[0].set_ylim(-6, 6); axs[0].set_aspect('equal')

axs[1].scatter(data_small[:, 0], data_small[:, 1], alpha=0.6, color='blue')
axs[1].set_title(f"Small det(Σ): {np.linalg.det(cov_small):.2f}")
axs[1].set_xlim(-6, 6); axs[1].set_ylim(-6, 6); axs[1].set_aspect('equal')

axs[2].scatter(data_zero[:, 0], data_zero[:, 1], alpha=0.6, color='red')
axs[2].set_title("det(Σ) = 0 (Perfect correlation)")
axs[2].set_xlim(-6, 6); axs[2].set_ylim(-6, 6); axs[2].set_aspect('equal')

for ax in axs:
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)

plt.tight_layout()
plt.show()