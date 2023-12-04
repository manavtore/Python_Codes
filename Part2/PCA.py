import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# 1. Data Correlation
correlation_matrix = np.corrcoef(X, rowvar=False)
print("Correlation Matrix:")
print(correlation_matrix)

# 2. Data Rescaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data:")
print(X_scaled[:5])  # Display the first 5 rows

# 3. Dimensionality Reduction using PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
print("\nPrincipal Components:")
print(X_pca[:5])  # Display the reduced data

# Plot the original and reduced data
plt.figure(figsize=(12, 6))

# Original Data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Reduced Data after PCA
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Reduced Data after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# Explained Variance Ratio
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)
