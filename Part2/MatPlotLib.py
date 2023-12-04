import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# Data Visualization
# Pairplot for pairwise feature comparison
sns.pairplot(df, hue='Target', palette='viridis')
plt.suptitle('Pairwise Feature Comparison')
plt.show()

# Boxplot for each feature
plt.figure(figsize=(12, 6))
for i, feature in enumerate(feature_names):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(x='Target', y=feature, data=df, palette='viridis')
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Feature Extraction Analysis using PCA
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.show()

# Display the contribution of each principal component
components_df = pd.DataFrame({
    'Principal Component': range(1, len(explained_variance_ratio) + 1),
    'Explained Variance Ratio': explained_variance_ratio,
    'Cumulative Variance Ratio': cumulative_variance_ratio
})
print(components_df)
