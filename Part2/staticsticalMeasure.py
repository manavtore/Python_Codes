import numpy as np
import matplotlib.pyplot as plt

# Generate example data
np.random.seed(42)
data1 = np.random.normal(loc=5, scale=2, size=100)
data2 = np.random.normal(loc=5, scale=2, size=100)

# Compute statistical measures
mean1 = np.mean(data1)
mean2 = np.mean(data2)

variance1 = np.var(data1)
variance2 = np.var(data2)

std_dev1 = np.std(data1)
std_dev2 = np.std(data2)

covariance = np.cov(data1, data2)[0, 1]

correlation = np.corrcoef(data1, data2)[0, 1]

standard_error1 = np.std(data1) / np.sqrt(len(data1))
standard_error2 = np.std(data2) / np.sqrt(len(data2))

# Display the distribution of samples graphically
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data1, bins=20, color='blue', alpha=0.7, label='Data 1')
plt.title('Distribution of Data 1')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(data2, bins=20, color='orange', alpha=0.7, label='Data 2')
plt.title('Distribution of Data 2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Print computed statistical measures
print(f"Mean of Data 1: {mean1}")
print(f"Mean of Data 2: {mean2}")
print(f"Variance of Data 1: {variance1}")
print(f"Variance of Data 2: {variance2}")
print(f"Standard Deviation of Data 1: {std_dev1}")
print(f"Standard Deviation of Data 2: {std_dev2}")
print(f"Covariance between Data 1 and Data 2: {covariance}")
print(f"Correlation between Data 1 and Data 2: {correlation}")
print(f"Standard Error of Data 1: {standard_error1}")
print(f"Standard Error of Data 2: {standard_error2}")
