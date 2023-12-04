import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic class test results data (assuming a normal distribution)
mean_score = 70
std_dev_score = 10
num_students = 1000

class_test_results = np.random.normal(loc=mean_score, scale=std_dev_score, size=num_students)

# Plot the Normal Distribution
plt.figure(figsize=(10, 6))
plt.hist(class_test_results, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

# Plot the probability density function (PDF) for the normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_score, std_dev_score)
plt.plot(x, p, 'k', linewidth=2)

plt.title('Normal Distribution of Class Test Results')
plt.xlabel('Test Scores')
plt.ylabel('Frequency')

# Identify Skewness and Kurtosis
skewness = skew(class_test_results)
kurt = kurtosis(class_test_results)

plt.text(50, 0.025, f'Skewness: {skewness:.2f}\nKurtosis: {kurt:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.show()
