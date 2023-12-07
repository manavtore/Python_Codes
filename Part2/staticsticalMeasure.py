data = [4, 2, 7, 1, 8, 5, 2, 7, 2, 6]

# Mean
mean = 0
for num in data:
    mean += num
mean /= len(data)
print(f"Mean: {mean}")

# Mode
counts = {}
for num in data:
    if num in counts:
        counts[num] += 1
    else:
        counts[num] = 1
mode = None
max_count = 0
for num, count in counts.items():
    if count > max_count:
        mode = num
        max_count = count
print(f"Mode: {mode}")

# Median
sorted_data = sorted(data)
n = len(sorted_data)
if n % 2 == 0:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
else:
    median = sorted_data[n // 2]
print(f"Median: {median}")

# Standard Deviation
variance = sum((x - mean)**2 for x in data) / (len(data) - 1)
std_dev = variance**0.5
print(f"Standard Deviation: {std_dev}")

# Covariance
other_data = [3, 6, 8, 1, 9, 4, 5, 7, 2, 6]
covariance = sum((data[i] - mean) * (other_data[i] - np.mean(other_data)) for i in range(len(data))) / (len(data) - 1)
print(f"Covariance: {covariance}")

# Correlation
correlation = covariance / (std_dev * np.std(other_data))
print(f"Correlation: {correlation}")

# Standard Error
se_mean = std_dev / (len(data)**0.5)
print(f"Standard Error of the Mean: {se_mean}")

plt.hist(data, bins=range(min(data), max(data) + 1), edgecolor='black', alpha=0.7)
plt.title('Histogram of Example Data')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
