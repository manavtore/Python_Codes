import numpy as np

# 1. Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])  # Create a 1-dimensional array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # Create a 2-dimensional array
arr_zeros = np.zeros((2, 3))  # Create an array of zeros
arr_ones = np.ones((3, 2))  # Create an array of ones
arr_range = np.arange(0, 10, 2)  # Create an array with values in a specified range
arr_linspace = np.linspace(0, 1, 5)  # Create an array with linearly spaced values

# 2. Array operations
arr_sum = np.sum(arr2)  # Sum of all elements in the array
arr_mean = np.mean(arr2)  # Mean of the array
arr_max = np.max(arr2)  # Maximum value in the array
arr_min = np.min(arr2)  # Minimum value in the array
arr_transpose = np.transpose(arr2)  # Transpose of the array

# 3. Array indexing and slicing
element = arr1[2]  # Access an element at a specific index
subset = arr1[1:4]  # Slice the array to get a subset
arr2_row1 = arr2[0, :]  # Access the entire first row
arr2_col2 = arr2[:, 1]  # Access the entire second column

# 4. Mathematical functions
arr_exp = np.exp(arr1)  # Exponential function
arr_log = np.log(arr1)  # Natural logarithm
arr_sin = np.sin(arr1)  # Sine function
arr_sqrt = np.sqrt(arr1)  # Square root

# 5. Linear algebra
matrix_product = np.dot(arr2, arr2_transpose)  # Matrix multiplication
eigenvalues, eigenvectors = np.linalg.eig(arr2)  # Eigenvalues and eigenvectors

# 6. Random sampling
random_uniform = np.random.rand(3, 2)  # Random values from a uniform distribution
random_normal = np.random.randn(3, 2)  # Random values from a normal distribution
random_choice = np.random.choice(arr1, size=3, replace=False)  # Random sampling from an array

# 7. Statistics
arr_std = np.std(arr2)  # Standard deviation
arr_var = np.var(arr2)  # Variance
arr_corrcoef = np.corrcoef(arr2)  # Correlation coefficient

# 8. Shape manipulation
arr_reshape = np.reshape(arr1, (5, 1))  # Reshape the array
arr_flatten = arr2.flatten()  # Flatten a multi-dimensional array
arr_stack = np.stack((arr1, arr2), axis=0)  # Stack arrays along a new axis

# 9. Boolean operations
arr_greater_than_3 = arr1 > 3  # Boolean array indicating elements greater than 3
arr_logical_and = np.logical_and(arr1 > 2, arr1 < 5)  # Element-wise logical AND

# 10. Constants
pi_value = np.pi  # Mathematical constant pi
euler_constant = np.e  # Euler's number

# Note: This is not an exhaustive list, and there are many more functions available in NumPy.
