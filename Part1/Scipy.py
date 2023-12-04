import Part1.numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm, chi2, pearsonr
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.linalg import svd
from scipy.spatial.distance import euclidean
from scipy.special import comb, erf
from scipy.constants import c, G
from scipy.fftpack import fft, ifft
from scipy.integrate import quad, simps
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.ndimage import gaussian_filter, binary_erosion
from scipy.sparse import csr_matrix, diags
from scipy.io import loadmat, savemat

# 1. Minimize a Function
minimize_result = minimize()

# 2. Curve Fitting
params, covariance = curve_fit()

# 3. Probability Density Function (PDF) of a Normal Distribution
x_values, pdf_values = norm.pdf()

# 4. Cumulative Distribution Function (CDF) of a Chi-Square Distribution
x_values_chi2, cdf_values_chi2 = chi2.cdf()

# 5. Pearson Correlation Coefficient
corr_coeff, p_value = pearsonr()

# 6. Interpolation
interp_func = interp1d()

# 7. Find Peaks in a Signal
peaks, _ = find_peaks()

# 8. Singular Value Decomposition
U, S, Vt = svd()

# 9. Euclidean Distance
distance = euclidean()

# 10. Binomial Coefficient
binomial_coeff = comb()

# 11. Error Function
error_function = erf()

# 12. Speed of Light and Gravitational Constant
speed_of_light = c
gravitational_constant = G

# 13. Fast Fourier Transform (FFT)
fft_result = fft()

# 14. Inverse Fast Fourier Transform (IFFT)
ifft_result = ifft()

# 15. Numerical Integration (Quad)
area, error = quad()

# 16. Numerical Integration (Simpson's Rule)
area_simpson = simps()

# 17. Hierarchical Clustering (Linkage)
clusters = linkage()

# 18. Dendrogram Plotting
dendrogram_plot = dendrogram()

# 19. Gaussian Filtering
smoothed_array = gaussian_filter()

# 20. Binary Erosion
eroded_array = binary_erosion()

# Additional examples:
# 21. Sparse Matrix Creation (CSR format)
sparse_matrix_csr = csr_matrix()

# 22. Diagonal Matrix Construction
diagonal_matrix = diags()

# 23. Load MATLAB File
loaded_data = loadmat()

# 24. Save MATLAB File
savemat()
