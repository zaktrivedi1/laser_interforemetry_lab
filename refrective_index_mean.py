import numpy as np

FILE_NAME = "refractive_index.csv"
WRITE_TO_FILE = False
# Load the data from file, skipping the header line.
data = np.genfromtxt(FILE_NAME, delimiter=',', skip_header=1)

# Columns (assuming):
# Column 0: identifier,
# Column 1: ls_wavelengths,
# Column 2: ls_errors,
# Column 3: fft_wavelengths,
# Column 4: fft_errors.
n = data[:, 0]
n_error = data[:, 1]

# Compute weights as 1 / (error^2)
n_weights = 1 / (n_error**2)

# Calculate weighted means
weighted_mean = np.sum(n_weights * n) / np.sum(n_weights)

# Calculate the error on the weighted mean
weighted_mean_error = np.sqrt(1 / np.sum(n_weights))


print("n Weighted Mean Wavelength: {0} ± {1}".format(
    weighted_mean, weighted_mean_error))


# if WRITE_TO_FILE:
#   with open("mean_wavelengths.txt", "a") as period_file:
#      period_file.write(
#         f"{FILE_NAME}, {weighted_mean_ls:.12f} m, {weighted_mean_error_ls:.12f} m, {weighted_mean_fft:.12f} m, {weighted_mean_error_fft:.12f} m\n")
#print("Mean wavelengths saved")
