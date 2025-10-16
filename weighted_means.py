import numpy as np

FILE_NAME = "laser+lamp_wavelengths.txt"
WRITE_TO_FILE = True
# Load the data from file, skipping the header line.
data = np.genfromtxt(FILE_NAME, delimiter=',', skip_header=1)

# Columns (assuming):
# Column 0: identifier,
# Column 1: ls_wavelengths,
# Column 2: ls_errors,
# Column 3: fft_wavelengths,
# Column 4: fft_errors.
ls_wavelengths = data[:, 1]
fft_wavelengths = data[:, 3]
ls_errors = data[:, 2]
fft_errors = data[:, 4]

# Compute weights as 1 / (error^2)
ls_weights = 1 / (ls_errors**2)
fft_weights = 1 / (fft_errors**2)

# Calculate weighted means
weighted_mean_ls = np.sum(ls_weights * ls_wavelengths) / np.sum(ls_weights)
weighted_mean_fft = np.sum(fft_weights * fft_wavelengths) / np.sum(fft_weights)

# Calculate the error on the weighted mean
weighted_mean_error_ls = np.sqrt(1 / np.sum(ls_weights))
weighted_mean_error_fft = np.sqrt(1 / np.sum(fft_weights))

print("LS Weighted Mean Wavelength: {:.12f} ± {:.12f}".format(
    weighted_mean_ls, weighted_mean_error_ls))
print("FFT Weighted Mean Wavelength: {:.12f} ± {:.12f}".format(
    weighted_mean_fft, weighted_mean_error_fft))

if WRITE_TO_FILE:
    with open("mean_wavelengths.txt", "a") as period_file:
        period_file.write(
            f"{FILE_NAME}, {weighted_mean_ls:.12f} m, {weighted_mean_error_ls:.12f} m, {weighted_mean_fft:.12f} m, {weighted_mean_error_fft:.12f} m\n")
    print("Mean wavelengths saved")
