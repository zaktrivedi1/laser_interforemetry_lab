import numpy as np

FILE_NAME = "two_laser_coherance_length.txt"
WRITE_TO_FILE = True
# Load the data from file, skipping the header line.
data = np.genfromtxt(FILE_NAME, delimiter=',', skip_header=21, max_rows=5)
print(data)
# Columns (assuming):
# Column 0: identifier,
# Column 1: ls_wavelengths,
# Column 2: ls_errors,
# Column 3: fft_wavelengths,
# Column 4: fft_errors.
lc = data[:, 1]
lc_error = data[:, 2]
ls_linewidth = data[:, 3]
ls_error = data[:, 4]
fft_linewidth = data[:, 5]
fft_error = data[:, 6]

# Compute weights as 1 / (error^2)
ls_weights = 1 / (ls_error**2)
fft_weights = 1 / (fft_error**2)
lc_weights = 1/(lc_error**2)

# Calculate weighted means
weighted_mean_ls = np.sum(ls_weights * ls_linewidth) / np.sum(ls_weights)
weighted_mean_fft = np.sum(fft_weights * fft_linewidth) / np.sum(fft_weights)
weighted_mean_lc = np.sum(lc_weights * lc) / np.sum(lc_weights)

# Calculate the error on the weighted mean
weighted_mean_error_ls = np.sqrt(1 / np.sum(ls_weights))
weighted_mean_error_fft = np.sqrt(1 / np.sum(fft_weights))
weighted_mean_error_lc = np.sqrt(1 / np.sum(lc_weights))

print("LS Weighted Mean Linewidth: {:.12f} ± {:.12f}".format(
    weighted_mean_ls, weighted_mean_error_ls))
print("FFT Weighted Mean Linewidth: {:.12f} ± {:.12f}".format(
    weighted_mean_fft, weighted_mean_error_fft))
print("Weighted Mean Coherance Length: {:.12f} ± {:.12f}".format(
    weighted_mean_lc, weighted_mean_error_lc))

if WRITE_TO_FILE:
    with open("mean_twolaser_lc_linewidth.txt", "a") as period_file:
        period_file.write(
            f"Lamp, {weighted_mean_ls:.12f} m, {weighted_mean_error_ls:.12f} m, {weighted_mean_fft:.12f} m, {weighted_mean_error_fft:.12f} m, {weighted_mean_lc:.12f} m, {weighted_mean_error_lc:.12f} m\n")
    print("Mean wavelengths saved")
