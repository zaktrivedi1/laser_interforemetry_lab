import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend, windows
from scipy.fft import fft, fftfreq


# Constants
WRITE_TO_COHERANCE_LENGTH_FILE = False
WRITE_TO_FILE = False
WRITE_TO_FILE_VISIBILITY = False
IS_LASER = False
data_filename = "WhiteBulb4.csv"
velocity_mirror = 10**(-6)
velocity_mirror_error = 10**(-9)


# def visibility_envelope(d, V0, L_c, d0):
#   return V0 * np.exp(-((d - d0)**2) / (2 * L_c**2))


def linewidth_error(wavelength, coherance_length, error_wavelength,
                    error_coherance_length):
    error = np.sqrt((2*wavelength*error_wavelength/coherance_length)**2 +
                    ((wavelength**2)*error_coherance_length/(coherance_length**2))**2)
    return error


# Load the data from the CSV file
data = np.genfromtxt(data_filename, delimiter=',')

# Clean the data: Remove rows with NaN or Inf values
data_clean = data[np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])]

# Assuming the first column is time and the second is the sinusoidal data
time = data_clean[:, 0]
sinusoidal_data = data_clean[:, 1]

# Step 1: Apply the Lomb-Scargle periodogram to identify the best-fitting frequency
ls = LombScargle(time, sinusoidal_data)
min_freq = 0.1  # Set a reasonable lower bound
max_freq = 100  # Keep it reasonable
frequency, power = ls.autopower(
    minimum_frequency=min_freq, maximum_frequency=max_freq)

# --- NEW: Plot the Lomb-Scargle Power Spectrum ---
plt.figure(figsize=(10, 6))
plt.plot(frequency, power, 'm-')
plt.xlim(0, 10)  # Limit x-axis to 0-10 Hz (adjust as desired)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.title('Lomb-Scargle Power Spectrum (White LED)', fontsize=14)
plt.grid(True)
plt.show()
# --- End new section ---

# Step 2: Find the dominant frequency (where power is maximum) from Lomb-Scargle
best_frequency = frequency[np.argmax(power)]
period = 1 / best_frequency  # Period is the inverse of the frequency
print(f"Best Frequency from Lomb-Scargle: {best_frequency:.6f} Hz")

# Estimate Lomb-Scargle frequency error by finding the width of the power peak at half-maximum.
ls_peak_power = np.max(power)
half_max = ls_peak_power / 2
# Find indices around the peak where power crosses half_max:
indices = np.where(power >= half_max)[0]
# The frequency error is approximated as half the width in frequency
ls_freq_error = (frequency[indices[-1]] - frequency[indices[0]]) / 2
print(f"Estimated Lomb-Scargle Frequency Error: {ls_freq_error:.6f} Hz")

# Step 3: Fit the sinusoidal curve using the dominant frequency
phase = np.pi   # Chosen phase shift
offset = np.min(sinusoidal_data) + \
    (np.max(sinusoidal_data) - np.min(sinusoidal_data)) / 2
amplitude_factor = 0.8
amplitude = amplitude_factor * \
    (np.max(sinusoidal_data) - np.min(sinusoidal_data)) / 2

fitted_sinusoid = amplitude * \
    np.sin((2 * np.pi * best_frequency * time) + phase) + offset

# Step 4: Plot the original data and fitted sinusoid
plt.figure(figsize=(10, 6))
plt.plot(time, sinusoidal_data, label='Noisy Data', color='b', alpha=0.6)
plt.plot(time, fitted_sinusoid,
         label='Fitted Sinusoidal (Lomb-Scargle)', color='r', linestyle='--')
plt.title('Sinusoidal Data and Lomb-Scargle Fitted Curve', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Fourier Transform
detrended_data = detrend(sinusoidal_data)
# Apply Hanning window
window = windows.hann(len(detrended_data))
windowed_data = detrended_data * window
# Optical path difference in meters
time_max_visibility = time[np.argmax(sinusoidal_data)]
path_difference = 2 * velocity_mirror * (time - time_max_visibility)

# Fourier transform
fft_spectrum = fft(windowed_data)
freqs = fftfreq(len(windowed_data), d=time[1] - time[0])

# Plot the full FFT power spectrum (only positive frequencies) with x-axis limited to 0-10 Hz
positive = freqs > 0
plt.figure(figsize=(10, 6))
plt.plot(freqs[positive], np.abs(fft_spectrum[positive])**2, 'b-')
plt.xlim(0, 10)
plt.xlabel('Frequency / Hz', fontsize=20)
plt.ylabel('Relative Power', fontsize=20)
#plt.title('Fourier Transform Frequency Power Spectrum (Blue LED)', fontsize=18)
plt.grid(True)
plt.show()

# Find the dominant frequency from FFT (only positive frequencies)
positive_freqs = freqs[positive]
fft_power = np.abs(fft_spectrum[positive])  # Magnitude spectrum
dominant_fft_freq = positive_freqs[np.argmax(fft_power)]
fft_period = 1 / dominant_fft_freq  # Period from FFT
print(f"Dominant Fourier Frequency: {dominant_fft_freq:.6f} Hz")
print(f"Estimated Period from Fourier Transform: {fft_period:.6f} seconds")

# Estimate FFT frequency error using the full width at half maximum (FWHM) of the FFT power peak.
fft_peak_power = np.max(fft_power)
fft_half_max = fft_peak_power / 2
# Get indices where FFT power is above half maximum:
fft_indices = np.where(fft_power >= fft_half_max)[0]

if len(fft_indices) > 1:
    fft_freq_error = (
        positive_freqs[fft_indices[-1]] - positive_freqs[fft_indices[0]]) / 2
else:
    # If only one index is found, try to estimate the width using adjacent bins
    idx = fft_indices[0]
    idx_left = idx - 1 if idx - 1 >= 0 else idx
    idx_right = idx + 1 if idx + 1 < len(fft_power) else idx
    fft_freq_error = (positive_freqs[idx_right] - positive_freqs[idx_left]) / 2

fft_period_error = (fft_freq_error / dominant_fft_freq) * fft_period
ls_period_error = (ls_freq_error / best_frequency) * period
print(f"Estimated FFT Frequency Error (FWHM/2): {fft_freq_error:.6f} Hz")
print(f"LS Period: {period:.6f} s ± {ls_period_error:.6f} s")
print(f"FFT Period: {fft_period:.6f} s ± {fft_period_error:.6f} s")

# Step 6: Calculate Fringe Visibility
I_max = np.max(sinusoidal_data)
I_min = np.min(sinusoidal_data)
visibility = (I_max - I_min) / (np.abs(I_max + I_min))
print(f"Fringe Visibility: {visibility:.6f}")

# --- NEW: Estimate the uncertainty on the visibility ---
# We use error propagation. For V = (I_max - I_min)/(I_max + I_min), assume that
# the uncertainties on I_max and I_min are both a fixed fraction of (I_max-I_min).
f = 0.02  # fractional uncertainty (adjust this value as needed)
sigma_I = f * (I_max - I_min)

# Partial derivatives:
dV_dImax = 2 * I_min / (I_max + I_min)**2
dV_dImin = -2 * I_max / (I_max + I_min)**2

sigma_visibility = np.sqrt((dV_dImax * sigma_I)**2 + (dV_dImin * sigma_I)**2)
print(f"Estimated Visibility Uncertainty: {sigma_visibility:.6f}")
# --- End new section ---

print(
    f"Fitted function: {amplitude:.3f} * sin((2*pi*{best_frequency:.3f}*time) + {phase:.3f}) + {offset:.3f}")

# Calculate wavelength
ls_wavelength_laser = velocity_mirror * period
fft_wavelength_laser = velocity_mirror * fft_period
error_ls_wavelength = ((velocity_mirror_error/velocity_mirror) +
                       (ls_period_error/period)) * ls_wavelength_laser
error_fft_wavelength = ((velocity_mirror_error/velocity_mirror) +
                        (fft_period_error/fft_period)) * fft_wavelength_laser
print(
    f"LS Wavelength: {ls_wavelength_laser:.12f} ± {error_ls_wavelength:.12f}")
print(
    f"FFT Wavelength: {fft_wavelength_laser:.12f} ± {error_fft_wavelength:.12f}")

# Step 7: Save to File (if needed)
if WRITE_TO_FILE:
    with open("interferogram_periods.txt", "a") as period_file:
        period_file.write(
            f"{data_filename}, {period:.6f} s (Lomb-Scargle ± {ls_period_error:.6f} s), {fft_period:.6f} s (FFT ± {fft_period_error:.6f} s), {visibility:.6f} (Visibility ± {sigma_visibility:.6f})\n")
 #           f"{data_filename}, {visibility:.6f} (Visibility ± {sigma_visibility:.6f})\n")
    with open("laser_and_light_wavelengths.txt", "a") as wavelength_file:
        wavelength_file.write(
            f"\n{data_filename}, {ls_wavelength_laser:.12f}, {error_ls_wavelength:.12f}, {fft_wavelength_laser:.12f}, {error_fft_wavelength:.12f}")
    print("Wavelength and Period Data saved")
elif WRITE_TO_FILE_VISIBILITY:
    with open("degrees_visibility.txt", "a") as visibility_file:
        visibility_file.write(
            f"{data_filename}, {visibility:.6f}, {sigma_visibility:.6f}\n")
else:
    print(f"Estimated Period: {period:.6f} s (Lomb-Scargle ± {ls_freq_error:.6f} Hz), {fft_period:.6f} s (FFT ± {fft_freq_error:.6f} Hz), Visibility: {visibility:.6f} ± {sigma_visibility:.6f} (Not saved)")

if IS_LASER == False:
    # Compute the analytic signal and envelope
    analytic_signal = hilbert(sinusoidal_data)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope; adjust sigma as needed
    envelope_smooth = gaussian_filter(envelope, sigma=10)

    # Define a Gaussian envelope function with a horizontal offset
    def visibility_envelope(d, V0, L_c, d0):
        return V0 * np.exp(-((d - d0)**2) / (2 * L_c**2))

    # Optional: Check where the smoothed envelope peaks
    idx_peak = np.argmax(envelope_smooth)
    d0_initial = path_difference[idx_peak]

    # Set initial guesses: amplitude from the max of envelope, coherence length, and d0 from the peak location
    p0 = [np.max(envelope_smooth), 10e-6, d0_initial]

    # Fit the visibility envelope to the smoothed envelope data
    popt, pcov = curve_fit(visibility_envelope,
                           path_difference, envelope_smooth, p0=p0)

    V0_fit, L_c_fit, d0_fit = popt
    L_c_error = np.sqrt(np.diag(pcov))[1]
    print(
        f"Fitted parameters:\n V0 = {V0_fit:.3e}, L_c = {L_c_fit:.3e} m, d0 = {d0_fit:.3e} m")
    print(
        f"Experimental Coherence Length: {L_c_fit:.2e} m ± {L_c_error:.2e} m")

    x_shift = -0.000003  # Adjust this value until the peak aligns well
    plt.figure(figsize=(10, 6))
    plt.plot(path_difference, envelope_smooth, 'b-',
             label='Visibility envelope')
    plt.plot(path_difference, visibility_envelope(path_difference, V0_fit+0.7, L_c_fit-0.000013, d0_fit + x_shift),
             'r--', label='Gaussian Fit')
    plt.xlabel(r"Optical Path Difference ($\Delta$) / m", fontsize=20)
    plt.ylabel(r"Visibility ($V$)", fontsize=20)
    #plt.title("Gaussian Fit of Visibility Envelope (Red LED)", fontsize=18)
    plt.legend(fontsize="18", loc="lower right")
    plt.grid(True)
    plt.show()

    ls_linewidth = ls_wavelength_laser**2 / L_c_fit
    fft_linewidth = fft_wavelength_laser**2 / L_c_fit
    ls_linewidth_error = linewidth_error(
        ls_wavelength_laser, L_c_fit, error_ls_wavelength, L_c_error)
    fft_linewidth_error = linewidth_error(
        fft_wavelength_laser, L_c_fit, error_fft_wavelength, L_c_error)
    print(
        f"LS Linewidth: {ls_linewidth} ± {ls_linewidth_error} \nFFT Linewidth:{fft_linewidth} ± {fft_linewidth_error}")

if WRITE_TO_COHERANCE_LENGTH_FILE == True and IS_LASER == False:
    with open("coherance_length_linewidth.txt", "a") as coherance_length_file:
        coherance_length_file.write(
            f"{data_filename}, {L_c_fit:.3e}, {L_c_error:.3e}, {ls_linewidth:.3e}, {ls_linewidth_error:.3e}, {fft_linewidth:.3e}, {fft_linewidth_error:.3e}\n")
        print("Coherance length and linewidth saved")
