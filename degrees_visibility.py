import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from the file
data = np.genfromtxt("degrees_visibility.txt", delimiter=',')
degrees = data[:, 0]
visibilities = data[:, 1]
visibility_uncertainties = data[:, 2]

# Define the cos^2 model function


def cos2_fit(x, A, phase, offset):
    return A * np.cos(np.deg2rad(x) + phase)**2 + offset


# Initial guess for [A, phase, offset]
initial_guess = [1, 0, 0.2]

# Perform the curve fitting with error propagation
popt, pcov = curve_fit(cos2_fit, degrees, visibilities, sigma=visibility_uncertainties,
                       p0=initial_guess, absolute_sigma=True)

# Extract best-fit parameters and uncertainties
A_fit, phase_fit, offset_fit = popt
A_err, phase_err, offset_err = np.sqrt(np.diag(pcov))

# Print results
print("Fitted parameters with uncertainties:")
print(f"A = {A_fit:.4f} ± {A_err:.4f}")
print(f"phase = {phase_fit:.4f} ± {phase_err:.4f} rad")
print(f"offset = {offset_fit:.4f} ± {offset_err:.4f}")

# Generate fitted curve
x_fit = np.linspace(np.min(degrees), np.max(degrees), 500)
y_fit = cos2_fit(x_fit, *popt)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(degrees, visibilities, yerr=visibility_uncertainties,
             fmt='o', ecolor='black', capsize=5, label='Data')
plt.plot(x_fit, y_fit, 'r-',
         label=r'$cos^2(\theta)$ Fit')
plt.xlabel(r'Relative Angle ($\theta$) / $^{\circ}$', fontsize=20)
plt.ylabel(r'Visibility ($V$)', fontsize=20)
plt.grid(True)
plt.legend(fontsize=18)
# plt.tight_layout()
plt.show()
