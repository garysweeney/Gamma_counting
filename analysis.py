import numpy as np
from read_cnf import read_cnf_file
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_dic = read_cnf_file('./bkg_Gary.CNF')
channel = data_dic['Channels']
counts = data_dic['Channels data']

def calibration_curve(channels):
    return 0.3865 * channels + 0.5

def fit_gauss(channels, counts):

    # Define the Gaussian function
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    # Initial guess for the parameters
    initial_guess = [max(counts), channels[np.argmax(counts)], np.std(channels)]

    # Fit the Gaussian function to the data
    popt, pcov = curve_fit(gaussian, channels, counts, p0=initial_guess)

    # Extract the fitted parameters
    a, x0, sigma = popt

    # Create data for the fitted Gaussian curve
    x_fit = np.linspace(min(channels), max(channels), 1000)
    y_fit = gaussian(x_fit, *popt)

    # Plot the original data and the fitted Gaussian
    plt.figure(figsize=(10, 6))
    plt.plot(channels, counts, 'o', label='Data')
    plt.plot(x_fit, y_fit, label='Fitted Gaussian')
    plt.title('Gaussian Fit to Data')
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the fitted parameters
    print(f"Amplitude (a): {a}")
    print(f"Mean (x0): {x0}")
    print(f"Standard deviation (sigma): {sigma}")

energy = calibration_curve(channel)

K_40_energy, K_40_counts = [], []

for i in range(len(energy)):
    if energy[i] > 1456 and energy[i] < 1464:
        K_40_energy.append(energy[i])
        K_40_counts.append(counts[i])
        
fit_gauss(K_40_energy, K_40_counts)

plt.plot(energy, counts)
plt.yscale("log")
plt.show()
