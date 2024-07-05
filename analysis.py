import numpy as np
from read_cnf import read_cnf_file

from scipy.optimize import curve_fit
import scipy

import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


data_dic = read_cnf_file('./bkg_Gary.CNF')
channel = data_dic['Channels']
counts = data_dic['Channels data']

def calibration_curve(channels):
    return 0.3865 * channels + 0.5

def detector_efficiency(energy):
    return np.exp(8.5 * np.exp(-0.00061 * energy) - 13.6)

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
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the fitted parameters
    print(f"Amplitude (a): {a}")
    print(f"Mean (x0): {x0}")
    print(f"Standard deviation (sigma): {sigma}")
    return a, x0, sigma

def intergrate_gaussian(lower_limit, upper_limit, mean, std, amplitude):

    def gaussian(x, mean, std, amplitude):
        return amplitude * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    
    result, error = scipy.integrate.quad(gaussian, lower_limit, upper_limit, args=(mean, std, amplitude))

    print("Integrated spectrum: {} +/- {}".format(-result, error))

    return -result, error

def compute_rate(event, efficiency, time):
    print("Event Rate: {}Bq".format(event / (efficiency * time))) 

energy = calibration_curve(channel)
K_40_energy, K_40_counts = [], []

for i in range(len(energy)):
    if energy[i] > 1456 and energy[i] < 1464:
        K_40_energy.append(energy[i])
        K_40_counts.append(counts[i])
        
amplitude, mean, std = fit_gauss(K_40_energy, K_40_counts)

upper_limit = 1456
lower_limit = 1464
result, error = intergrate_gaussian(lower_limit, upper_limit, mean, std, amplitude)

compute_rate(result, detector_efficiency(1460), 86400)


plt.plot(energy, counts)
plt.xlabel("Energy (keV)",fontsize=12)
plt.ylabel("Counts (/day)",fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale("log")
plt.show()
