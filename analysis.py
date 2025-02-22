import numpy as np
from read_cnf import read_cnf_file

from scipy.optimize import curve_fit
import scipy

import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

bkg_data_dic = read_cnf_file('./bkg_Gary.CNF')
bkg_channel = np.array(bkg_data_dic['Channels']).astype(float)
bkg_counts = np.array(bkg_data_dic['Channels data']).astype(float)
bkg_counts /= (86400 / 3600 / 24)

PTFE_data_dic = read_cnf_file('./PTFE_Gary.CNF')
PTFE_channel = np.array(PTFE_data_dic['Channels']).astype(float)
PTFE_counts = np.array(PTFE_data_dic['Channels data']).astype(float)
PTFE_counts /= (341914.16 / 3600 / 24)

def calibration_curve(channels):
    return 0.3865 * channels + 0.5

def detector_efficiency(energy, counts):

    efficiency_energy = np.zeros(len(energy))
    efficiency_counts = np.zeros(len(energy))

    for i in range(len(energy)):
        efficiency_energy[i] = np.exp(8.5 * np.exp(-0.00061 * energy[i]) - 13.6)

    for i in range(len(energy)):
        efficiency_counts[i] = float(counts[i]) / efficiency_energy[i]

    return efficiency_counts

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

PTFE_energy = calibration_curve(PTFE_channel)
bkg_energy = calibration_curve(bkg_channel)

PTFE_count_eff = detector_efficiency(PTFE_energy, PTFE_counts)
bkg_count_eff = detector_efficiency(bkg_energy, bkg_counts)

plt.plot(bkg_energy, bkg_counts)
plt.plot(PTFE_energy, PTFE_counts)
plt.xlabel("Energy (keV)",fontsize=12)
plt.ylabel("Rate (counts/day)",fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale("log")
plt.show()
