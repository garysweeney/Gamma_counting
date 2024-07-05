import numpy as np
from read_cnf import read_cnf_file
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================================
# ============================== Gaussian fit for channel peaks ==============================
# ============================================================================================
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

data_dic = read_cnf_file('./Ba_133_Gary.CNF')
channel = data_dic['Channels']
counts = data_dic['Channels data']

#plt.plot(channel, counts)
#plt.xlabel("Channel")
#plt.ylabel("Counts")
#plt.show()

# Fit peak
peak_channel = channel[np.where(channel >= 910)]
peak_counts = counts[np.where(channel >= 910)]
peak_channel = peak_channel[:20]
peak_counts = peak_counts[:20]

#fit_gauss(peak_channel, peak_counts)

# ============================================================================================
# ============================= Linear fit for calibration curve =============================
# ============================================================================================
def fit_line(channels, energy):

    # Fit a linear model to the data and get the covariance matrix
    popt, pcov = np.polyfit(channels, energy, 1, cov=True)
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))

    # Create data for the fitted linear model
    x_fit = np.linspace(min(channels), max(channels), 1000)
    y_linear_fit = slope * x_fit + intercept

    # Plot the original data and the fitted linear model
    plt.figure(figsize=(10, 6))
    plt.plot(channels, energy, 'o', label='Data')
    plt.plot(x_fit, y_linear_fit, label='Fitted Linear Model')
    plt.title('Linear Fit to Data')
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the fitted parameters and their uncertainties
    print(f"Slope: {slope} ± {slope_err}")
    print(f"Intercept: {intercept} ± {intercept_err}")

data = np.genfromtxt("./calibration_curve.dat")
energy = data[:,0]
channels = data[:,1]
channel_uncert = data[:,2]

#fit_line(channels, energy)

# ============================================================================================
# ============================= Detector efficiency calculations =============================
# ============================================================================================

def fit_exp(energy, ratio):

    log_ratio = [np.log(i) for i in ratio]

    def exp(x,a,b,c):
        return a * np.exp(-b * x) + c

    # Fit a linear model to the data and get the covariance matrix
    popt, pcov = curve_fit(exp, energy, log_ratio, p0=[0.1,0.001,0.])

    a, b, c = popt

    # Create data for the fitted linear model
    x_fit = np.linspace(min(energy)-81, max(energy), 1000)
    y_fit = a * np.exp(-b * x_fit) + c

    a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    print(a,b,c)
    print(a_err, b_err, c_err)

    # Plot the original data and the fitted exponential
    plt.figure(figsize=(10, 6))
    plt.plot(energy, np.exp(log_ratio), 'o', label='Data')
    plt.plot(x_fit, np.exp(y_fit), label='Fitted Exponential')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Efficiency')
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.show()

data = np.genfromtxt("./detector_efficiency.dat")
energy = data[:,1]
observed_counts = data[:,2]
current_activity = data[:,3]
intensity = data[:,4]
estimated_counts = np.zeros(len(energy))

for i in range(len(energy)):
    estimated_counts[i] = current_activity[i] * 1000 * intensity[i]

ratio = observed_counts / estimated_counts

fit_exp(energy, ratio)