import numpy as np
from read_cnf import read_cnf_file
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

"""

They did some weird stuff...................

"""


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

    def quadratic(x, a, b, c):
        return a + b*x + c*x**2

    # Fit a quadratic model to the data and get the covariance matrix
    popt, pcov = curve_fit(quadratic, channels, energy, p0=[0.1,0.001,0.])
    a, b, c = popt

    # Fit a linear model to the data and get the covariance matrix
    #popt, pcov = np.polyfit(channels, energy, 1, cov=True)
    a, b, c = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))

    # Create data for the fitted linear model
    x_fit = np.linspace(min(channels), max(channels), 1000)
    y_fit = a + b*x_fit + c * x_fit**2

    # Plot the original data and the fitted linear model
    plt.figure(figsize=(10, 6))
    plt.plot(channels, energy, 'o', label='Data')
    plt.plot(x_fit, y_fit, label=f"E(C) = ({round(a,1)} ± {round(a_err,1)}) + ({round(b,4)} ± {round(b_err,4)})C + ({round(c,8)} ± {round(c_err,8)})C^2")
    plt.title('Linear Fit to Data')
    plt.xlabel('Channels')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print the fitted parameters and their uncertainties
    print(f"E(C) = ({a} ± {a_err}) + ({b} ± {b_err})C + ({c} ± {c_err})C^2")
"""
data = np.genfromtxt("./calibration_curve.dat")
energy = data[:,0]
channels = data[:,1]
channel_uncert = data[:,2]

fit_line(channels, energy)
"""
# ============================================================================================
# ============================= Detector efficiency calculations =============================
# ============================================================================================

def fit_exp(energy, ratio):

    def efficiency_ft(x,v1,v2,v3,v4,v5,v6):
        return 10 ** (v1 + v2 * x + v3 / x + v4 / x**2 + v5 / x**3 + v6 / x**4)

    # Fit a linear model to the data and get the covariance matrix
    popt, pcov = curve_fit(efficiency_ft, energy, ratio, p0=[0.01,0.01,0.001,0.0001,0.0001,0.0001], maxfev=20000)

    v1,v2,v3,v4,v5,v6 = popt

    # Create data for the fitted linear model
    x_fit = np.linspace(min(energy), max(energy), 1000)
    y_fit = 10 ** (v1 + v2 * x_fit + v3 / x_fit + v4 / x_fit**2 + v5 / x_fit**3 + v6 / x_fit**4)

    #a_err, b_err, c_err = np.sqrt(np.diag(pcov))
    #print(a,b,c)
    #print(a_err, b_err, c_err)

    # Plot the original data and the fitted exponential
    plt.figure(figsize=(10, 6))
    plt.plot(energy, ratio, 'o', label='Data')
    plt.plot(x_fit, y_fit, label='Fitted Exponential')
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