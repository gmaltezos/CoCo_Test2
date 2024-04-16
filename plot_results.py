from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

def plot_results(glucose, insulin_bolus, i, m, time):

    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.set_figheight(20)
    fig.set_figwidth(50)

    # Subplot 1: Glucose
    ax[0].plot(time, glucose, marker='o', color='red', linewidth=2, label='Glucose [mg/dl]')

    ax[0].fill_between(np.array([time[0], time[-1]]), np.array([70, 70]), np.array([180, 180]), color='green', alpha=0.2,
                       label='Target range')

    ax[0].grid()
    ax[0].legend()

    # Subplot 2: Meals
    markerline, stemlines, baseline = ax[1].stem(time, m*5, basefmt='k:', label='CHO [g]')
    plt.setp(stemlines, 'color', (70.0 / 255, 130.0 / 255, 180.0 / 255))
    plt.setp(markerline, 'color', (70.0 / 255, 130.0 / 255, 180.0 / 255))
    ax[1].grid()
    ax[1].legend()

    # Subplot 3: Insulin
    markerline, stemlines, baseline = ax[2].stem(time, insulin_bolus, basefmt='k:', label='Bolus insulin [U]')
    plt.setp(stemlines, 'color', (50.0 / 255, 205.0 / 255, 50.0 / 255))
    plt.setp(markerline, 'color', (50.0 / 255, 205.0 / 255, 50.0 / 255))

    ax[2].plot(time, i * 60, color='black', linewidth=2, label='Basal insulin [U/h]')

    ax[2].grid()
    ax[2].legend()

    plt.show()