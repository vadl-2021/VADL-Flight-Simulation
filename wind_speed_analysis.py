# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:42:35 2021

@author: Ryan Burinescu

Rocket simulation for various wind speeds
"""

import os
from parameters import *
from numpy import sin, cos, tan, arctan2, pi, sqrt, interp, real, mean
from numpy.linalg import norm
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl import load_workbook
from shutil import copy
from datetime import date

# SETUP
# ============================================================================

plot_folder = 'Example' # name of subfolder to save plots in

save_plots = True # save plots as .png files

# Specify Wind Speeds
# ----------------------------------------------------------------------------

wind_speeds = [0, 5, 10, 15, 20] # wind speed range [mph]

# ============================================================================

# Specify Plot Parameters
# ----------------------------------------------------------------------------

lw = 2 # linewidth
c = 'black' # line color

# Altitude & Drift Arrays
# ----------------------------------------------------------------------------

x_array = []
z_array = []

# Run Rocket Flight Simulation
# ----------------------------------------------------------------------------

for k in range(len(wind_speeds)):
    
    print('-----------------------\n')
    print(f'{wind_speeds[k]} mph Winds\n')
    print('-----------------------')
    
    stream = open('parameters.py')
    read_file = stream.read()
    exec(read_file)
    wr = wind_speeds[k] # set current wind speed
    stream = open('calculate_trajectory.py')
    read_file = stream.read()
    exec(read_file)
    
    x_array.append(x) # store current drift array
    z_array.append(z) # store current altitude array

# Generate Plots
# ----------------------------------------------------------------------------

# create subfolder (if it does not exist)
path = 'Flights/' + plot_folder + '_' + str(date.today()) + '/'
os.makedirs(path, exist_ok=True)

# drift, altitude
fig, ax = plt.subplots()
for k in range(len(wind_speeds)):
    ax.plot(x_array[k], z_array[k], lw=lw, label=f'{wind_speeds[k]} mph')
plt.xlabel('Horizontal Drift [ft]')
plt.ylabel('Altitude [ft]')
plt.title('Wind Speed Analysis')
plt.grid()
plt.legend()

if save_plots:
    fig.savefig(path + 'wind_speed_analysis.png')
