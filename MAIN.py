# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:42:35 2021

@author: Ryan Burinescu

Run rocket flight simulation and generate plots.
"""

import os
from parameters import *
from numpy import sin, cos, tan, arctan2, pi, sqrt, interp, real, mean
from numpy.linalg import norm
import matplotlib.pyplot as plt
from datetime import date

# Specify Which Plots to Generate
# ============================================================================

plot_folder = 'Example' # name of subfolder to save plots in

save_plots = True # save plots as .png files

# x-axis, y-axis
plot_x_z = True # horizontal drift, altitude
plot_t_z = True # time, altitude
plot_t_vz = True # time, vertical velocity
plot_t_az = True # time, vertical 
plot_t_theta = True # time, pitch angle
plot_landing_energies = True # landing energies

plot_z_w = False # altitude, wind speed
plot_t_SSM = False # time, static stability margin
plot_t_m = False # time, mass

# ============================================================================

# Specify Plot Parameters
# ----------------------------------------------------------------------------

lw = 2 # linewidth
c = 'black' # line color

# rocket section names (for landing energy plots)
sections = ['Fore Section', 'Middle Section', 'Aft Section']

# Specify Relevant Limits & Requirements
# ----------------------------------------------------------------------------

max_landing_energy = 75 # landing energy limit [ft-lbf]

# Run Rocket Flight Simulation
# ----------------------------------------------------------------------------

stream = open('calculate_trajectory.py')
read_file = stream.read()
exec(read_file)

# Generate Plots
# ----------------------------------------------------------------------------

# create subfolder (if it does not exist)
path = 'Plots/' + plot_folder + '_' + str(date.today()) + '/'
os.makedirs(path, exist_ok=True)

# horizontal drift, altitude
if plot_x_z:
    fig = plt.figure()
    plt.plot(x,z,linewidth=lw,color=c)
    plt.title('Altitude vs. Horizontal Drift')
    plt.xlabel('Horizontal Drift [ft]')
    plt.ylabel('Altitude [ft]')
    plt.grid()
    plt.xlim(0, drift+1500)
    plt.ylim(0, apogee+500)
    if save_plots:
        fig.savefig(path + 'x_z.png')
        
# time, altitude
if plot_t_z:
    fig = plt.figure()
    plt.plot(t,z,linewidth=lw,color=c)
    plt.title('Altitude vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Altitude [ft]')
    plt.grid()
    plt.xlim(0, t[i_land])
    plt.ylim(0, apogee+500)
    if save_plots:
        fig.savefig(path + 't_z.png')
        
# time, vertical velocity
if plot_t_vz:
    fig = plt.figure()
    plt.plot(t,vz,linewidth=lw,color=c)
    plt.title('Vertical Velocity vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Vertical Velocity [fps]')
    plt.grid()
    plt.xlim(0, t[i_land])
    if save_plots:
        fig.savefig(path + 't_vz.png')
        
# time, vertical acceleration
if plot_t_az:
    fig = plt.figure()
    plt.plot(t,az,linewidth=lw,color=c)
    plt.title('Vertical Acceleration vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Vertical Accelertation [G]')
    plt.grid()
    plt.xlim(0, t[i_land])
    if save_plots:
        fig.savefig(path + 't_az.png')
        
# time, pitch angle (during ascent)
if plot_t_theta:
    fig = plt.figure()
    plt.plot(t[0:i_apogee],theta[0:i_apogee],linewidth=lw,color=c)
    plt.title('Pitch Angle vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch Angle [deg]')
    plt.grid()
    plt.xlim(0, t[i_apogee])
    if save_plots:
        fig.savefig(path + 't_theta.png')
        
# landing energies
if plot_landing_energies:
    fig = plt.figure()
    plt.bar(5+np.array([0,5,10]),[KE_fore, KE_mid, KE_aft],tick_label=sections,color=c)
    plt.hlines(max_landing_energy,0,20,color='r',linestyles='dashed',label='Landing Energy Limit')
    plt.annotate(str(KE_fore),(5.5,KE_fore-2))
    plt.annotate(str(KE_mid),(10.5,KE_mid-2))
    plt.annotate(str(KE_aft),(15.5,KE_aft-2))
    plt.title('Section Landing Energies')
    plt.ylabel('Landing Energy [ft-lbf]')
    plt.legend()
    plt.xlim(0,20)
    plt.grid(axis='y')
    if save_plots:
        fig.savefig(path + 'landing_energies.png')

# altitude, wind speed (during ascent)
if plot_z_w:
    fig = plt.figure()
    plt.plot(z[0:i_apogee],w[0:i_apogee],linewidth=lw,color=c)
    plt.title('Wind Speed vs. Altitude')
    plt.xlabel('Altitude [ft]')
    plt.ylabel('Wind Speed [mph]')
    plt.ylim(0, max(w))
    plt.grid()
    if save_plots:
        fig.savefig(path + 'z_w.png')
        
# time, static stability margin (during ascent)
if plot_t_SSM:
    fig = plt.figure()
    plt.plot(t[0:i_apogee],SSM[0:i_apogee],linewidth=lw,color=c)
    plt.title('Static Stability Margin vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Static Stability Margin [cal]')
    plt.xlim(0, t[i_apogee])
    plt.ylim(SSM[0], max(SSM)+0.1)
    plt.grid()
    if save_plots:
        fig.savefig(path + 't_SSM.png')
        
# time, mass
if plot_t_m:
    fig = plt.figure()
    plt.plot(t[0:i_land],m,linewidth=lw,color=c)
    plt.title('Mass vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Mass [lbm]')
    plt.xlim(0, t[i_land])
    plt.ylim(min(m)-0.1, max(m)+0.1)
    plt.grid()
    if save_plots:
        fig.savefig(path + 't_m.png')
        