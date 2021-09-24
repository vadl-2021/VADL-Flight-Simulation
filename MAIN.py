# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:42:35 2021

@author: Ryan Burinescu

Run rocket flight simulation and generate plots and/or flysheet.
"""

import os
from parameters import *
from numpy import sin, cos, tan, arctan2, pi, sqrt, interp, real, mean
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import animation
from openpyxl import Workbook
from openpyxl import load_workbook
from shutil import copy
from datetime import date

# SETUP
# ============================================================================

# Plot Setup
# ----------------------------------------------------------------------------
plot_folder = 'Example' # name of subfolder to save flight info/plots in

save_plots = True # save plots as .png files

draw_rocket = True # generate a drawing of the rocket as well as the CP and CG

# x-axis, y-axis
plot_x_z = True # horizontal drift, altitude
plot_t_z = True # time, altitude
plot_t_vz = True # time, vertical velocity
plot_t_az = True # time, vertical 
plot_t_theta = True # time, pitch angle
plot_landing_energies = True # landing energies

plot_thrust_curve = False # time, thrust
plot_z_w = False # altitude, wind speed
plot_t_SSM = False # time, static stability margin
plot_t_m = False # time, mass

# Flysheet Setup
# ----------------------------------------------------------------------------

generate_flysheet = True # set to True to generate flysheet

# Flight Trajectory Animation Setup
# ----------------------------------------------------------------------------

ani_res = 0.01 # animation resolution [s]
generate_animation = True # set to True to generate animation

# ============================================================================

# Specify Plot Parameters
# ----------------------------------------------------------------------------

lw = 2 # linewidth
c = 'black' # line color

ms = 20 # marker size (for flight trajectory animation)

# rocket section names (for landing energy plots)
sections = ['Fore Section', 'Middle Section', 'Aft Section']

# Specify Relevant Limits & Requirements
# ----------------------------------------------------------------------------

# NOTE: This is just for plotting/demonstration purposes (e.g. showing a line
# for the landing energy requirements in a plot)
max_landing_energy = 75 # landing energy limit [ft-lbf]

# Run Rocket Flight Simulation
# ----------------------------------------------------------------------------

stream = open('calculate_trajectory.py')
read_file = stream.read()
exec(read_file)

# Generate Plots
# ----------------------------------------------------------------------------

# create subfolder (if it does not exist)
path = 'Flights/' + plot_folder + '_' + str(date.today()) + '/'
os.makedirs(path, exist_ok=True)

if draw_rocket:
    stream = open('draw_rocket.py')
    read_file = stream.read()
    exec(read_file)
    if save_plots:
        drawing_fig.savefig(path + 'rocket_drawing.png')

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

# time, thrust (motor thrust curve)
if plot_thrust_curve:
    fig = plt.figure()
    plt.plot(motor_data[:,0],motor_data[:,1],linewidth=lw,color=c)
    plt.title('Thrust vs. Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust [N]')
    plt.ylim(0,max(motor_data[:,1]+100))
    plt.grid()
    if save_plots:
        fig.savefig(path + 'thrust_curve.png')

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
        
# Flysheet Generation
# ----------------------------------------------------------------------------
        
if generate_flysheet:
    
    copy('Flysheet.xlsx', path) # copy flysheet to flight folder
    
    filename = path + 'Flysheet.xlsx'
    workbook = load_workbook(filename)
    s = workbook.active # get workbook sheet
    
    # Vehicle Properties
    s['D4'] = round(L,1) # rocket length [in]
    s['D5'] = round(D,1) # rocket diameter [in]
    s['D6'] = round(m[0],1) # gross lift off weight [lb]
    
    # Motor Properties
    s['C13'] = f"{int(max(Th)*N_to_lbf)}/{int(mean(Th[1:i_LRE])*N_to_lbf)}" # max./avg. thrust [lb]
    #s['C14'] = round() # total impulse [lbf-s]
    s['C15'] = f"{round((m_motor[0])*kg_to_lbm*16,1)}/{round((m_motor[i_MECO])*kg_to_lbm*16,1)}" # motor mass before/after burn [oz]
    s['C16'] = round(Th[i_LRE]) # lift off thrust [N]
    
    # Stability Analysis
    s['D20'] = round(CP0,1) # center of pressure [in from nose]
    s['D21'] = round(CG0,1) # center of gravity [in from nose]
    s['D22'] = round(SSM[1],2) # static stablilty margin on pad [cal]
    s['D23'] = round(SSM[i_LRE],2) # static stability margin at rail exit [cal]
    s['D24'] = round(Th_to_W_avg,1) # average thrust-to-weight ratio
    s['D26'] = round(v[i_LRE],1) # rail exit velocity [fps]
    
    # Ascent Analysis
    s['D29'] = v_max # max. velocity [fps]
    s['D30'] = round(max(M_inf),2) # max. Mach number
    s['D31'] = round(a_max*G_to_fps2) # max. acceleration [ft/s^2]
    s['D33'] = apogee # predicted apogee [ft]
    
    # Recovery System Properties - Overall
    s['D36'] = t_descent # total descent time [s]
    s['D37'] = round(20*mph_to_mps*t_descent*m_to_ft) # total drift in 20 mph winds (using product of wind speed and descent time) [ft]
    
    # Recovery System Properties - Drogue Parachute
    s['J19'] = round(Dp_d_f*m_to_in) # drogue parachute diameter [in]
    s['J22'] = round(abs(vz[i_drogue])) # velocity at drogue deployment [fps]
    s['J23'] = round(abs(vz[i_main])) # drogue terminal velocity [fps]
    s['H30'] = round(KE_main_1) # KE at deployment of drogue section 1 [ft-lbf]
    s['I30'] = round(KE_main_2) # KE at deployment of drogue section 2 [ft-lbf]
    s['J30'] = 'N/A'
    s['K30'] ='N/A'
    
    # Recovery System Properties - Main Parachute
    s['J35'] = round(Dp_m_f*m_to_in) # main parachute diameter [in]
    s['J38'] = round(abs(vz[i_main])) # velocity at main deployment [fps]
    s['J39'] = round(abs(vz[i_land])) # main terminal velocity [fps]
    s['H46'] = round(KE_fore) # landing energy of main section 1 [ft-lbf]
    s['I46'] = round(KE_mid) # landing energy of main section 2 [ft-lbf]
    s['J46'] = round(KE_aft) # landing energy of main section 3 [ft-lbf]
    s['K46'] = 'N/A'
    
    workbook.save(filename=filename)
    
# Flight Trajectory Animation
# ----------------------------------------------------------------------------

if generate_animation:
    fig, ax = plt.subplots()
    
    l = plt.plot(x, z, lw=lw, color=c)
    plt.grid()
    plt.xlabel('Horizontal Drift [ft]')
    plt.ylabel('Altitude [ft]')
    plt.title('Flight Trajectory Animation')
    
    time_stamp = ax.text(0.05,0.90, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                    transform=ax.transAxes, ha="left")
    
    ax = plt.axis(xlim=(0, drift+1500), ylim=(0, apogee+500))
    
    dot, = plt.plot([], [], '*', markersize=ms, color=gold)
    
    def animate(i):
        i = int(i)
        if np.mod(i,1/dt/10) == 0:
            time_stamp.set_text(f"t = {round(t[i],1)} s")
        dot.set_data(x[i], z[i])
        return dot, time_stamp,
    
    frames = np.arange(0,len(t),1/dt*ani_res)
    
    # create animation using the animate() function
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                          interval=1/dt*ani_res, blit=True, repeat=True,
                                          repeat_delay=2000, cache_frame_data=False)

    plt.show()
    
