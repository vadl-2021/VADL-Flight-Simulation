# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:30:00 2021

@author: Ryan Burinescu

Specifies input parameters for simulated rocket flight.
"""

import yaml
import numpy as np
import pandas as pd
from functions import *

# Specify Configuration Files
# ============================================================================

# rocket configuration file
rocket_config = 'example_rocket.yaml'

# motor configuration file
motor_config = 'example_motor.yaml'

# launch configuration file
launch_config = 'example_launch.yaml'

# ============================================================================

# Load Configuration Files
# ----------------------------------------------------------------------------

# rocket configuration
with open('Configuration Files/Rocket Configuration Files/' + rocket_config, 'r') as stream:
    rc = yaml.safe_load(stream)

# motor configuration   
with open('Configuration Files/Motor Configuration Files/' + motor_config, 'r') as stream:
    mc = yaml.safe_load(stream)
    
# launch configuration
with open('Configuration Files/Launch Configuration Files/' + launch_config, 'r') as stream:
    lc = yaml.safe_load(stream)

# Data Files
# ----------------------------------------------------------------------------

# rocket section masses and lengths

# FORMAT: | Section Name | Mass [lbm] | Length [in] |
# first row must be nose cone
# last row must be boat tail
rocket_data_file = rc['rocket_data_file']

# motor thrust (and optional propellent mass) data

# FORMAT: | Time [s] | Thrust [N] | Mass [kg] (optional) |
thrust_data_file = mc['thrust_data_file']

# Conversion Factors
# ----------------------------------------------------------------------------

# inputs
ft_to_m = 0.3048
in_to_m = 0.0254
deg_to_rad = 0.0174533
lbm_to_kg = 0.453592
mph_to_mps = 0.44704
in2_to_m2 = 0.00064516

# outputs
mps2_to_G = 0.101971621
m_to_ft = 3.28084
m_to_in = 39.3701
rad_to_deg = 57.2958
N_to_lbf = 0.2248089431
J_to_ftlbf = 0.737562
kg_to_lbm = 2.20462
G_to_fps2 = 32.174

# Hardcoded Data Overrides
# ----------------------------------------------------------------------------

# rocket drag coefficient
CD_override = rc['CD_override'] # set to true to override axial rocket CD
CDr = rc['CDr'] # hardcoded rocket CD

# vehicle center of gravity (***not including motor components***)
CG_vehicle_override = rc['CG_vehicle_override'] # set to true to override vehicle CG
CG_vehicle = rc['CG_vehicle'] # hardcoded vehicle CG [m]
CG_vehicle = CG_vehicle*in_to_m

# center of pressure
CP_override = rc['CP_override'] # set to True to override CP
CP0 = rc['CP0'] # hardcoded CP [m]
CP0 = CP0*in_to_m

# Computational Parameters
# ----------------------------------------------------------------------------

dt = lc['dt'] # time step [s]
max_simulation_time = lc['max_simulation_time'] # max. allowable time for simulation [s]

# Mission Sequence Parameters
# ----------------------------------------------------------------------------

# boolean overrides
drogue_failure = lc['drogue_failure'] # set to True to override drogue deployment
main_failure = lc['main_failure'] # set to True to override main deployment

# parachute deployments
drogue_time_delay = lc['drogue_time_delay'] # time after apogee for drogue deployment [s]
z_deploy_main = lc['z_deploy_main']*ft_to_m # main parachute deployment altitude [m]

# Launch Setup Parameters
# ----------------------------------------------------------------------------

theta_rail = lc['theta_rail']*deg_to_rad # launch angle [rad]
L_rail = lc['L_rail']*in_to_m # launch rail length [m]

# Physical Parameters
# ----------------------------------------------------------------------------

# weather dependent
T0 = lc['T0'] # U.S. Standard Atmosphere surface air temperature [K]
p0 = lc['p0'] # U.S. Standard Atmosphere surface air pressure [Pa]

# constants
g0 = 9.80665 # gravitational acceleration at mean sea level [m/s^2]
rho0 = 1.225 # U.S. Standard Atmosphere surface air density [kg/m^3]
B = 6.5e-3 # temperature lapse rate in troposphere [K/m]
R = 287 # ideal gas constant for air [J/(kg*K)]                                                               
gamma = 1.4 # spec. heat ratio for air 

# dynamic viscosity data
mu_data = np.loadtxt('dynamic_viscosity_data.csv',skiprows=1,delimiter=',')

# wind
wr = lc['wr']*mph_to_mps # reference wind speed [m/s]
zr = 10 # base altitude [m]
xi = 7 # power law denominator

# initial geopotential altitude
eps = lc['eps']*ft_to_m # launch site elevation [m]
RE = 6.37e6 # Earth radius [m]

# Rocket Section Mass and Length Parameters
# ----------------------------------------------------------------------------

# extra mass (unaccounted for from section mass totals) [kg]
# assumed to be distributed evenly throughout the rocket
m_extra = rc['m_extra']*lbm_to_kg

# masses and lengths of rocket sections (from spreadsheet)
fname = 'Rocket Section Data/' + rocket_data_file
rocket_data = pd.read_excel(fname, usecols='B:C').to_numpy()
mass_array = rocket_data[:,0]*lbm_to_kg # rocket section mass array [kg]
L_array = rocket_data[:,1]*in_to_m # rocket section length array [m]
N_sections = len(rocket_data) # number of rocket sections

# masses of recovery sections (for kinetic energy calculations)
fore_start = rc['fore_start']
fore_end = rc['fore_end']
mid_start = rc['mid_start']
mid_end = rc['mid_end']
m_fore = sum(mass_array[fore_start:fore_end+1]) # fore section
m_mid = sum(mass_array[mid_start:mid_end+1]) # middle section
m_aft = sum(mass_array[mid_end+1:N_sections+1]) # aft section

# Rocket Geometry Parameters
# ----------------------------------------------------------------------------

D = rc['D']*in_to_m # rocket diameter [m]

l_frb = rc['l_frb']*in_to_m # forward rail button location (from nose cone) [m]

t_airframe = rc['t_airframe']*in_to_m # rocket airframe thickness [m]

CDr_side = rc['CDr_side'] # rocket side drag coefficient

# fins
l_fin = rc['l_fin']*in_to_m # length to fins (from nose cone) [m]
cm = rc['cm']*in_to_m # fin midchord length [m]
cr = rc['cr']*in_to_m # fin root length [m]
ct = rc['ct']*in_to_m # fin tip length [m]
s = rc['s']*in_to_m # fin semispan [m]
t_fin = rc['t_fin']*in_to_m # fin thickness [m]

# boat tail
L_tail_c = rc['L_tail_c']*in_to_m # boat tail curved section length [m]
L_tail_f = rc['L_tail_f']*in_to_m # boat tail flat section length [m]
L_tail_rr = rc['L_tail_rr']*in_to_m # retaining ring length from end of boat tail [m]
D_tail = rc['D_tail']*in_to_m # boat tail aft diameter [m]
D_tail_rr = rc['D_tail_rr']*in_to_m # retaining ring diameter [m]

# Rocket Motor Parameters
# ----------------------------------------------------------------------------

# motor geometry
L_motor = mc['L_motor'] # motor length [m]
D_motor = mc['D_motor'] # motor diameter [m]

# motor mass
m_motor0 = mc['m_motor0'] # total motor initial mass [kg]
m_prop0 = mc['m_prop0'] # initial motor propellent mass [kg]
m_casing = m_motor0 - m_prop0 # motor casing mass [kg]

# motor data
fname = 'Motor Data/' + thrust_data_file
motor_data = pd.read_csv(fname).to_numpy()

t_burn = motor_data[-1,0] # motor burn time

# add propellent mass [kg] to motor data array (if not included in csv)
m_prop = np.zeros(len(motor_data))[...,None]
if motor_data.shape[1] < 3:
    for i in range(len(motor_data)):
        slope = m_prop0/t_burn
        m_prop[i] = m_prop0 - slope*motor_data[i,0]
    motor_data = np.append(motor_data,m_prop,1) # propellent mass [kg]
    
# Drogue Parachute Parameters
# ----------------------------------------------------------------------------
    
Dp_d_f = rc['Dp_d_f']*in_to_m # projected diameter of fully-open parachute [m]
CD_d_f = rc['CD_d_f'] # drag coefficient of fully-open parachute
n_d = rc['n_d'] # drogue parachute-specific parameter (toroidal = 9, elliptical = 8)
m_d = rc['m_d']*lbm_to_kg # drogue parachute mass [kg]

L_shock_cord_d = rc['L_shock_cord_d']*ft_to_m # total drogue shock cord length [m]

# specifies type of drag coefficient given ("projected" or "nominal")
# NOTE: Fruity-Chutes provides projected drag coefficient
CD_d_type = rc['CD_d_type']

# Main Parachute Parameters
# ----------------------------------------------------------------------------

Dp_m_f = rc['Dp_m_f']*in_to_m # main parachute diameter [m]
CD_m_f = rc['CD_m_f'] # main parachute vertical drag coefficient
n_m = rc['n_m'] # main parachute-specific parameter (toroidal = 9, elliptical = 8)
m_m = rc['m_m']*lbm_to_kg # main parachute mass [kg]

L_shock_cord_m = rc['L_shock_cord_m']*ft_to_m # total main shock cord length [m]

# specifies type of drag coefficient given ("projected" or "nominal")
# NOTE: Fruity-Chutes provides projected drag coefficient
CD_m_type = rc['CD_m_type']

# General Parachute Parameters
# ----------------------------------------------------------------------------

CD_side_f = 0.38699 # parachute side drag coefficient (from CFD)
CL_side_f = 0.36465 # parachute side lift coefficient (from CFD)

# Derived Parameters
# ----------------------------------------------------------------------------

# vehicle mass (without motor casing and propellent) [kg]
m_vehicle = sum(mass_array)

# total vehicle length [m]
L = sum(L_array)

# nose cone length
L_nose = L_array[0]

# length to boat tail
l_tail = L-(L_tail_c+L_tail_f+L_tail_rr)

# dry mass [kg]
m_dry = m_vehicle + m_casing

# lengths to CGs of individual sections
l_CG_array = np.zeros(N_sections)
l_CG_array[0] = L_array[0]/2 # nose cone CG
for i in range(1,N_sections):
    l_CG_array[i] = sum(L_array[0:i]) + L_array[i]/2

l_CG_motor = L - L_motor/2 # length to motor CG

# calculate vehicle CG if not hardcoded
if CG_vehicle_override == False:
    CG_vehicle = 0
    for i in range(N_sections):
        CG_vehicle += mass_array[i] * l_CG_array[i]
    CG_vehicle = CG_vehicle/(m_vehicle - m_extra)
    
# initial wet CG
CG0 = (m_vehicle*CG_vehicle+m_motor0*l_CG_motor)/(m_vehicle+m_motor0)

# initial launch altitude
h_launch = geopotential_altitude(CG0, eps, RE)

# initial motor moment of inertia
I_motor_0_CG0 = m_motor0/24*(4*L_motor**2+24*(CG0-l_CG_motor)**2+3*D_motor**2)

# moments of inertia of individual sections
Ro = D/2 # outer radius [m]
Ri = Ro-t_airframe # inner radius [m]
I_array = np.zeros(N_sections)
for i in range(N_sections):
    I_array[i] = moment_of_inertia(mass_array[i], L_array[i], CG0, 
                                   l_CG_array[i], Ro, Ri)

# initial wet moment of inertia
I0 = I_motor_0_CG0 + sum(I_array)

# fins
A_fin = (s*(cr+ct))/2 # fin projected area [m^2]
A_fin_e = A_fin+(cr*D)/2 # fin extended area [m^2]

# nose cone
A_nose = (L_nose*D)/2 # approximate nose cone projected side area (triange) [m^2]

# boat tail
L_tail = L_tail_c+L_tail_f # total boat tail length [m]
A_tail = L_tail_f*D+(L_tail_c*(D+D_tail))/2+L_tail_rr*D_tail_rr # boat tail projected side area [m^2]

# cylindrical body section
L_body = L-L_nose-L_tail # total body section length [m]
A_body = L_body*D # body section projected side area [m^2]

# overall rocket area
A = (np.pi/4)*D**2 # rocket axial projected area [m^2]
Ar_side = A_body+A_nose+A_tail # total rocket side area [m^2]

# drogue parachute
Do_d_f = Dp_d_f*np.sqrt(2) # nominal diameter of fully open drogue parachute [m]
Sp_d_f = (np.pi/4)*Dp_d_f**2
So_d_f = 2*Sp_d_f
S_side_d = Sp_d_f/2

# main parachute
Do_m_f = Dp_d_f*np.sqrt(2) # nominal diameter of fully open main parachute [m]
Sp_m_f = (np.pi/4)*Dp_d_f**2
So_m_f = 2*Sp_d_f
S_side_m = Sp_d_f/2

# center of pressure
if CP_override == False:
    [_,_,_,_,CP0] = aerodynamics(0,0,cm,D,L,L_nose,L_body,A_body,l_tail,L_tail_c,
                                 L_tail_f,D_tail,A_fin,A_fin_e,l_fin,t_fin,s,cr,
                                 ct,0,A,CD_override,CDr)

