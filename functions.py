# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 18:30:45 2021

@author: Ryan Burinescu

Contains various functions for rocket simulation.
"""

import numpy as np
from numpy import sin, cos, tan, arctan2, pi, sqrt, interp, real, mean
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Geopotential Altitude
# ----------------------------------------------------------------------------

# Calculates the geopotential altitude based on a geometric altitude 
# referenced to the elevation of the launch site above mean sea level.

# INPUT(S): z - geometric altitude above launch site [m]
#           eps - launch site elevation above mean sea level [m]
#           RE - Earth radius [m]
# OUTPUT(S): h - geopotential altitude [m/s^2]

def geopotential_altitude(z,eps,RE):
    h = (RE/(RE+eps+z))*(eps+z)
    return h

# Moment of Inertia
# ----------------------------------------------------------------------------
    
# Calculates the moment of inertia for individual rocket sections.

# INPUT(S): mi - mass of section [m]
#           Li - length of section [m]
#           CG0 - vehicle CG [m]
#           l_CG_i - length from nose cone to section CG [m]
#           Ro_i - outer radius of section [m]
#           Ri_i - inner radius of section [m]
# 
# OUTPUT(S): Ii - moment of inertia of section [m]

def moment_of_inertia(mi,Li,CG0,l_CG_i,Ro_i,Ri_i):
    Ii = (mi/6)*(Li**2+6*(CG0-l_CG_i)**2+(3*(Ro_i**4-Ri_i**4)/(Ro_i**2-Ri_i**2)))
    return Ii

# Aerodynamics
# ----------------------------------------------------------------------------

# Calculates the aerodynamic coefficients and CP on the rocket for given 
# geometries/flight conditions.

# INPUT(S): Re - Reynolds number at given altitude
#           alpha - angle of attack [rad]
#           cm - fin midchord length [m]
#           D - rocket diameter [m]
#           L - entire rocket length [m]
#           L_nose - nose cone length [m]
#           L_body - rocket body total length [m]
#           A_body - rocket body projected side area [m^2]
#           l_tail - length to boat tail [m]
#           L_tail_c - boat tail curved section length [m]
#           L_tail_f - boat tail flat section length [m]
#           D_tail - boat tail aft diameter [m]
#           A_fin - fin area [m^2]
#           A_fin_e - extended fin area [m^2]
#           l_fin - length to fins [m]
#           t_fin - fin thickness [m]
#           s - fin semispan [m]
#           cr - fin root chord length [m]
#           ct - fin tip chord length [m]
#           M_inf - free stream Mach number
#           A - rocket cross-sectional area [m^2]
#           CD_override - boolean CD override
#           CDr - reference drag coefficient [1]
#           
# OUTPUT(S): CA - axial coefficient
#            CN - normal coefficient
#            CD - drag coefficient
#            CL - lift coefficient
#            CP - center of pressure [m]

def aerodynamics(Re,alpha,cm,D,L,L_nose,L_body,A_body,l_tail,L_tail_c,L_tail_f
                 ,D_tail,A_fin,A_fin_e,l_fin,t_fin,s,cr,ct,M_inf,A,CD_override,CDr):

    # we assume the max. angle of attack for these equations is 10 deg
    if alpha > 0.174533:
        alpha = 0.174533
    
    # nose cone
    CNa_nose = 2
    CN_nose = CNa_nose*alpha
    lcp_nose = 0.466*L_nose # for a tangent ogive
    
    # body tube
    CNa_body = (3/2)*(A_body/A)*alpha
    CN_body = CNa_body*alpha
    
    # fins
    xt = (cr-ct)/2
    CNa_fin = (1+(D/(D+2*s)))*((16*(s/D)**2)/(1+sqrt(1+((2*cm)/(cr+ct))**2)))
    CN_fin = CNa_fin*alpha
    lcp_fin = l_fin+(xt/3)*(cr+2*ct)/(cr+ct)+(1/6)*(cr+ct+(cr*ct)/(cr+ct))
    
    # boat tail
    CNa_tail = 2*((D_tail/D)**2-1)
    CN_tail = CNa_tail*alpha
    lcp_tail = l_tail+L_tail_f+(L_tail_c/3)*(1+(1-D/D_tail)/(1-(D/D_tail)**2))
    
    # normal coefficient
    CN = (CN_nose+CN_body+CN_fin+CN_tail)/sqrt(1-M_inf**2)
    
    # Re_core = Re from input
    if Re <= 5e5:
        Cf_core = 1.328/sqrt(Re)
    else:
        Cf_core = 0.074/Re**(1/5)-(5e5)/Re*((0.074)/(Re**(1/5))-(1.328)/(sqrt(Re)))
    
    Re_fin = Re/L*cm
    if Re_fin <= 5e5:
        Cf_fin = 1.328/sqrt(Re_fin)
    else:
        Cf_fin = 0.074/Re_fin**(1/5)-(5e5)/Re_fin*((0.074)/(Re_fin**(1/5))-(1.328)/(sqrt(Re_fin)))
    
    CD_core = ((1+60*(D/L)**3+L_body/(400*D)))*((2.7*L_nose)/D+(4*(L_body+L_tail_f))/D+2*(1-(D_tail/D))*(L_tail_c/D))*Cf_core
    
    CD_fin = ((32*A_fin_e)/(pi*D**2))*((1+((2*t_fin)/cm)))*Cf_fin
    
    CD_fb = ((32*(A_fin_e-A_fin))/(pi*D**2))*((1+((2*t_fin)/cm)))*Cf_fin
    
    CD_base = (0.029/sqrt(CD_fb))*(D_tail/D)**3
    
    CD0 = CD_core+CD_fin+CD_fb+CD_base
    
    delta = 0.716*alpha+0.73
    
    eta = 0.573*alpha+0.56
    
    CD_core_a = 2*delta*alpha**2+(3.6*eta*alpha**3*(1.36*L-0.55*L_nose))/(pi*D**2)
    
    R_fin = (2*s+D)/D
    
    k_fb = 0.8065*R_fin**2+1.553*R_fin
    
    k_bf = 0.1935*R_fin**2+0.8147*R_fin+1
    
    CD_fin_a = alpha**2*((4.8*A_fin_e)/(pi*D**2)+((12.48*A_fin)/(pi*D**2))*(k_fb-k_bf-1))
    
    # drag coefficient
    CD = (CD0+CD_core_a+CD_fin_a)/sqrt(1-M_inf**2)
    
    # override drag coefficient
    if CD_override:
        CD = CDr*(alpha*(.3/.2967)+1)
    
    # axial coefficient
    CA = (1/cos(alpha))*(CD-CN*sin(alpha))
    
    # lift coefficient
    CL = (CN-CD*sin(alpha))/cos(alpha)
    
    # center of pressure
    CP = (CNa_nose*lcp_nose+CNa_fin*lcp_fin+CNa_tail*lcp_tail)/(CNa_nose+CNa_fin+CNa_tail)
        
    return [CA,CN,CD,CL,CP]

# Atmospheric Conditions
# ----------------------------------------------------------------------------
    
# Calculates the density, dynamic viscosity, and temperature of air at a 
# given altitude.

# INPUT(S): z - altitude above launch site [m]
#           T0 - air temperature at launch site [K]
#           p0 - air pressure at launch site [Pa]
#           R - ideal gas constant for air [J/(kg.K)]
#           B - temperature lapse rate in the troposphere [K/m]
#           mu_data - array storing dynamic viscosity [Pa.s] as a function
#                     of temperature [K]
# OUTPUT(S): rho - air density [kg/m^3]
#            T - air temperaure [K]
#            p - air pressure [Pa]
#            mu - air dynamic viscosity [Pa.s]
    
def atmosphere(h,T0,p0,R,B,g0,mu_data):

    # air temperature
    T = T0 - B*h
    
    # air pressure
    p = p0*(T/T0)**(g0/(R*B))
    
    # air density
    rho = p/(R*T)
    
    # interpolation to find dynamic viscosity
    mu = interp(T,mu_data[:,0],mu_data[:,1])
    
    return [rho,T,p,mu]

# Motor State
# ----------------------------------------------------------------------------
    
# Provides the instantaneous state of the rocket motor.

# INPUT(S): t - time since launch [s]
#           motor_data - burn time [s], thrust [N], propellant mass [kg]
#           m_casing - motor casing mass [kg]
#           
# OUTPUT(S): Th - thrust [N]
#            m_motor - motor mass [kg]
#            m_prop - propellant mass [kg]
#            mdot - propellant mass flow rate [kg/s]

def motor_state(t,motor_data,m_casing):

    # thrust
    Th = interp(t,motor_data[:,0],motor_data[:,1])
    
    # propellant mass
    m_prop = interp(t,motor_data[:,0],motor_data[:,2])
    
    # total motor mass
    m_motor = m_prop + m_casing
    
    # propellant mass flow rate
    m_prop0 = motor_data[0,2]
    mdot = (m_prop0-m_prop)/t

    return [Th,m_motor,m_prop,mdot]

# Rocket Inertia
# ----------------------------------------------------------------------------

# Calculates the inertial parameters of the rocket.

# INPUT(S): m_vehicle - rocket (including lander) dry mass [kg]
#           CG_vehicle - rocket with lander dry CG (excluding motor) [m]
#           m_motor - instantaneous mass of the motor and casing [kg]
#           l_CG_motor - length (from nose cone) of motor CG [m]
#           D_motor - motor diameter [m]
#           m_prop0 - initial mass of propellant [kg]
#           m_prop - instantaneous mass of propellant [kg]
#           m_motor - initial mass of motor [kg]
#           L_motor - motor length [m]
#           CG0 - initial rocket wet CG [m]
#           I0 - initial rocket moment of inertia [kg*m^2]
#           
# OUTPUT(S): m - instaneous total mass [kg]
#            CG - instantaneous CG [m]
#            I - instantaneous moment of inertia [kg*m^2]

def rocket_inertia(m_vehicle,CG_vehicle,m_motor,l_CG_motor,D_motor,m_prop0,
                   m_prop,m_motor0,L_motor,CG0,I0):
    
    # total mass
    m = m_vehicle + m_motor
    
    # CG
    CG = (m_vehicle*CG_vehicle+m_motor*l_CG_motor)/(m_vehicle+m_motor)
    
    # moment of inertia
    I_burn_CG0 = ((m_prop0-m_prop)/6)*(L_motor**2+6*(CG0-l_CG_motor)**2+12*(D_motor)**2*((m_prop0-m_prop)/m_prop0)**2)
    I_CG0 = I0-I_burn_CG0
    I = I_CG0-m**(CG0-CG)**2
    
    return [m,CG,I]    

# Acceleration due to Gravity
# ----------------------------------------------------------------------------

# Calculates the gravitational acceleration at a specified altitude above the 
# ground level of the launch site.

# INPUT(S): g0 - gravitational acceleration at mean sea level [m/s^2]
#           z - altitude above launch site [m]
#           eps - launch site elevation above mean sea level [m]
#           RE - Earth radius [m]
# OUTPUT(S): g - gravitational acceleration [m/s^2]
    
def gravity(g0,z,eps,RE):
    g = g0*(RE/(RE+eps+z))**2
    return g

# Wind
# ----------------------------------------------------------------------------
    
# Calculates the wind velocity at a specified altitude above the launch site.

# INPUT(S): wr - reference (base) wind velocity [m/s]
#           zr - reference (base) altitude [m]
#           z - altitude above launch site [m]
#           xi - denominator of exponent in power law_
# OUTPUT(S): w - wind speed [m/s]
    
def wind(wr,zr,z,xi):
    # calculates sustained wind speed
    w = wr*(z/zr)**(1/xi)
    return w

# Parachute State
# ----------------------------------------------------------------------------
    
# Calculates the drag coefficient, nominal (surface) 
# area, and projected side area of a parachute as a function of time as the
# parachute opens (if the parachute is approximately fully open, the
# fully-opened values are returned).

# INPUT(S): t_open - time since deployment [s]
#           tf - filling time (time to fully open) [s]
#           Dp_f - fully-open projected diameter [m]projected
#           CD_f - fully-open drag coefficient (nominal or ) [1]
#           CD_side_f - fully-open lateral drag coefficient [1]
#           CL_side_f - fully-open lateral lift coefficient [1]
#           CD_type - type of drag coefficient (nominal or projected)
# OUTPUT(S): CD - instantaneous drag coefficient (nominal or projected) [1]
#            CD_side - instantaneous sideslip drag coefficient [1]
#            CL_side - instantaneous sideslip lift coefficient [1]
#            S - instantaneous parachute area (nominal or projected) [m^2]
#            S_side - instantaneous sideslip area [m^2]
    
def parachute_state(t_open,tf,Dp_f,CD_f,CD_side_f,CL_side_f,CD_type):
    
    # drag coefficient, lift coefficient, nominal area, and projected
    # diameter of parachute
    if t_open < tf:
        Dp = Dp_f*(t_open/tf)
        CD = 1 + (CD_f-1)*(t_open/tf)**2
        CD_side = 1 + (CD_side_f-1)*(t_open/tf)**2
        CL_side = CL_side_f*(CD/CD_f)
    else:
        Dp = Dp_f
        CD = CD_f
        CD_side = CD_side_f
        CL_side = CL_side_f
    
    # projected area
    Sp = pi*Dp**2/4

    # nominal area
    So = Sp*2
    
    # projected side area of parachute
    S_side = Sp/2
    
    # selects which area to return
    if CD_type == 'nominal':
        S = So
    else:
        S = Sp
    
    return [CD,CD_side,CL_side,S,S_side]

# Draw Rectangle
# ----------------------------------------------------------------------------

def draw_rectangle(start,L,R,res,col,lwid):
    
    finish = start+L
    x = np.arange(start,finish,res)
    y = np.arange(-R,R,res)
    
    plt.plot(x,R*np.ones(len(x)),color=col,linewidth=lwid) # top
    plt.plot(x,-R*np.ones(len(x)),color=col,linewidth=lwid) # bottom
    plt.plot(start*np.ones(len(y)),y,color=col,linewidth=lwid) # left
    plt.plot(finish*np.ones(len(y)),y,color=col,linewidth=lwid) # right
