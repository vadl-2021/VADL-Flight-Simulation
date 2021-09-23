# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:22:58 2021

@author: Ryan Burinescu

Simulates a dual-deployment rocket flight based on a set of input parameters.
"""

import os
from numpy import sin, cos, tan, arctan2, pi, sqrt, interp, real, mean
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Preallocate Arrays
# ----------------------------------------------------------------------------

t = np.arange(0,max_simulation_time,dt) # time array [s]
N = len(t) # time array length

# physical arrays
rho = np.zeros(N) # air density at rocket altitude [kg/m^3]
rho_d = np.zeros(N) # air density at drogue parachute altitude [kg/m^3]
rho_m = np.zeros(N) # air density at main parachute altitude [kg/m^3]
mu = np.zeros(N) # air viscosity at rocket altitude [Pa*s]
T = np.zeros(N) # air temperature at rocket altitude [K]
p = np.zeros(N) # air pressure at rocket altitude [Pa]
g = np.zeros(N) # gravitational acceleration at rocket altitude [m/s^2]

# rocket kinematics arrays (center of gravity)
a = np.zeros(N) # total acceleration [m/s^2]
ax = np.zeros(N) # rocket x acceleration [m/s^2]
az = np.zeros(N) # rocket z acceleration [m/s^2]
v = np.zeros(N) # total rocket velocity [m/s]
vx = np.zeros(N) # rocket x velocity [m/s]
vz = np.zeros(N) # rocket z velocity [m/s]
w = np.zeros(N) # wind velocity at rocket altitude [m/s]
x = np.zeros(N) # rocket drift [m]
z = np.zeros(N) # rocket geometric altitude [m]
h = np.zeros(N) # rocket geopotential altitude [m]
theta_ddot = np.zeros(N) # rocket angular acceleration [rad/s^2]
theta_dot = np.zeros(N) # rocket angular velocity [rad/s]
theta = np.zeros(N) # rocket pitch angle [rad]

# kinematics arrays (forward rail button)
xfrb = np.zeros(N) # forward rail button x drift [m/s]
zfrb = np.zeros(N) # forward rail button z altitude [m/s]

# kinematics arrays (center of pressure)
v_CP_x = np.zeros(N) # rocket CP x velocity [m/s]
v_CP_z = np.zeros(N) # rocket CP z velocity [m/s]

# kinematics arrays (parachutes)
zd = np.zeros(N) # drogue parachute altitude [m]
hd = np.zeros(N) # drogue parachute geopotential altitude [m]
wd = np.zeros(N) # wind velocity at drogue parachute altitude [m]
xm = np.zeros(N) # main parachute drift [m]
zm = np.zeros(N) # main parachute altitude [m]
hm = np.zeros(N) # main parachute geopotential altitude [m]
vxm = np.zeros(N) # main parachute horizontal velocity [m/s]
vzm = np.zeros(N) # main parachute vertical velocity [m/s]
wm = np.zeros(N) # wind velocity at main parachute altitude [m]
axm = np.zeros(N) # main parachute horizontal acceleration [m/s^2]
azm = np.zeros(N) # main parachute vertical acceleration [m/s^2]

# main parachute arrays
CD_m = np.zeros(N) # main parachute drag coefficient 
CD_m_side = np.zeros(N) # main parachute lateral drag coefficient 
CL_m_side = np.zeros(N) # main parachute lateral lift coefficient 
S_m = np.zeros(N) # main parachute area [m^2]
S_m_side = np.zeros(N) # main parachute side area [m^2]

# drogue parachute arrays
CD_d = np.zeros(N) # drogue parachute drag coefficient 
CD_d_side = np.zeros(N) # drogue parachute lateral drag coefficient 
CL_d_side = np.zeros(N) # drogue parachute lateral lift coefficient 
S_d = np.zeros(N) # drogue parachute area [m^2]
S_d_side = np.zeros(N) # drogue parachute side area [m^2]

# inertial arrays
Th = np.zeros(N) # motor thrust [N]
Th_n = np.zeros(N) # normal motor thrust [N]
Th_t = np.zeros(N) # tangential motor thrust [N]
m = np.zeros(N) # rocket mass [kg]
m_motor = np.zeros(N) # motor mass [kg]
m_prop = np.zeros(N) # propellant mass [kg]
mdot = np.zeros(N) # mass flow rate [kg/s]
CG = np.zeros(N) # rocket center of gravity [m]
I = np.zeros(N) # rocket moment of inertia [kg*m^2]

# aerodynamic arrays
CP = np.zeros(N) # center of pressure [m]
Re = np.zeros(N) # Reynolds number 
M_inf = np.zeros(N) # freestream Mach number 
SSM = np.zeros(N) # rocket static statibility margin [cal]
phi = np.zeros(N) # freestream velocity direction [rad]
alpha = np.zeros(N) # angle of attack [rad]
CA = np.zeros(N) # axial coefficient 
CN = np.zeros(N) # normal coefficient 
CD = np.zeros(N) # drag coefficient 
CL = np.zeros(N) # lift coefficient 
v_inf = np.zeros(N) # freestream velocity [m/s]
v_inf_x = np.zeros(N) # horizontal freestream velocity during descent [m/s]
v_inf_z = np.zeros(N) # vertical freestream velocity during descent [m/s]

# force arrays (rocket)
FD_x = np.zeros(N) # lateral drag force on system during descent [N]
FD_z = np.zeros(N) # vertical drag force on system during descent [N]
FA = np.zeros(N) # axial force on rocket during ascent [N]
FN = np.zeros(N) # normal force on system during ascent [N]
FDr = np.zeros(N) # vertical drag force on rocket body during descent [N]
FDr_side = np.zeros(N) # lateral drag force on rocket body during descent [N]

# force arrays (parachutes)
FDd = np.zeros(N) # vertical drag force on drogue parachute [N]
FLd = np.zeros(N) # vertical lift force on drogue parachute [N]
FDd_side = np.zeros(N) # lateral drag force on drogue parachute [N]
FDm = np.zeros(N) # vertical drag force on main parachute [N]
FLm = np.zeros(N) # vertical lift force on main parachute [N]
FDm_side = np.zeros(N) # lateral drag force on main parachute [N]

# Non-Zero Initial Conditions
# ----------------------------------------------------------------------------

# launch angle
theta[0] = theta_rail

# position of CG
x0 = CG0*sin(theta[0])
z0 = CG0*cos(theta[0])
x[0] = x0
z[0] = z0

# geopotential altitude
h[0] = h_launch

# atmospheric conditions
[rho[0],T[0],p[0],mu[0]] = atmosphere(h[0],T0,p0,R,B,g0,mu_data)

# motor state
[Th[0],m_motor[0],m_prop[0],mdot[0]] = motor_state(t[0],motor_data,m_casing)

# inertial properties
[m[0],CG[0],I[0]] = rocket_inertia(m_vehicle,CG_vehicle,m_motor[0],
                                     l_CG_motor,D_motor,m_prop0,m_prop[0],
                                     m_motor0,L_motor,CG0,I0)

# center of pressure
CP[0] = CP0

# static stability margin
SSM[0] = (CP[0]-CG[0])/D

# (1) Powered Ascent on Launch Rail
# ----------------------------------------------------------------------------

# loop index
i = 0

# trajectory integration
while (norm([xfrb[i],zfrb[i]])) < L_rail:
    
    # geopotential altitude
    h[i] = geopotential_altitude(z[i],eps,RE)
    
    # gravitational acceleration
    g[i] = gravity(g0,z[i],h_launch,RE)
    
    # thrust, motor mass, and propellant mass
    [Th[i],m_motor[i],m_prop[i],mdot[i]] = motor_state(t[i],motor_data,m_casing)
    
    # inertial parameters
    [m[i],CG[i],I[i]] = rocket_inertia(m_vehicle,CG_vehicle,m_motor[i],
                                         l_CG_motor,D_motor,m_prop0,m_prop[i],
                                         m_motor0,L_motor,CG0,I0)
    
    # center of pressure and static stability margin
    CP[i] = CP0
    SSM[i] = (CP[i]-CG[i])/D
    
    # accelerations
    ax[i+1] = (Th[i]/m[i])*sin(theta[i])
    az[i+1] = (Th[i]/m[i])*cos(theta[i]) - g[i]
    a[i+1] = norm([ax[i+1],az[i+1]])
    
    # integration of kinematics
    x[i+1] = x[i] + vx[i]*dt
    z[i+1] = max(z[i] + vz[i]*dt,0)
    if z[i+1] < 0:
        z[i+1] = 0
    vx[i+1] = max(vx[i] + ax[i]*dt,0)
    vz[i+1] = max(vz[i] + az[i]*dt,0)
    v[i+1] = max(norm([vx[i+1],vz[i+1]]),0)
    theta[i+1] = theta_rail
    
    # position of forward rail button
    xfrb[i+1] = x[i+1] - (l_frb-CG_vehicle)*sin(theta[i+1])
    zfrb[i+1] = z[i+1] - (l_frb-CG_vehicle)*cos(theta[i+1])
    
    # increments time
    t[i+1] = t[i] + dt
    
    # increments loop index
    i = i+1

# stores iteration number at main launch rail exit (LRE)
i_LRE = i

# (2) Off-Rail Powered Ascent
# ----------------------------------------------------------------------------

# trajectory integration
while t[i] < (t_burn-dt):
    
    # geopotential altitude
    h[i] = geopotential_altitude(z[i],eps,RE)
    
    # gravitational acceleration
    g[i] = gravity(g0,z[i],eps,RE)
    
    # thrust, motor mass, and propellant mass
    [Th[i],m_motor[i],m_prop[i],mdot[i]] = motor_state(t[i],motor_data,m_casing)
    
    # inertial parameters
    [m[i],CG[i],I[i]] = rocket_inertia(m_vehicle,CG_vehicle,m_motor[i],
                                         l_CG_motor,D_motor,m_prop0,m_prop[i],
                                         m_motor0,L_motor,CG0,I0)
    
    # stability margin
    SSM[i] = (CP[i-1]-CG[i])/D
    
    # thrust vectors (only real component of Th_t is taken because at very
    # small Th/mdot the argument of square root may become negative)
    Th_n[i] = mdot[i]*theta_dot[i-1]*(L-CG[i])
    Th_t[i] = mdot[i]*sqrt((Th[i]/mdot[i])**2-((L-CG[i])*theta_dot[i])**2)
    Th_t[i] = real(Th_t[i])
    
    # center of pressure velocities
    v_CP_x[i] = vx[i] + D*theta_dot[i]*cos(theta[i])*SSM[i]
    v_CP_z[i] = vz[i] - D*theta_dot[i]*sin(theta[i])*SSM[i]

    # flow conditions
    w[i] = wind(wr,zr,z[i],xi)
    
    phi[i] = arctan2(v_CP_x[i]+w[i],v_CP_z[i])
    v_inf[i] = sqrt((v_CP_x[i]+w[i])**2 + v_CP_z[i]**2)
    alpha[i] = phi[i] - theta[i]
    [rho[i],T[i],p[i],mu[i]] = atmosphere(h[i],T0,p0,R,B,g0,mu_data)
    Re[i] = rho[i]*v_inf[i]*L/mu[i]
    M_inf[i] = v_inf[i]/sqrt(gamma*R*T[i])
    
    # aerodynamic coefficients & CP
    [CA[i],CN[i],CD[i],CL[i],CP[i]] = aerodynamics(Re[i],alpha[i],cm,D,L,
                                                   L_nose,L_body,A_body,l_tail,
                                                   L_tail_c,L_tail_f,D_tail,
                                                   A_fin,A_fin_e,l_fin,t_fin,s,
                                                   cr,ct,M_inf[i],A,
                                                   CD_override,CDr)
    
    # aerodynamic forces
    FA[i] = 0.5*CA[i]*A*rho[i]*v_inf[i]**2
    FN[i] = 0.5*CN[i]*A*rho[i]*v_inf[i]**2
    
    # accelerations
    ax[i+1] = ((Th_n[i]-FN[i])/m[i])*cos(theta[i])+((Th_t[i]-FA[i])/m[i])*sin(theta[i])
    az[i+1] = ((FN[i]-Th_n[i])/m[i])*sin(theta[i])+((Th_t[i]-FA[i])/m[i])*cos(theta[i]) - g[i]
    a[i+1] = norm([ax[i+1],az[i+1]])
    theta_ddot[i+1] = ((FN[i]*D*SSM[i])-(mdot[i]*theta_dot[i]*(L-CG[i])**2))/I[i]
    
    # integration of kinematics
    x[i+1] = x[i] + vx[i]*dt
    z[i+1] = z[i] + vz[i]*dt
    vx[i+1] = vx[i] + ax[i]*dt
    vz[i+1] = vz[i] + az[i]*dt
    v[i+1] = norm([vx[i+1],vz[i+1]])
    theta[i+1] = theta[i] + theta_dot[i]*dt
    theta_dot[i+1] = theta_dot[i] + theta_ddot[i]*dt
    
    # increments time
    t[i+1] = t[i] + dt
    
    # increments loop index
    i = i+1

# stores iteration number at main engine cutoff (MECO)
i_MECO = i

# (3) Coast Ascent
# ----------------------------------------------------------------------------

# trajectory integration
while vz[i] > 0:
    
    # geopotential altitude
    h[i] = geopotential_altitude(z[i],eps,RE)
    
    # gravitational acceleration
    g[i] = gravity(g0,z[i],eps,RE)
    
    # inertial parameters
    m[i] = m[i-1]
    CG[i] = CG[i-1]
    I[i] = I[i-1]
    
    # stability margin
    SSM[i] = (CP[i-1]-CG[i])/D
    
    # center of pressure velocities
    v_CP_x[i] = vx[i] + D*theta_dot[i]*cos(theta[i])*SSM[i]
    v_CP_z[i] = vz[i] - D*theta_dot[i]*sin(theta[i])*SSM[i]

    # flow conditions
    w[i] = wind(wr,zr,z[i],xi)
    
    phi[i] = arctan2(v_CP_x[i]+w[i],v_CP_z[i])
    v_inf[i] = sqrt((v_CP_x[i]+w[i])**2 + v_CP_z[i]**2)
    alpha[i] = phi[i] - theta[i]
    [rho[i],T[i],p[i],mu[i]] = atmosphere(h[i],T0,p0,R,B,g0,mu_data)
    Re[i] = rho[i]*v_inf[i]*L/mu[i]
    M_inf[i] = v_inf[i]/sqrt(gamma*R*T[i])
    
    # aerodynamic coefficients & CP
    [CA[i],CN[i],CD[i],CL[i],CP[i]] = aerodynamics(Re[i],alpha[i],cm,D,L,
                                                   L_nose,L_body,A_body,l_tail,
                                                   L_tail_c,L_tail_f,D_tail,
                                                   A_fin,A_fin_e,l_fin,t_fin,s,
                                                   cr,ct,M_inf[i],A,
                                                   CD_override,CDr)
    
    # aerodynamic forces
    FA[i] = 0.5*CA[i]*A*rho[i]*v_inf[i]**2
    FN[i] = 0.5*CN[i]*A*rho[i]*v_inf[i]**2
    
    # accelerations
    ax[i+1] = -(FN[i]/m[i])*cos(theta[i]) - (-FA[i]/m[i])*sin(theta[i])
    az[i+1] = (FN[i]/m[i])*sin(theta[i])-(FA[i]/m[i])*cos(theta[i]) - g[i]
    a[i+1] = norm([ax[i+1],az[i+1]])
    theta_ddot[i+1] = ((FN[i]*D)/I[i])*SSM[i]
    
    # integration of kinematics
    x[i+1] = x[i] + vx[i]*dt
    z[i+1] = z[i] + vz[i]*dt
    vx[i+1] = vx[i] + ax[i]*dt
    vz[i+1] = vz[i] + az[i]*dt
    v[i+1] = norm([vx[i+1],vz[i+1]])
    theta[i+1] = theta[i] + theta_dot[i]*dt
    theta_dot[i+1] = theta_dot[i] + theta_ddot[i]*dt
    
    # increments time
    t[i+1] = t[i] + dt
    
    # increments loop index
    i = i+1

# stores iteration number at apogee
i_apogee = i

# calculates average drag coefficient for middle 40# of ascent phase
CD_avg = mean(CD[round(0.3*i_apogee):round(0.7*i_apogee)])

# (4) Free Fall Descent
# ----------------------------------------------------------------------------

while (t[i]-t[i_apogee]) < drogue_time_delay:
   
    # geopotential altitude
    h[i] = geopotential_altitude(z[i],eps,RE)
    
    # gravitational acceleration
    g[i] = gravity(g0,z[i],eps,RE)
    
    # inertial parameters of the rocket (they are now constant)
    m[i] = m[i-1]
    CG[i] = CG[i-1]
    I[i] = I[i-1]

    # stability margin
    SSM[i] = SSM[i-1]
      
    # center of pressure velocities
    v_CP_x[i] = vx[i] + D*theta_dot[i]*cos(theta[i])*SSM[i]
    v_CP_z[i] = vz[i] - D*theta_dot[i]*sin(theta[i])*SSM[i]
    
    # wind speed
    w[i] = wind(wr,zr,z[i],xi)
    
    phi[i] = arctan2(v_CP_x[i]+w[i],v_CP_z[i])
    v_inf[i] = sqrt((v_CP_x[i]+w[i])**2 + v_CP_z[i]**2)
    alpha[i] = phi[i] - theta[i]
    [rho[i],T[i],p[i],mu[i]] = atmosphere(h[i],T0,p0,R,B,g0,mu_data)
    Re[i] = rho[i]*v_inf[i]*L/mu[i]
    M_inf[i] = v_inf[i]/sqrt(gamma*R*T[i])
    
    # aerodynamic coefficients & CP
    [CA[i],CN[i],CD[i],CL[i],CP[i]] = aerodynamics(Re[i],alpha[i],cm,D,L,
                                                   L_nose,L_body,A_body,l_tail,
                                                   L_tail_c,L_tail_f,D_tail,
                                                   A_fin,A_fin_e,l_fin,t_fin,s,
                                                   cr,ct,M_inf[i],A,
                                                   CD_override,CDr)
    
    # aerodynamic forces
    FA[i] = 0.5*CA[i]*A*rho[i]*v_inf[i]**2
    FN[i] = 0.5*CN[i]*A*rho[i]*v_inf[i]**2
    
    # accelerations
    ax[i+1] = -(FA[i]/m[i])*sin(theta[i]) - (FN[i]/m[i])*cos(theta[i])
    az[i+1] = -(FA[i]/m[i])*cos(theta[i]) + (FN[i]/m[i])*sin(theta[i]) - g[i]
    a[i+1] = norm([ax[i+1],az[i+1]])
    theta_ddot[i+1] = ((FN[i]*D)/I[i])*SSM[i]

    # integration of kinematics
    x[i+1] = x[i] + vx[i]*dt
    z[i+1] = z[i] + vz[i]*dt
    vx[i+1] = vx[i] + ax[i]*dt
    vz[i+1] = vz[i] + az[i]*dt
    v[i+1] = norm([vx[i+1],vz[i+1]])
    theta[i+1] = theta[i] + theta_dot[i]*dt
    theta_dot[i+1] = theta_dot[i] + theta_ddot[i]*dt
    
    # increments time
    t[i+1] = t[i] + dt
    
    # increments loop index
    i = i+1

# stores iteration number at drogue deployment
i_drogue = i

# (5) Drogue Descent
# ----------------------------------------------------------------------------

# kinetic energy at drogue deployment
KE_drogue = 0.5*m_dry*v[i_drogue]**2

# saves original vertical drag coeff. on rocket in case drogue fails
CDr_original = CDr

# measures time since drogue deployment
t_open_d = 0

# drogue filling time
tf_d = n_d*Do_d_f/abs(v[i_drogue])

# instantaneous nominal area of drogue parachute
So_d = 0

# rocket reference area for vertical aerodynamic calculations
Ar = A
CDr = (1.42*1.41*A_fin + 0.56*A_body)/A

# trajectory integration until auxiliary deployment
while (z[i] > z_deploy_main):
    
    # geopotential altitudes
    h[i] = geopotential_altitude(z[i],eps,RE)
    hd[i] = geopotential_altitude(zd[i],eps,RE)
    
    # gravitational acceleration
    g[i] = gravity(g0,z[i],eps,RE)
   
    # flow conditions
    w[i] = wind(wr,zr,z[i],xi)
    wd[i] = wind(wr,zr,zd[i],xi)
    
    v_inf_x[i] = -((w[i]+wd[i])/2) - vx[i]
    v_inf_z[i] = -vz[i]
    [rho[i],T[i],p[i],mu[i]] = atmosphere(h[i],T0,p0,R,B,g0,mu_data)
    [rho_d[i],T[i],p[i],mu[i]] = atmosphere(hd[i],T0,p0,R,B,g0,mu_data)
    
    # drag coefficient, lateral drag coefficient, lateral lift coefficient
    # area, lateral area, and area ratio of drogue
    [CD_d[i],CD_d_side[i],CL_d_side[i],S_d[i],S_d_side[i]] = parachute_state(t_open_d,tf_d,Dp_d_f,CD_d_f,CD_side_f,CL_side_f,CD_d_type)
    
    # overrides relevant parameters if the drogue fails
    if drogue_failure:
        CD_d[i] = 0
        S_d[i] = 0
        S_d_side[i] = 0
        CDr = CD_avg
    
    # inertial parameters
    m[i] = m_dry - m_d
    
    # aerodynamic forces
    FDd[i] = 0.5*rho_d[i]*CD_d[i]*S_d[i]*v_inf_z[i]**2
    FLd[i] = 0.5*rho_d[i]*CL_d_side[i]*S_d_side[i]*v_inf_x[i]**2
    FDd_side[i] = 0.5*rho_d[i]*CD_d_side[i]*S_d_side[i]*v_inf_x[i]**2
    FDr[i] = 0.5*rho[i]*CDr*Ar*v_inf_z[i]**2
    FDr_side[i] = 0.5*rho[i]*CDr_side*Ar_side*v_inf_x[i]**2
    
    # accelerations
    ax[i+1] = (FDd_side[i]+FDr_side[i])/m[i]
    if (v_inf_x[i] < 0):
        ax[i+1] = -ax[i+1]
    az[i+1] = (FDd[i]+FLd[i]+FDr[i])/m[i]-g[i]
    a[i+1] = norm([ax[i+1],az[i+1]])
    
    # integration of kinematics
    x[i+1] = x[i] + vx[i]*dt
    z[i+1] = z[i] + vz[i]*dt
    vx[i+1] = vx[i] + ax[i]*dt
    vz[i+1] = vz[i] + az[i]*dt
    v[i+1] = norm([vx[i+1],vz[i+1]])
    theta[i+1] = theta[i] + theta_dot[i]*dt
    theta_dot[i+1] = theta_dot[i] + theta_ddot[i]*dt
    
    # drogue parachute altitude
    zd[i+1] = z[i+1] + L_shock_cord_d
    
    # increments time
    t[i+1] = t[i] + dt
    t_open_d = t_open_d+dt
    
    # increments loop index
    i = i + 1

# stores iteration number at rocket main (auxiliary) deployment
i_main = i

# (6) Main Deployment
# ----------------------------------------------------------------------------

# kinetic energies at main deployment
KE_main_1 = 0.5*(m_fore)*v[i_main]**2
KE_main_2 = 0.5*(m_mid+m_aft)*v[i_main]**2

# main parachute altitude at deployment
z_main = z[i_main]

# restores vertical rocket drag coefficient (for descent) to original value
CDr = CDr_original

# measures time starting when main begins to open
t_open_m = 0

# main filling time
tf_m = n_m*Do_m_f/abs(v[i-1])

# instantaneous nominal area of main parachute
So_m = 0

# rocket reference area for vertical aerodynamic calculations
Ar = 3*A

# stores index where main opens
i_main_open = i

# trajectory integration until payload jettison
while (z[i] > 0):

    # geopotential altitude
    h[i] = geopotential_altitude(z[i],eps,RE)
    hm[i] = geopotential_altitude(zm[i],eps,RE)
    
    # gravitational acceleration
    g[i] = gravity(g0,z[i],eps,RE)
    
    # flow conditions
    w[i] = wind(wr,zr,z[i],xi)
    wm[i] = wind(wr,zr,zm[i],xi)
    
    v_inf_x[i] = -wm[i] - vx[i]
    v_inf_z[i] = -vz[i]
    [rho[i],T[i],p[i],mu[i]] = atmosphere(h[i],T0,p0,R,B,g0,mu_data)
    [rho_m[i],T[i],p[i],mu[i]] = atmosphere(hm[i],T0,p0,R,B,g0,mu_data)
    
    # drag coefficient, lateral drag coefficient, lateral lift coefficient
    # area, lateral area, and area ratio of main
    [CD_m[i],CD_m_side[i],CL_m_side[i],S_m[i],S_m_side[i]] = parachute_state(t_open_m,tf_m,Dp_m_f,CD_m_f,CD_side_f,CL_side_f,CD_m_type)
    
    # overrides relevant parameters if the main fails
    if main_failure:
        CD_m[i] = 0
        S_m[i] = 0
        S_m_side[i] = 0
        CDr = CD_avg
    
    # inertial parameters
    m[i] = m_dry - m_d - m_m
    
    # aerodynamic forces
    FDm[i] = 0.5*rho_m[i]*CD_m[i]*S_m[i]*v_inf_z[i]**2
    FLm[i] = 0.5*rho_m[i]*CL_m_side[i]*S_m_side[i]*v_inf_x[i]**2
    FDm_side[i] = 0.5*rho_m[i]*CD_m_side[i]*S_m_side[i]*v_inf_x[i]**2
    FDr[i] = 0.5*rho[i]*CDr*Ar*v_inf_z[i]**2
    FDr_side[i] = 0.5*rho[i]*CDr_side*Ar_side*v_inf_x[i]**2
    
    # accelerations
    ax[i+1] = (FDm_side[i]+FDr_side[i])/m[i]
    if (v_inf_x[i] < 0):
        ax[i+1] = -ax[i+1]
    az[i+1] = max((FDm[i]+FLm[i]+FDr[i])/m[i]-g[i],0)
    a[i+1] = norm([ax[i+1],az[i+1]])
    
    # integration of kinematics
    x[i+1] = x[i] + vx[i]*dt
    z[i+1] = z[i] + vz[i]*dt
    vx[i+1] = vx[i] + ax[i]*dt
    vz[i+1] = vz[i] + az[i]*dt
    v[i+1] = norm([vx[i+1],vz[i+1]])
    theta[i+1] = theta[i] + theta_dot[i]*dt
    theta_dot[i+1] = theta_dot[i] + theta_ddot[i]*dt
    
    # drogue parachute altitude
    zd[i+1] = z[i+1] + L_shock_cord_d
    
    # main parachute altitude
    zm[i+1] = z[i+1] + L_shock_cord_m + L_shock_cord_d
    
    # increments time
    t[i+1] = t[i] + dt
    t_open_m = t_open_m + dt
    
    # increments loop index
    i = i + 1
    
# stores iteration number at rocket landing
i_land = i

# landing energies of rocket sections
KE_fore = 0.5*m_fore*vz[i_land]**2
KE_mid = 0.5*m_mid*vz[i_land]**2
KE_aft = 0.5*m_aft*vz[i_land]**2

# Liftoff Thrust & Thrust-to-Weight Ratio
# ----------------------------------------------------------------------------

Th_liftoff = Th[i_LRE]
Th_to_W = Th[0:i_MECO]/(m[0:i_MECO]*g[0:i_MECO])
Th_to_W_avg = round(mean(Th_to_W),1) # average thrust-to-weight ratio

# Trim Arrays & Convert Units
# ----------------------------------------------------------------------------

i_trim = i_land + 1

# time array
t = t[0:i_trim]

# rocket kinematics
a = a[0:i_trim]*mps2_to_G
ax = ax[0:i_trim]*mps2_to_G
az = az[0:i_trim]*mps2_to_G
v = v[0:i_trim]*m_to_ft
vx = vx[0:i_trim]*m_to_ft
vz = vz[0:i_trim]*m_to_ft
w = w[0:i_trim]/mph_to_mps
x = x[0:i_trim]*m_to_ft
z = z[0:i_trim]*m_to_ft
h = h[0:i_trim]*m_to_ft
theta_ddot = theta_ddot[0:i_trim]*rad_to_deg
theta_dot = theta_dot[0:i_trim]*rad_to_deg
theta = theta[0:i_trim]*rad_to_deg

# rocket inertia
m = m[0:i_trim-1]*kg_to_lbm
m_vehicle = m_vehicle*kg_to_lbm
m_dry = m_dry*kg_to_lbm
m_wet = m[0]
CG = CG[0:i_trim]*m_to_in

# rocket aerodynamics
CP = CP[0:i_trim]*m_to_in
Re = Re[0:i_trim]
M_inf = M_inf[0:i_trim]
SSM = SSM[0:i_trim]
phi = phi[0:i_trim]*rad_to_deg
alpha = alpha[0:i_trim]*rad_to_deg
CA = CA[0:i_trim]
CN = CN[0:i_trim]
CD = CD[0:i_trim]
CL = CL[0:i_trim]
v_inf = v_inf[0:i_trim]*m_to_ft
v_inf_x = v_inf_x[0:i_trim]*m_to_ft
v_inf_z = v_inf_z[0:i_trim]*m_to_ft

# rocket forces
FA = FA[0:i_trim]
FN = FN[0:i_trim]
FDr = FDr[0:i_trim]
FDr_side = FDr_side[0:i_trim]

# parachutes
Dp_d_f = Dp_d_f*m_to_in
Dp_m_f = Dp_m_f*m_to_in

# kinetic energies
KE_fore = round(KE_fore*J_to_ftlbf,1)
KE_mid = round(KE_mid*J_to_ftlbf,1)
KE_aft = round(KE_aft*J_to_ftlbf,1)
KE_drogue = round(KE_drogue*J_to_ftlbf,1)
KE_main_1 = round(KE_main_1*J_to_ftlbf,1)
KE_main_2 = round(KE_main_2*J_to_ftlbf,1)

# Flight Report
# ----------------------------------------------------------------------------

print('\n=============\n')
print('FLIGHT REPORT\n')
print('=============\n')

# apogee
apogee = round(max(z))
print(f'Apogee: {apogee} ft\n')

# time to apogee
t_apogee = round(t[i_apogee],1)
print(f'Time to Apogee: {t_apogee} s\n')

# net drift
drift = abs(round(x[-1]))
print(f'Net Drift: {drift} ft\n')

# drift from apogee
apogee_drift = abs(round(x[-1] - x[i_apogee]))
print(f'Drift From Apogee: {apogee_drift} ft\n')

# avg. thrust-to-weight ratio
print(f'Avg. Thrust-to-Weight Ratio: {Th_to_W_avg}\n');

# max. acceleration
a_max = round(max(a),1)
print(f'Max. Acceleration: {a_max} G\n')

# max. velocity
v_max = round(max(v))
print(f'Max. Velocity: {v_max} fps\n')

# static stability margin at take off
SSM_TO = round(SSM[0],2)
print(f'SSM at Take Off: {SSM_TO} cal\n')

# static stability margin at launch rail exit
SSM_LRE = round(SSM[i_LRE],2)
print(f'SSM at Rail Exit: {SSM_LRE} cal\n')

# launch rail exit velocity
v_LRE = round(v[i_LRE],1)
print(f'Velocity at Rail Exit: {v_LRE} fps\n')

# descent time (from apogee)
t_descent = round(t[i_land] - t_apogee,1)
print(f'Descent Time (from apogee): {t_descent} s\n')

# drogue descent velocity
v_descent_d = abs(round(vz[i_main],1))
print(f'Drogue Descent Velocity: {v_descent_d} fps\n')

# main descent velocity
v_descent_m = abs(round(vz[i_land],1))
print(f'Main Descent Velocity: {v_descent_m} fps\n')

# rocket section landing energies
print(f'Landing Energy - Fore Section: {KE_fore} ft-lbf\n')
print(f'Landing Energy - Middle Section: {KE_mid} ft-lbf\n')
print(f'Landing Energy - Aft Section: {KE_aft} ft-lbf\n')
