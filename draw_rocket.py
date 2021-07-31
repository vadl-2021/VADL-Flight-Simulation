# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 23:58:21 2021

@author: Ryan Burinescu

Draws rocket, CG, and CP
"""

import os
from parameters import *
import matplotlib.pyplot as plt
import numpy as np

# show stability
show_stability = True

# drawing resolution
res = 0.001

# Convert Units
# ----------------------------------------------------------------------------

# overall
D = D*m_to_in
R = D/2
L = L*m_to_in

# CG & CP
CG0 = CG0*m_to_in
CP0 = CP0*m_to_in

# nose cone
L_nose = L_nose*m_to_in

# fins
l_fin = l_fin*m_to_in                                                        
cm = cm*m_to_in                                                    
cr = cr*m_to_in                                                  
ct = ct*m_to_in 

# boat tail
L_tail = L_tail*m_to_in
L_tail_f = L_tail_f*m_to_in
L_tail_c = L_tail_c*m_to_in
L_tail_rr = L_tail_rr*m_to_in

# motor
L_motor = L_motor*m_to_in
D_motor = D_motor*m_to_in
R_motor = D_motor/2

# section lengths
L_array = L_array*m_to_in

# section diameters
D_tail = D_tail*m_to_in
D_tail_rr = D_tail_rr*m_to_in 

# Draw
# ----------------------------------------------------------------------------

plt.style.use('default')

c = 'black' # color
lw = 1.7 # line width

gold = '#DAB420'

fig1 = plt.figure(1,figsize=(16,4))

# nose cone (power series)
x_nose = np.arange(0,L_nose,res)
r_nose = R*(x_nose/L_nose)**(.5)
plt.plot(x_nose,r_nose,c,linewidth=lw)
plt.plot(x_nose,-r_nose,c,linewidth=lw)

# body sections
for i in range(1,N_sections-1):
    L_start = sum(L_array[0:i])
    draw_rectangle(L_start, L_array[i], R, res, c, lw)

# motor
L_start = L-L_motor
draw_rectangle(L_start,L_motor,R_motor,res,gold,lw)

# tail section tube
L_start = sum(L_array[0:-1])
L_tail_rect = L_array[-1] - L_tail_f - L_tail_c
draw_rectangle(L_start,L_tail_rect,R,res,c,lw)

# boat tail
R2 = D_tail/2 # aft radius
l_tail = l_tail*m_to_in

Rt = ((R2-R)**2+L_tail**2)/(2*abs(R2-R)) # radius of curvature

lt = np.arange(l_tail,l_tail+L_tail,res)
rt = R-Rt+sqrt(Rt**2-(lt-l_tail)**2)
plt.plot(lt,rt,color=c,linewidth=lw)
plt.plot(lt,-rt,color=c,linewidth=lw)

# retention ring
L_start = L-L_tail_rr
draw_rectangle(L_start,L_tail_rr,R2,res,c,lw)

# top fin
h = cm   # height 
a = ct   # top side
b = cr    # base 
# frame vertices
A = np.array([l_fin, R])
B = np.array([l_fin+b, R])
C = np.array([l_fin+0.5*(b-a)+a, R+h])
D1 = np.array([l_fin+0.5*(b-a), R+h])   
coor = np.array([A, B, C, D1])   
plt.fill(coor[:,0], coor[:,1], c)

# bottom fin
h = cm   # height 
a = ct   # top side
b = cr   # base 
# frame vertices
A = np.array([l_fin, -R])
B = np.array([l_fin+b, -R]) 
C = np.array([l_fin+0.5*(b-a)+a, -R-h])
D1 = np.array([l_fin+0.5*(b-a), -R-h])   
coor = np.array([A, B, C, D1])   
plt.fill(coor[:,0], coor[:,1], c)

if show_stability:
    # CG
    cg, = plt.plot(CG0,0,'b.',markersize=15)
    cg.set_label(f'CG: {round(CG0,1)} in')

    # CP
    cp, = plt.plot(CP0,0,'r.',markersize=15)
    cp.set_label(f'CP: {round(CP0,1)} in')

    # format legend
    plt.legend(fontsize='x-large',loc=2)
    
    plt.grid()

plt.title('VADL Launch Vehicle')
plt.xlabel('Length [in]')
plt.ylabel('Radius [in]')
plt.xlim([-5, L+5])
plt.ylim([-15, 15])
plt.axis('equal')

plt.show()

#saveas(fig1,'C:\Users\HP\Desktop\VADL\Simulations\TRAJECTORY\Fullscale\Figures\rocket_drawing.png')

# Print Info
# ----------------------------------------------------------------------------

print(f'\nStatic Stability Margin on Launch Rail: {round(abs(CP0-CG0)/D,2)}\n')
print(f'CG: {round(CG0,1)} in\n')
print(f'CP: {round(CP0,1)} in\n')
print(f'Total Length: {round(L,1)} in\n')


