from time import time
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eig
import pandas as pd

from aerobeamlib.UVLM_functions import AIC
from aerobeamlib.FEM_functions import newmark_beta
from aerobeamlib.gust_generator import discrete_gust

# Import your aerodynamics model here
from X56A_aero_mesh import X56A_aero
# Import your structural model here
from X56A_FEM_mesh import X56A_structures

# Geometry properties
ref_chord = 0.2743

# Time Stepping Simulation Parameters
V_inf = 10
AoA = 0
rho = 1.225
final_time = 10
    
# Numerical Parameters
num_chord = 8
num_span1 = 4
num_span2 = 4
num_span3 = 18
wake_length = 10
num_span = num_span1 + num_span2 + num_span3
num_dof = 3*num_span + 2
ds = 1/num_chord
num_wake = int(wake_length/ds)
dt = ds*ref_chord/(V_inf)
timesteps = int(final_time/dt)


start = time()
########### AERODYNAMICS ##########
aerogrid = X56A_aero(num_chord, num_wake, num_span1, num_span2, num_span3, ds)
aerogrid.plot_grid()
print(f"Number of bound panels: {num_chord* num_span}")
print(f"Number of wake panels: {num_wake * num_span}")
print(f"Total number of panels: {(num_wake + num_chord) * num_span}")
print("Using Numba...")
print('Setting up AIC matrices...')
# Parallel
start = time()
A, B = AIC(aerogrid.panel_vortex, aerogrid.wake_vortex, aerogrid.collocation, aerogrid.normal, num_chord, num_span, num_wake)
end = time()  
print(f"AIC Setup Time = {end-start}")
print('Done')

########### STRUCTURAL DYNAMICS ##########
print("Setting up structural matrices...")
femgrid = X56A_structures(num_span1, num_span2, num_span3)
print('Done')

########## AERO-STRUCTURE COUPLING ##########
print("Performing FSI coupling...")

# Newmark Beta Method (Avg Acceleration)
N_1, N_2 = newmark_beta(femgrid.M_global, femgrid.K_global, dt, alpha=0.01)

V_vector = np.array([V_inf, 0, V_inf*AoA])

# Downwash mapping
T_zeros = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
T_wake = np.zeros((num_wake*num_span, N_1.shape[0]))

# Heave downwash mapping
T_1 = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))

for i in range(num_chord):
    for j in range(num_span):
        T_1[(i*num_span)+j, j*3] = 0.5
        T_1[(i*num_span)+j ,-2] = 1 
        if j != 0:  
            T_1[(i*num_span)+j, j*3-3] = 0.5
        
T_1 = np.concatenate((T_zeros, T_1, T_zeros), axis=1)
T_heave = np.concatenate((T_1, T_wake), axis=0)

# Pitch velocity mapping
T_2 = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
for i in range(num_chord):
    for j in range(num_span):
        T_2[(i*num_span)+j, j*3+2] = 0.5 * (0.5*(femgrid.nodes[j, 0] + femgrid.nodes[j+1, 0]) - aerogrid.collocation[(i*num_span)+j, 0])
        T_2[(i*num_span)+j ,-1] = femgrid.cg_x - aerogrid.collocation[(i*num_span)+j, 0]
        if j != 0:  
            T_2[(i*num_span)+j, j*3-1] = 0.5 * (0.5*(femgrid.nodes[j, 0] + femgrid.nodes[j+1, 0]) - aerogrid.collocation[(i*num_span)+j, 0])

T_2 = np.concatenate((T_zeros, T_2, T_zeros), axis=1)
T_pitch = np.concatenate((T_2, T_wake), axis=0)

# Pitch angle downwash mapping
T_3 = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
for i in range(num_chord):
    for j in range(num_span):
        T_3[(i*num_span)+j, j*3+2] = 0.5
        T_3[(i*num_span)+j ,-1] = 1
        if j != 0:  
            T_3[(i*num_span)+j, j*3-1] = 0.5
            
T_3 = np.concatenate((T_3, T_zeros, T_zeros), axis=1)
T_aoa = -V_inf * np.concatenate((T_3, T_wake), axis=0)

# Full Downwash Mapping
T = T_heave + T_pitch + T_aoa

# Force Mapping
Phi_new = np.zeros((num_span*num_chord, num_span*num_chord))
Phi_old = np.zeros((num_span*num_chord, num_span*num_chord))

for i in range(num_chord*num_span):
    Phi_new[i,i] = (rho*np.dot(V_vector, aerogrid.tau_i[i])/aerogrid.delta_c[i] + rho/dt + rho*np.dot(V_vector, aerogrid.tau_j[i])/aerogrid.delta_b[i])*aerogrid.delta_c[i]*aerogrid.delta_b[i]
    Phi_old[i,i] = (-rho/dt)*aerogrid.delta_c[i]*aerogrid.delta_b[i]
    if i%num_span != 0:
        Phi_new[i, i-1] = (-rho*np.dot(V_vector, aerogrid.tau_j[i])/aerogrid.delta_b[i])*aerogrid.delta_c[i]*aerogrid.delta_b[i]
    if i-num_span >=0:
        Phi_new[i, i-num_span] = (-rho*np.dot(V_vector, aerogrid.tau_i[i])/aerogrid.delta_c[i])*aerogrid.delta_c[i]*aerogrid.delta_b[i]
        
force_mapping = np.zeros((femgrid.M_global.shape[0], num_span*num_chord))

for i in range(num_span):
    for j in range(num_chord):
        force_mapping[3*i,i+j*num_span] = 0.5
        force_mapping[3*i+1,i+j*num_span] = -femgrid.ele_len[i]/8
        force_mapping[3*i+2,i+j*num_span] = 0.5*(femgrid.nodes[i+1, 0] - aerogrid.quarterchord[i+j*num_span, 0])
        
        force_mapping[-2, i+j*num_span] = 1
        force_mapping[-1, i+j*num_span] = femgrid.cg_x - aerogrid.quarterchord[i+j*num_span, 0]
        if i < num_span-1:
            force_mapping[3*i,i+j*num_span+1] = 0.5
            force_mapping[3*i+1,i+j*num_span+1] = femgrid.ele_len[i]/8
            force_mapping[3*i+2,i+j*num_span+1] = 0.5*(femgrid.nodes[i+1, 0] - aerogrid.quarterchord[i+j*num_span+1, 0])
            
Psi_1 = np.matmul(force_mapping, Phi_new)
Psi_2 = np.matmul(force_mapping, Phi_old)
    
Psi_1 = np.concatenate((Psi_1, np.zeros((femgrid.M_global.shape[0], num_span*num_wake))),axis=1)
Psi_2 = np.concatenate((Psi_2, np.zeros((femgrid.M_global.shape[0], num_span*num_wake))),axis=1)

Psi_1 = np.concatenate((Psi_1, np.zeros((2*femgrid.M_global.shape[0], (num_chord+num_wake)*num_span))), axis=0)
Psi_2 = np.concatenate((Psi_2, np.zeros((2*femgrid.M_global.shape[0], (num_chord+num_wake)*num_span))), axis=0)

Matrix_11 = np.concatenate((A, -T), axis=1)
Matrix_12 = np.concatenate((-Psi_1, N_1), axis=1)
Matrix_1 = np.concatenate((Matrix_11, Matrix_12), axis=0)

Matrix_21 = np.concatenate((B, np.zeros(((num_chord+num_wake)*num_span, 3*femgrid.M_global.shape[0]))), axis=1)
Matrix_22 = np.concatenate((Psi_2, -N_2), axis=1)
Matrix_2 = np.concatenate((Matrix_21, Matrix_22), axis=0)

########## DISCRETE GUST ##########
print("Generating discrete gust...")

H = 10   # Gust penetration (half of gust profile)
upwash = discrete_gust(V_inf, aerogrid.collocation, dt, timesteps, H)
downwash = -upwash
downwash = np.concatenate((downwash, np.zeros((num_wake*num_span + 3*num_dof, timesteps))))
Disturbance = np.matmul(inv(Matrix_1), downwash)


########## STATE SPACE FORMULATION ##########
A_sys = np.matmul(inv(Matrix_1), Matrix_2)
B_sys = -V_inf*np.concatenate((aerogrid.cs1, aerogrid.cs2, aerogrid.cs3), axis=1)
B_sys = np.concatenate((B_sys, np.zeros((num_span*num_wake + 3*num_dof,3))), axis=0)
B_sys = np.matmul(inv(Matrix_1), B_sys)

C_1 = np.zeros((3,(num_chord+num_wake)*num_span))
C_2 = np.zeros((3, num_dof))
C_zeros = np.zeros((3, num_dof))
C_2[0,-2] = 1 # Fuselage diaplacement
C_2[1,-1] = 1 # Fuselage pitch
C_2[2,-5] = 1 # Tip displacement

C_sys = np.concatenate((C_1, C_2, C_zeros, C_zeros), axis=1)

D_sys = np.zeros((3,3))


########## TIME STEPPING ##########
print("Starting simulation...")
x_old = np.zeros((num_chord+num_wake)*num_span + 3*num_dof)
U = np.array([0, 0, 0.0])
y = np.zeros(3)
sim_time = []
fuselage_disp = []    
fuselage_pitch = []  
rel_tip_disp = []
CS1 = []
CS2 = []
CS3 = []  
for t in range(timesteps):
    sim_time.append(t*dt)
    print(f"Timestep: {t}")
    x_new = np.matmul(A_sys, x_old)  + Disturbance[:,t]
    y = np.matmul(C_sys, x_new)
    fuselage_disp.append(y[0])
    fuselage_pitch.append(y[1]*180/np.pi)
    rel_tip_disp.append(y[2]/0.5*100)
    CS1.append(U[0]*180/np.pi)
    CS2.append(U[1]*180/np.pi)
    CS3.append(U[2]*180/np.pi)
    x_old = x_new

end = time()

print(f"Total runtime = {end-start}")

fig1 = plt.figure()
ax1 = fig1.subplots()
ax1.plot(sim_time, fuselage_disp)
ax1.set_xlabel('time, s')
ax1.set_ylabel('C.G. displacement')
plt.grid()

fig2 = plt.figure()
ax2 = fig2.subplots()
ax2.plot(sim_time, fuselage_pitch)
ax2.set_xlabel('time, s')
ax2.set_ylabel('C.G. pitch')
plt.grid()

fig3 = plt.figure()
ax3 = fig3.subplots()
ax3.plot(sim_time, rel_tip_disp)
ax3.set_xlabel('time, s')
ax3.set_ylabel('relative tip displacement')
plt.grid()


plt.show()
    