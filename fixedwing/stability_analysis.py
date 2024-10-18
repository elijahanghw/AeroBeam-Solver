from time import time
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eig
import pandas as pd

from aerobeamlib.UVLM_functions import AIC
from aerobeamlib.FEM_functions import newmark_beta

# Import your aerodynamics model here
from aero_mesh import wing_aero
# Import your structural model here
from FEM_mesh import wing_structures

# Geometry properties
ref_chord = 1.8288

# Flutter Simulation Parameters #
V_start = 100
V_end = 180
V_interval = 1
AoA = 0
rho = 1.02

# Numerical Parameters
num_chord = 8
num_span = 16
wake_length = 10

ds = 1/num_chord
# ds = 0.5
num_wake = int(wake_length/ds)

alpha = 0.005

########### AERODYNAMICS ##########
aerogrid = wing_aero(num_chord, num_wake, num_span, ds)
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
femgrid = wing_structures(num_span)
print('Done')

########## AERO-STRUCTURE COUPLING ##########
print("Starting FSI analysis...")
flutter_flag = 0
flutter_speed = 0
# For exporting of eigenvalues
# df = pd.DataFrame(columns = ["v_inf", "re", "im"])
for V_inf in range(V_start,V_end+1, V_interval):
    dt = ds*ref_chord/(V_inf)

    # Newmark Beta Method (Avg Acceleration)
    N_1, N_2 = newmark_beta(femgrid.M_global, femgrid.K_global, dt)

    V_vector = np.array([V_inf, 0, V_inf*AoA])
    
    T_zeros = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
    T_wake = np.zeros((num_wake*num_span, N_1.shape[0]))
    
    # Heave downwash mapping
    T_1 = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
    
    for i in range(num_chord):
        for j in range(num_span):
            T_1[(i*num_span)+j, j*3] = 0.5
            if j != 0:  
                T_1[(i*num_span)+j, j*3-3] = 0.5
    
    T_1 = np.concatenate((T_zeros, T_1, T_zeros), axis=1)
    T_heave = np.concatenate((T_1, T_wake), axis=0)
    
    # Pitch velocity mapping
    T_2 = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
    for i in range(num_chord):
        for j in range(num_span):
            T_2[(i*num_span)+j, j*3+2] = 0.5 * (0.5*(femgrid.nodes[j, 0] + femgrid.nodes[j+1, 0]) - aerogrid.collocation[(i*num_span)+j, 0])
            if j != 0:  
                T_2[(i*num_span)+j, j*3-1] = 0.5 * (0.5*(femgrid.nodes[j, 0] + femgrid.nodes[j+1, 0]) - aerogrid.collocation[(i*num_span)+j, 0])
    T_2 = np.concatenate((T_zeros, T_2, T_zeros), axis=1)
    T_pitch = np.concatenate((T_2, T_wake), axis=0)
    
    # Pitch angle downwash mapping
    T_3 = np.zeros((num_chord*num_span, femgrid.M_global.shape[0]))
    for i in range(num_chord):
        for j in range(num_span):
            T_3[(i*num_span)+j, j*3+2] = 0.5
            #T_3[(i*num_span)+j ,-1] = 1
            if j != 0:  
                T_3[(i*num_span)+j, j*3-1] = 0.5
            
    T_3 = np.concatenate((T_3, T_zeros, T_zeros), axis=1)
    T_aoa = -V_inf * np.concatenate((T_3, T_wake), axis=0)
    
    # Full Downwash Mapping
    T = T_heave + T_pitch + T_aoa

    # Compute forces
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
    
    # Eigenvalue Problem
    Matrix_eigen = np.matmul(inv(Matrix_1), Matrix_2)
    
    w, v = eig(Matrix_eigen)

    # Convert discrete to continuous eigenvalues
    s = np.log(w)/dt       # Exact transform
    s_real = np.real(s)
    s_im = np.imag(s)

    p = s_im.argsort()
    s_im = s_im[p]
    s_real = s_real[p]
    
    speed = V_inf * np.ones_like(s_real)

    print("Velocity: " + str(V_inf) + " m/s" + "\t" + "max eigenvalue real: " + str(max(s_real)))
    
    # Export Eigenvalues as .csv file
    # df_v = pd.DataFrame({"v_inf":V_inf*np.ones_like(s_real), "re":s_real, "im":s_im})
    # df = df._append(df_v, ignore_index=True)
    # filename = "eigenvalues_full.csv"

    if flutter_flag == 0 and max(s_real) > 0:
        flutter_flag = 1
        flutter_speed = V_inf
        print("Flutter occured.")

    if V_inf == V_start:
        plt.scatter(s_real, s_im, color='r', marker='x')
    elif V_inf == V_end:
        plt.scatter(s_real, s_im, color='b', marker='x')
    else:
        plt.scatter(s_real, s_im, color='k', marker='x')
    
plt.xlabel("Real(s)")
plt.ylabel("Im(s)")
plt.axvline(0, color='grey', linestyle="--")

plt.xlim([-40, 10])
plt.ylim([0, 120])
plt.grid()


print("Plotting eigenvalues")
if flutter_flag == 1:
    print("Flutter speed: " + str(flutter_speed) + "m/s")
elif flutter_flag == 0:
    print("No flutter.")
print("Simulation complete.")
# df.to_csv(filename, index=False)
plt.show()