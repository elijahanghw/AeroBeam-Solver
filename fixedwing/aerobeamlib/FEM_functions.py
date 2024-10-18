from re import I
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eig
import pandas as pd

def modal_reduction(M, K, num_modes, RB = 0):
    w, v = eig(K, M)
    
    omega = np.sqrt(w)
    p = omega.argsort()
    omega = omega[p]
    v = v[:, p]
    w_dia = np.diag(omega**2)
    
    # Mode truncation
    w_truncatated = omega[0:num_modes]**2
    v_truncated = v[:,0:num_modes]
    
    if RB == 1:   
        v_0 = v_truncated[:,0]
        v_1 = v_truncated[:,1]
        
        v_0_new = v_0/v_0[-1] - v_1/v_1[-1]
        v_1_new = v_0/v_0[-2] - v_1/v_1[-2]
        
        v_truncated[:,0] = v_0_new
        v_truncated[:,1] = v_1_new
        
    
    
    # Mass normalize eigenvectors
    for i in range(num_modes):
        NF = np.matmul(np.transpose(v_truncated[:,i]), M)
        NF = np.matmul(NF, v_truncated[:,i])
        NF = np.sqrt(NF)
        v_truncated[:,i] = v_truncated[:,i]/NF

    M_truncated = np.eye(num_modes)
    K_truncated = w_dia[0:num_modes, 0:num_modes]
    
    return M_truncated, K_truncated, w_truncatated, v_truncated

def newmark_beta(M, K, dt, alpha=0.005):
    phi = 0.5 + alpha
    beta = 0.25*(phi+0.5)**2

    zero_matrix = np.zeros_like(M)
    identity = np.eye(M.shape[0])
    
    S_1 = M + dt**2*beta*K
    S_2 = dt**2*(0.5-beta)*K

    N_11 = np.concatenate((zero_matrix, zero_matrix, S_1), axis=1)
    N_12 = np.concatenate((zero_matrix, -identity, dt*phi*identity), axis=1)
    N_13 = np.concatenate((-identity, zero_matrix, dt**2*beta*identity), axis=1)

    N_1 = np.concatenate((N_11, N_12, N_13), axis = 0)

    N_21 = np.concatenate((K, K*dt, S_2), axis=1)
    N_22 = np.concatenate((zero_matrix, identity, (1-phi)*dt*identity), axis=1)
    N_23 = np.concatenate((identity, dt*identity, (0.5-beta)*dt**2*identity), axis=1)

    N_2 = np.concatenate((N_21, N_22, N_23), axis = 0)
    
    return N_1, N_2