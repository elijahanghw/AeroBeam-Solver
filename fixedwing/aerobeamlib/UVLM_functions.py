from time import time
import numpy as np
from numba import jit
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.linalg import eig


@jit()
def VRTXLINE(p, pt1, pt2, Gamma):
    """Computes velocity induced by a line vortex segment

    Args:
        p (ndarray): Coordinates of collocation point
        pt1 (ndarray): Coordinates of line vortex point 1
        pt2 (ndarray): Coordinates of line vortex point 2
        Gamma (float): Vortex strength of the segment

    Returns:
        float: u, v, w induced velocity components
    """    
    epsilon = 1e-50

    R1 = p-pt1
    R2 = p-pt2
    R0 = pt2-pt1
    R1xR2 = np.cross(R1,R2)
    R1xR2_norm = norm(R1xR2)
    
    r1 = norm(R1)
    r2 = norm(R2)
    
    # Rankine Vortex Core
    if R1xR2_norm < epsilon:
        K = Gamma/(4*np.pi)*(R1xR2_norm/epsilon**2)*(np.dot(R0,R1)/r1 - np.dot(R0,R2)/r2)
    
    else:
        K = Gamma/(4*np.pi*R1xR2_norm**2)*(np.dot(R0,R1)/r1 - np.dot(R0,R2)/r2)

    V = K*R1xR2

    u = V[0]
    v = V[1]
    w = V[2]

    return u, v, w


@jit()
def VRTXRING(collocation, pt1, pt2, pt3, pt4, Gamma):
    """Computes vortex ring induced velocity at a point

    Args:
        collocation (ndarray): Coordinates of collocation point
        pt1 (ndarray): Coordinates of vortex ring point 1
        pt2 (ndarray): Coordinates of vortex ring point 2
        pt3 (ndarray): Coordinates of vortex ring point 3
        pt4 (ndarray): Coordinates of vortex ring point 4
        Gamma (float): Vortex strength of the ring

    Returns:
        float: u, v, w induced velocity components
    """    
    u1, v1, w1 = VRTXLINE(collocation, pt1, pt2, Gamma)
    u2, v2, w2 = VRTXLINE(collocation, pt2, pt3, Gamma)
    u3, v3, w3 = VRTXLINE(collocation, pt3, pt4, Gamma)
    u4, v4, w4 = VRTXLINE(collocation, pt4, pt1, Gamma)

    u = u1 + u2 + u3 + u4
    v = v1 + v2 + v3 + v4
    w = w1 + w2 + w3 + w4

    return u, v, w


@jit()
def AIC(panel_vortex, wake_vortex, collocation, normal, num_chord, num_span, num_wake):
    """Computes aerodyanamic influence coefficients with symmetrical boundary conditions

    Args:
        panel_vortex (ndarray): Array of bound vortex ring corner points
        wake_vortex (ndarray): Array of wake vortex ring corner points
        collocation (ndarray): Array of collocation points
        normal (ndarray): Array of panel normal vector
        num_chord (int): Number of chord wise panels
        num_span (int): Number of span wise panels
        num_wake (int): Number of wake wise panels

    Returns:
        _type_: A and B AIC matrices
    """    
    # Compute ICM from bound vortex (A_b) (CONSTANT)
    print("Setting up bound AIC...")
    A_b = np.empty((num_chord*num_span, num_chord*num_span))
    for i in range(num_chord*num_span):
        for j in range(num_chord*num_span):
            pt1 = panel_vortex[j,0]
            pt2 = panel_vortex[j,1]
            pt3 = panel_vortex[j,2]
            pt4 = panel_vortex[j,3]
            u_i, v_i, w_i = VRTXRING(collocation[i], pt1, pt2, pt3, pt4, 1)
            sym_collocation = collocation[i].copy()
            sym_collocation[1] = -sym_collocation[1]
            u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
            influence = np.dot(np.asarray([u_i+u_ii, v_i-v_ii, w_i+w_ii]), normal[i])
            A_b[i,j] = influence
                    
    # Compute ICM from wake vortex (A_w) (CONSTANT)
    print("Setting up wake AIC...")
    A_w = np.empty((num_chord*num_span, num_wake*num_span))
    for i in range(num_chord*num_span):
        for j in range(num_wake*num_span):
            pt1 = wake_vortex[j,0]
            pt2 = wake_vortex[j,1]
            pt3 = wake_vortex[j,2]
            pt4 = wake_vortex[j,3]
            u_i, v_i, w_i = VRTXRING(collocation[i], pt1, pt2, pt3, pt4, 1) 
            sym_collocation = collocation[i].copy()
            sym_collocation[1] = -sym_collocation[1]
            u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
            influence = np.dot(np.asarray([u_i+u_ii, v_i-v_ii, w_i+w_ii]), normal[i])
            A_w[i,j] = influence

    A_ww = np.eye(num_wake*num_span)
    A_wb = np.zeros((num_wake*num_span, num_chord*num_span))
    B_b = np.zeros((num_wake*num_span, num_chord*num_span))
    B_w = np.zeros((num_wake*num_span, num_wake*num_span))

    # Setup Wake Update Matrices
    for j in range(num_span):
        for k in range(num_wake):
            for i in range(num_chord):
                #for j in range(span_panels):
                if k == 0 and i == num_chord-1:
                    B_b[k*num_span+j, i*num_span+j] = 1

    for j in range(num_span):
        for i in range(num_wake):
            for k in range(num_wake):
                if i == 0:
                    pass
                elif ((i*num_span+j) - (k*num_span+j)) == num_span:
                    B_w[i*num_span+j, k*num_span+j] = 1
                    
    # Assemble full matrices and Vectors (CONSTANT)
    A_top = np.concatenate((A_b, A_w), axis=1)
    A_bot = np.concatenate((A_wb, A_ww), axis=1)
    A = np.concatenate((A_top, A_bot), axis=0)

    B_bot = np.concatenate((B_b, B_w), axis=1)
    B = np.concatenate((np.zeros((num_chord*num_span,((num_wake+num_chord)*num_span))), B_bot), axis=0)
    
    return A, B

def AIC_parallel_Atop(input):
    aerogrid = input[0]
    collocation = input[1]
    normal = input[2]
    num_chord = input[3]
    num_span = input[4]
    num_wake = input[5]
    row_bound = np.empty(num_chord*num_span)
    for j in range(num_chord*num_span):
        pt1 = aerogrid.panel_vortex[j,0]
        pt2 = aerogrid.panel_vortex[j,1]
        pt3 = aerogrid.panel_vortex[j,2]
        pt4 = aerogrid.panel_vortex[j,3]
        u_i, v_i, w_i = VRTXRING(collocation, pt1, pt2, pt3, pt4, 1)
        sym_collocation = collocation.copy()
        sym_collocation[1] = -sym_collocation[1]
        u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
        row_bound[j] = np.dot([u_i+u_ii, v_i-v_ii, w_i+w_ii], normal)
        
    row_wake = np.empty(num_wake*num_span)
    for j in range(num_wake*num_span):
        pt1 = aerogrid.wake_vortex[j,0]
        pt2 = aerogrid.wake_vortex[j,1]
        pt3 = aerogrid.wake_vortex[j,2]
        pt4 = aerogrid.wake_vortex[j,3]
        u_i, v_i, w_i = VRTXRING(collocation, pt1, pt2, pt3, pt4, 1)
        sym_collocation = collocation.copy()
        sym_collocation[1] = -sym_collocation[1]
        u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
        row_wake[j] = np.dot([u_i+u_ii, v_i-v_ii, w_i+w_ii], normal)
        
    row = np.concatenate((row_bound, row_wake))
    return row

def AIC_parallel_AbotB(num_chord, num_span, num_wake):
    A_ww = np.eye(num_wake*num_span)
    A_wb = np.zeros((num_wake*num_span, num_chord*num_span))
    B_b = np.zeros((num_wake*num_span, num_chord*num_span))
    B_w = np.zeros((num_wake*num_span, num_wake*num_span))

    # Setup Wake Update Matrices
    for j in range(num_span):
        for k in range(num_wake):
            for i in range(num_chord):
                #for j in range(span_panels):
                if k == 0 and i == num_chord-1:
                    B_b[k*num_span+j, i*num_span+j] = 1

    for j in range(num_span):
        for i in range(num_wake):
            for k in range(num_wake):
                if i == 0:
                    pass
                elif ((i*num_span+j) - (k*num_span+j)) == num_span:
                    B_w[i*num_span+j, k*num_span+j] = 1
                    
    A_bot = np.concatenate((A_wb, A_ww), axis=1)
    
    B_bot = np.concatenate((B_b, B_w), axis=1)
    B = np.concatenate((np.zeros((num_chord*num_span,((num_wake+num_chord)*num_span))), B_bot), axis=0)
    
    return A_bot, B