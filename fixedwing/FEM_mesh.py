import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.linalg import eig

from aerobeamlib.beam_FEM_generator import Beam

class wing_structures():
    def __init__(self, num_ele):
        # C.G. Properties
        M = 217.688
        I_alpha = 52.66944
        cg_x = 0.0
        self.cg_x = cg_x
        self.Mrr = np.array([[M, 0],
                             [0, I_alpha]])
                
        # Segment 1
        root = np.array([0.603504, 0, 0])
        tip = np.array([0.603504, 6.096, 0])

        EI = 9.77e6
        GJ = 0.99e6
        mu = 35.71
        I_0 = 8.64
        e_m = -0.18288

        beam = Beam(root, tip, EI, GJ, mu, I_0, num_ele, cg_x, e_m)
        K_bS, Mss_bS, Msr_bS = beam.structural_matrices()

        # Assemble Matrix
        self.Kss_global = np.zeros((3*(num_ele+1), 3*(num_ele+1)))
        self.Mss_global = np.zeros((3*(num_ele+1), 3*(num_ele+1)))
        self.Msr_global = np.zeros((3*(num_ele+1), 2))

        for i in range(3*(num_ele+1)):
            self.Msr_global[i,0] += Msr_bS[i,0]
            self.Msr_global[i,1] += Msr_bS[i,1]
            for j in range(3*(num_ele+1)):
                self.Kss_global[i,j] += K_bS[i,j]
                self.Mss_global[i,j] += Mss_bS[i,j]

        self.M_global = np.concatenate((self.Mss_global, self.Msr_global), axis=1)
        self.M_global = np.concatenate((self.M_global, np.concatenate((np.transpose(self.Msr_global), self.Mrr), axis=1)), axis=0)
        
        self.K_global = np.concatenate((self.Kss_global, np.zeros_like(self.Msr_global)), axis=1)
        self.K_global = np.concatenate((self.K_global, np.concatenate((np.transpose(np.zeros_like(self.Msr_global)), np.zeros_like(self.Mrr)), axis=1)), axis=0)
        
        # Sym BC
        # self.Kss_global = np.delete(self.Kss_global, 1, 0)
        # self.Kss_global = np.delete(self.Kss_global, 1, 1)
        # self.Mss_global = np.delete(self.Mss_global, 1, 0)
        # self.Mss_global = np.delete(self.Mss_global, 1, 1)
        
        # self.K_global = np.delete(self.K_global, 1, 0)
        # self.K_global = np.delete(self.K_global, 1, 1)
        # self.M_global = np.delete(self.M_global, 1, 0)
        # self.M_global = np.delete(self.M_global, 1, 1)
        
        # Fully Clamped BC
        self.Kss_global = self.Kss_global[3:, 3:]
        self.Mss_global = self.Mss_global[3:, 3:]
        self.K_global = self.K_global[3:-2, 3:-2]
        self.M_global = self.M_global[3:-2, 3:-2]
        
        # Coordinates of nodes
        self.nodes = beam.nodes
        # Length of elements
        self.ele_len = np.zeros(self.nodes.shape[0]-1)
        for element in range(self.nodes.shape[0] -1):
            self.ele_len[element] = norm(self.nodes[element+1] - self.nodes[element])