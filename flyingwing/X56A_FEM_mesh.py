import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.linalg import eig

from aerobeamlib.beam_FEM_generator import Beam

class X56A_structures():
    def __init__(self, num_ele1, num_ele2, num_ele3):
        # C.G. Properties
        M = 0.08
        I_alpha = 0.0012
        cg_x = 0.172
        self.cg_x = cg_x
        self.Mrr = np.array([[M, 0],
                             [0, I_alpha]])
                
        # Segment 1
        root1 = np.array([0.17735, 0, 0])
        tip1 = np.array([0.17735, 0.0607, 0])

        EI1 = 10000
        GJ1 = 10000
        mu1 =  0.01
        I_01 = 0.001
        e_m1 = 0

        beam_1 = Beam(root1, tip1, EI1, GJ1, mu1, I_01, num_ele1, cg_x, e_m1)
        K_bS1, Mss_bS1, Msr_bS1 = beam_1.structural_matrices()

        # Segment 2
        root2 = np.array([0.17735, 0.0607, 0])
        tip2 = np.array([0.17735, 0.1214, 0])

        EI2 = 10000
        GJ2 = 10000
        mu2 = 0.01
        I_02 = 0.001
        e_m2 = 0

        beam_2 = Beam(root2, tip2, EI2, GJ2, mu2, I_02, num_ele2, cg_x, e_m2)
        K_bS2, Mss_bS2, Msr_bS2 = beam_2.structural_matrices()

        # Segment 3
        root3 = np.array([0.17735, 0.1214, 0])
        tip3 = np.array([0.31515, 0.5, 0])
        
        EI3 = 0.3024
        GJ3 = 0.072
        mu3 = 0.069
        I_03 = 0.0012
        e_m3 = 0
        pt_mass = 0*0.00235

        beam_3 = Beam(root3, tip3, EI3, GJ3, mu3, I_03, num_ele3, cg_x, e_m3)
        K_bS3, Mss_bS3, Msr_bS3  = beam_3.structural_matrices()

        # Assemble Matrix
        self.Kss_global = np.zeros((3*(num_ele1+num_ele2+num_ele3+1), 3*(num_ele1+num_ele2+num_ele3+1)))
        self.Mss_global = np.zeros((3*(num_ele1+num_ele2+num_ele3+1), 3*(num_ele1+num_ele2+num_ele3+1)))
        self.Msr_global = np.zeros((3*(num_ele1+num_ele2+num_ele3+1), 2))

        for i in range(3*(num_ele1+1)):
            self.Msr_global[i,0] += Msr_bS1[i,0]
            self.Msr_global[i,1] += Msr_bS1[i,1]
            for j in range(3*(num_ele1+1)):
                self.Kss_global[i,j] += K_bS1[i,j]
                self.Mss_global[i,j] += Mss_bS1[i,j]
                
        for i in range(3*(num_ele2+1)):
            self.Msr_global[i+3*num_ele1,0] += Msr_bS2[i,0]
            self.Msr_global[i+3*num_ele1,1] += Msr_bS2[i,1]
            for j in range(3*(num_ele2+1)):       
                self.Kss_global[i+3*num_ele1,j+3*num_ele1] += K_bS2[i,j]
                self.Mss_global[i+3*num_ele1,j+3*num_ele1] += Mss_bS2[i,j]
                
        for i in range(3*(num_ele3+1)):
            self.Msr_global[i+3*(num_ele1+num_ele2),0] += Msr_bS3[i,0]
            self.Msr_global[i+3*(num_ele1+num_ele2),1] += Msr_bS3[i,1]
            for j in range(3*(num_ele3+1)):
                self.Kss_global[i+3*(num_ele1+num_ele2),j+3*(num_ele1+num_ele2)] += K_bS3[i,j]
                self.Mss_global[i+3*(num_ele1+num_ele2),j+3*(num_ele1+num_ele2)] += Mss_bS3[i,j]

        self.M_global = np.concatenate((self.Mss_global, self.Msr_global), axis=1)
        self.M_global = np.concatenate((self.M_global, np.concatenate((np.transpose(self.Msr_global), self.Mrr), axis=1)), axis=0)
        
        self.K_global = np.concatenate((self.Kss_global, np.zeros_like(self.Msr_global)), axis=1)
        self.K_global = np.concatenate((self.K_global, np.concatenate((np.transpose(np.zeros_like(self.Msr_global)), np.zeros_like(self.Mrr)), axis=1)), axis=0)
        
        # BFF Sym BC
        self.Kss_global = self.Kss_global[3:, 3:]
        self.Mss_global = self.Mss_global[3:, 3:]
        self.K_global = self.K_global[3:, 3:]
        self.M_global = self.M_global[3:, 3:]
        
        # Coordinates of nodes
        self.nodes = np.concatenate((beam_1.nodes[:-1], beam_2.nodes[:-1], beam_3.nodes))
        
        # Length of elements
        self.ele_len = np.zeros(self.nodes.shape[0]-1)
        for element in range(self.nodes.shape[0] -1):
            self.ele_len[element] = norm(self.nodes[element+1] - self.nodes[element])