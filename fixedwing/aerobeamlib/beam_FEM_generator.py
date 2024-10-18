import numpy as np
from numpy.linalg import norm

##########################################################################
# Beam():   Creates an Euler-Bernoulli beam object
#
#           Inputs:     (1) Coordinates of end points of the beam
#                       (2) Material properties of the beam
#                       (3) Discretization parameters
#
#           Function:   structural_matrices()
#           Outputs:    Assembled mass and stiffness matrices in
#                       body axis using 2-noded beam elements
##########################################################################

class Beam():
    def __init__(self, root, tip, EI, GJ, mu, I_0, num_ele, cg_x, e_m):
        self.root = root
        self.tip = tip
        self.EI = EI
        self.GJ = GJ
        self.mu = mu
        self.I_0 = I_0
        self.num_ele = num_ele
        self.cg_x = cg_x
        self.e_m = e_m
        
        self.length = norm(self.tip - self.root)/self.num_ele
        self.beam_length = norm(self.tip - self.root)
        
        self.nodes = np.zeros((self.num_ele+1, 3))
        
        for i in range(self.num_ele+1):
            self.nodes[i,0] = self.root[0] + i*(self.tip[0] - self.root[0])/self.num_ele
            self.nodes[i,1] = self.root[1] + i*(self.tip[1] - self.root[1])/self.num_ele
        
    def structural_matrices(self):
        # Coordinate transformation
        e_X = np.array([1, 0, 0])
        e_Y = np.array([0, 1, 0])
        e_Z = np.array([0, 0, 1])
        
        e_x = np.array([(self.tip[0] - self.root[0])/self.beam_length, (self.tip[1] - self.root[1])/self.beam_length, (self.tip[2] - self.root[2])/self.beam_length])
        d_2 = np.array([self.tip[0] - self.root[0], self.tip[1] - self.root[1], self.tip[2] - self.root[2]])
        #d_3 = np.array([self.ref[0] - self.root[0], self.ref[1] - self.root[1], self.ref[2] - self.root[2]])
        d_3 = np.array([0, 0, 1])
        e_y = np.cross(d_3, d_2)/norm(np.cross(d_3, d_2))
        e_z = np.cross(e_x, e_y)
        
        # DOF transformation matrix
        T_e = np.array([[np.dot(e_Z, e_z), 0, 0, 0, 0, 0],
                        [0, np.dot(e_X, e_y), np.dot(e_Y, e_y), 0, 0, 0],
                        [0, np.dot(e_X, e_x), np.dot(e_Y, e_x), 0, 0, 0],
                        [0, 0, 0, np.dot(e_Z, e_z), 0, 0],
                        [0, 0, 0, 0, np.dot(e_X, e_y), np.dot(e_Y, e_y)],
                        [0, 0, 0, 0, np.dot(e_X, e_x), np.dot(e_Y, e_x)]])
        
        # Set up beam matrices
        K_bS = np.zeros((3*(self.num_ele+1), 3*(self.num_ele+1)))
        Mss_bS = np.zeros((3*(self.num_ele+1), 3*(self.num_ele+1)))
        Msr_bS = np.zeros((3*(self.num_ele+1), 2))
        Sss_bS = np.zeros((3*(self.num_ele+1), 3*(self.num_ele+1)))
        
        # Elemental stiffness matrix in local axis
        K_eL = np.array([[12*self.EI/(self.length**3), -6*self.EI/(self.length**2), 0, -12*self.EI/(self.length**3), -6*self.EI/(self.length**2), 0],
                        [-6*self.EI/self.length**2, 4*self.EI/self.length, 0, 6*self.EI/self.length**2, 2*self.EI/self.length, 0],
                        [0, 0, self.GJ/self.length, 0, 0, -self.GJ/self.length],
                        [-12*self.EI/(self.length**3), 6*self.EI/(self.length**2), 0, 12*self.EI/(self.length**3), 6*self.EI/(self.length**2), 0],
                        [-6*self.EI/self.length**2, 2*self.EI/self.length, 0, 6*self.EI/self.length**2, 4*self.EI/self.length, 0],
                        [0, 0, -self.GJ/self.length, 0, 0, self.GJ/self.length]])
        
        # Elemental mass matrix in local axis
        Mss_eL = self.mu*self.length/420*np.array([[156, -22*self.length, 0, 54, 13*self.length, 0],
                                            [-22*self.length, 4*self.length**2, 0, -13*self.length, -3*self.length**2, 0],
                                            [0, 0, 140*self.I_0/self.mu, 0, 0, 70*self.I_0/self.mu],
                                            [54, -13*self.length, 0, 156, 22*self.length, 0],
                                            [13*self.length, -3*self.length**2, 0, 22*self.length, 4*self.length**2, 0],
                                            [0, 0, 70*self.I_0/self.mu, 0, 0, 140*self.I_0/self.mu]])
        
        # Elemental bending-torsion coupling matrix
        Sss_eL = self.mu*self.length*self.e_m/60*np.array([[0, 0, 21, 0, 0, 9],
                                [0, 0, -3*self.length, 0, 0, -2*self.length],
                                [21, -3*self.length, 0, 9, 2*self.length, 0],
                                [0, 0, 9, 0, 0, 21],
                                [0, 0, 2*self.length, 0, 0, 3*self.length],
                                [9, -2*self.length, 0, 21, 3*self.length, 0]])
        
        Mss_eL = Mss_eL + Sss_eL
        
        # Transform DOFs from local axis to body axis
        K_eS = np.matmul(np.matmul(np.transpose(T_e), K_eL), T_e)
        Mss_eS = np.matmul(np.matmul(np.transpose(T_e), Mss_eL), T_e)

        # Assemble beam matrices
        for i in range(self.num_ele):
            # Elemental flexible-rigid coupling (Dependant on nodal position)
            Msr_eL = np.array([[self.mu*self.length/2, self.mu*self.length/2*(0.7*(self.nodes[i,0] - self.cg_x) + 0.3*(self.nodes[i+1,0] - self.cg_x))],
                                   [-self.mu*self.length**2/12, -self.mu*self.length**2/4*(0.2*(self.nodes[i,0] - self.cg_x) + 2/15*(self.nodes[i+1,0] - self.cg_x))],
                                   [0, self.I_0*(np.dot(e_Y, e_x))*self.length/2],
                                   [self.mu*self.length/2, self.mu*self.length/2*(0.3*(self.nodes[i,0] - self.cg_x) + 0.7*(self.nodes[i+1,0] - self.cg_x))],
                                   [self.mu*self.length**2/12, self.mu*self.length**2/4*(2/15*(self.nodes[i,0] - self.cg_x) + 0.2*(self.nodes[i+1,0] - self.cg_x))],
                                   [0, self.I_0*(np.dot(e_Y, e_x))*self.length/2]])

            # Assemble coupling matrices
            Msr_eS = np.matmul(np.transpose(T_e), Msr_eL)
            row = i*3
            Msr_bS[row:row+Msr_eS.shape[0], 0:2] += Msr_eS
            
            # Assemble stiffness and mass(SS) matrices
            r, c = i*3, i*3
            K_bS[r:r+K_eS.shape[0], c:c+K_eS.shape[1]] += K_eS
            Mss_bS[r:r+Mss_eS.shape[0], c:c+Mss_eS.shape[1]] += Mss_eS
            
        
        # Transform DOFs from local axis to body axis
        # K_eS = np.matmul(np.matmul(np.transpose(T_e), K_eL), T_e)
        # M_eS = np.matmul(np.matmul(np.transpose(T_e), M_eL), T_e)
        
        # Assemble beam matrices
        # for element in range(self.num_ele):
        #     r, c = element*3, element*3
        #     K_bS[r:r+K_eS.shape[0], c:c+K_eS.shape[1]] += K_eS
        #     M_bS[r:r+M_eS.shape[0], c:c+M_eS.shape[1]] += M_eS
            
        
        
        return K_bS, Mss_bS, Msr_bS