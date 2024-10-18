import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

##########################################################################
# Segment():    Creates a quadralateral aerodynamic surface object
#
#               Inputs:     (1) Chord length of root and tip
#                           (2) Coordinates of leading edge of root and tip
#                           (3) Discretization parameters
#
#               Function:   bound_vortex_ring()
#               Outputs:    Coordinates of vertices of bound vortex rings
#                           Coordinates of collocation points
#                           Coordinates of quarterchord
#                           Normal and tangential vectors
#                           Panel chord and span
#
#               Function:   wake_vortex_ring()
#               Outputs:    Coordinates of vertices of wake vortex rings
##########################################################################

class Segment():
    def __init__(self, root_chord, root_LE, tip_chord, tip_LE, segment_span, segment_root, num_span, num_chord, num_wake, ref_chord, ds):
        self.root_chord = root_chord
        self.root_LE = root_LE
        self.tip_chord = tip_chord
        self.tip_LE = tip_LE
        self.segment_span = segment_span
        self.segment_root = segment_root
        self.num_span = num_span
        self.num_chord = num_chord
        self.num_wake = num_wake
        self.ref_chord = ref_chord
        self.ds = ds
       
        self.geometry_x = np.zeros((self.num_chord+1, self.num_span+1))
        self.geometry_y = np.zeros((self.num_chord+1, self.num_span+1))
        self.bound_vortex_x = np.zeros((self.num_chord+1, self.num_span+1))
        self.bound_vortex_y = np.zeros((self.num_chord+1, self.num_span+1))
        
        self.wake_vortex_x = np.zeros((num_wake+1, num_span+1))
        self.wake_vortex_y = np.zeros((num_wake+1, num_span+1))
        
        for i in range(self.num_chord+1):
            for j in range(self.num_span+1):
                self.geometry_y[i,j] = j*self.segment_span/self.num_span + self.segment_root
                self.geometry_x[i,j] = (j*self.tip_LE + (self.num_span-j)*self.root_LE)/(self.num_span) + i/self.num_chord*(j*self.tip_chord + (self.num_span-j)*self.root_chord)/(self.num_span)
                self.bound_vortex_y[i,j] = j*self.segment_span/self.num_span + self.segment_root
                if i == self.num_chord:
                    self.bound_vortex_x[i,j] = (j*self.tip_LE + (self.num_span-j)*self.root_LE)/(self.num_span) + (i)/self.num_chord*(j*self.tip_chord + (self.num_span-j)*self.root_chord)/(self.num_span) + 0.3*self.ref_chord*self.ds
                else:
                    self.bound_vortex_x[i,j] = (j*self.tip_LE + (self.num_span-j)*self.root_LE)/(self.num_span) + (i+0.25)/self.num_chord*(j*self.tip_chord + (self.num_span-j)*self.root_chord)/(self.num_span)
                # self.bound_vortex_x[i,j] = (j*self.tip_LE + (self.num_span-j)*self.root_LE)/(self.num_span) + (i+0.25)/self.num_chord*(j*self.tip_chord + (self.num_span-j)*self.root_chord)/(self.num_span)
        for k in range(self.num_wake+1):
            for j in range(self.num_span+1):
                self.wake_vortex_y[k,j] = j*self.segment_span/self.num_span + self.segment_root
                self.wake_vortex_x[k,j] = (j*self.tip_LE + (self.num_span-j)*self.root_LE)/(self.num_span) + (j*self.tip_chord + (self.num_span-j)*self.root_chord)/(self.num_span) + (k+0.3)*self.ref_chord*self.ds
                # self.wake_vortex_x[k,j] = (j*self.tip_LE + (self.num_span-j)*self.root_LE)/(self.num_span) + (1+0.25/self.num_chord)*(j*self.tip_chord + (self.num_span-j)*self.root_chord)/(self.num_span) + (k)*self.ref_chord*self.ds
                
    def bound_vortex_ring(self):
        panel_vortex = []
        collocation = []
        normal = []
        quarterchord = []
        tau_i = []
        tau_j = []
        delta_c = []
        delta_b = []
        
        for i in range(self.num_chord):
            for j in range(self.num_span):
                # Computation of vortex ring points for panel i,j
                pt1 = [self.bound_vortex_x[i,j], self.bound_vortex_y[i,j], 0]
                pt2 = [self.bound_vortex_x[i,j+1], self.bound_vortex_y[i,j+1], 0]
                pt3 = [self.bound_vortex_x[i+1,j+1], self.bound_vortex_y[i+1,j+1], 0]
                pt4 = [self.bound_vortex_x[i+1,j], self.bound_vortex_y[i+1,j], 0]

                panel_vortex.append([pt1, pt2, pt3, pt4])

                # Computation of collocation point
                collocation_pt = [((1/4*self.geometry_x[i,j] + 3/4*self.geometry_x[i+1,j]) + (1/4*self.geometry_x[i,j+1] + 3/4*self.geometry_x[i+1,j+1]))/2, (self.geometry_y[i,j] + self.geometry_y[i,j+1])/2 , 0]
                collocation.append(collocation_pt)

                # Computation of normal vector
                A_vector = np.array(pt3) - np.array(pt1)
                B_vector = np.array(pt2) - np.array(pt4)
                AxB = np.cross(A_vector,B_vector)
                normal.append(AxB/norm(AxB))

                # Computation of quarterchord
                quarterchord.append([(self.bound_vortex_x[i,j] + self.bound_vortex_x[i,j+1])/2, (self.bound_vortex_y[i,j] + self.bound_vortex_y[i,j+1])/2 , 0])
                
                # Computation of tangential vectors
                tau_i.append((np.array(pt4) - np.array(pt1))/norm(np.array(pt4) - np.array(pt1)))
                tau_j.append((np.array(pt2) - np.array(pt1))/norm(np.array(pt2) - np.array(pt1)))
                
                # Computation of panel chord and span
                # delta_c.append(norm(np.array(pt4) - np.array(pt1)))
                # delta_b.append(norm(np.array(pt2) - np.array(pt1)))
                
                p1 = [self.geometry_x[i,j], self.geometry_y[i,j], 0]
                p2 = [self.geometry_x[i,j+1], self.geometry_y[i,j+1], 0]
                p3 = [self.geometry_x[i+1,j+1], self.geometry_y[i+1,j+1], 0]
                p4 = [self.geometry_x[i+1,j], self.geometry_y[i+1,j], 0]
                
                delta_c.append(norm(np.array(p4) - np.array(p1)))
                delta_b.append(norm(np.array(p2) - np.array(p1)))
                

        panel_vortex = np.array(panel_vortex)
        panel_vortex = np.reshape(panel_vortex, (self.num_chord,self.num_span,4,3))

        collocation = np.array(collocation)
        collocation = np.reshape(collocation, (self.num_chord,self.num_span,3))

        normal = np.array(normal)
        normal = np.reshape(normal, (self.num_chord,self.num_span,3))

        quarterchord = np.array(quarterchord)
        quarterchord = np.reshape(quarterchord, (self.num_chord,self.num_span,3))    
        
        tau_i = np.array(tau_i)
        tau_i = np.reshape(tau_i, (self.num_chord,self.num_span,3))   
        
        tau_j = np.array(tau_j)
        tau_j = np.reshape(tau_j, (self.num_chord,self.num_span,3))
        
        delta_c = np.array(delta_c)
        delta_c = np.reshape(delta_c, (self.num_chord,self.num_span))
        
        delta_b = np.array(delta_b)
        delta_b = np.reshape(delta_b, (self.num_chord,self.num_span))
        
        return panel_vortex, collocation, normal, quarterchord, tau_i, tau_j, delta_c, delta_b
    
    def wake_vortex_ring(self):
        wake_vortex = []
        
        for k in range(self.num_wake):
            for j in range(self.num_span):
                # Computation of vortex ring points for wake k,j
                pt1 = [self.wake_vortex_x[k,j], self.wake_vortex_y[k,j], 0]
                pt2 = [self.wake_vortex_x[k,j+1], self.wake_vortex_y[k,j+1], 0]
                pt3 = [self.wake_vortex_x[k+1,j+1], self.wake_vortex_y[k+1,j+1], 0]
                pt4 = [self.wake_vortex_x[k+1,j], self.wake_vortex_y[k+1,j], 0]

                wake_vortex.append([pt1, pt2, pt3, pt4])

        wake_vortex = np.array(wake_vortex)
        wake_vortex = np.reshape(wake_vortex, (self.num_wake,self.num_span,4,3))
        
        return wake_vortex
