import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from aerobeamlib.aerodynamics_mesh_generator import Segment

class wing_aero():
    def __init__(self, num_chord, num_wake, num_span, ds):    
        # Simulation properties
        self.ref_chord = 1.8288
        # Segment 1
        root_chord = 1.8288
        root_LE = 0
        tip_chord = 1.8288
        tip_LE = 0
        segment_span = 6.096
        segment_root = 0

        segment = Segment(root_chord, root_LE, tip_chord, tip_LE, segment_span, segment_root, num_span, num_chord, num_wake, self.ref_chord, ds)

        panel_vortex, collocation, normal, quarterchord, tau_i, tau_j, delta_c, delta_b = segment.bound_vortex_ring()
        wake_vortex = segment.wake_vortex_ring()

        self.goland_geometry_x = segment.geometry_x
        self.goland_geometry_y = segment.geometry_y
        self.goland_bound_vortex_x = segment.bound_vortex_x
        self.goland_bound_vortex_y = segment.bound_vortex_y
        self.goland_wake_vortex_x = segment.wake_vortex_x
        self.goland_wake_vortex_y = segment.wake_vortex_y

        self.panel_vortex = np.reshape(panel_vortex, (num_chord*num_span, 4, 3))
        self.collocation = np.reshape(collocation, (num_chord*num_span, 3))
        self.normal = np.reshape(normal, (num_chord*num_span, 3))
        self.quarterchord = np.reshape(quarterchord, (num_chord*num_span, 3))
        self.tau_i = np.reshape(tau_i, (num_chord*num_span, 3))
        self.tau_j = np.reshape(tau_j, (num_chord*num_span, 3))
        self.delta_c = np.reshape(delta_c, (num_chord*num_span))
        self.delta_b = np.reshape(delta_b, (num_chord*num_span))
        self.wake_vortex = np.reshape(wake_vortex, (num_wake*num_span, 4, 3))

    def plot_grid(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')

        for i in range(self.goland_geometry_x.shape[0]):
            plt.plot(self.goland_geometry_x[i,:], self.goland_geometry_y[i,:], color='k')
            plt.plot(self.goland_bound_vortex_x[i,:], self.goland_bound_vortex_y[i,:], color='r')
            
        for j in range(self.goland_geometry_x.shape[1]):
            plt.plot(self.goland_geometry_x[:,j], self.goland_geometry_y[:,j], color='k')
            plt.plot(self.goland_bound_vortex_x[:,j], self.goland_bound_vortex_y[:,j], color='r')
            
        for i in range(self.goland_wake_vortex_x.shape[0]):
            plt.plot(self.goland_wake_vortex_x[i,:], self.goland_wake_vortex_y[i,:], color='b')
            
        for j in range(self.goland_wake_vortex_x.shape[1]):
            plt.plot(self.goland_wake_vortex_x[:,j], self.goland_wake_vortex_y[:,j], color='b')

        for panel in self.collocation:
            plt.scatter(panel[0], panel[1], color='r', marker='x')
                
        plt.xlabel('x')
        plt.ylabel('y')         
            
        plt.show()

    

        
        
