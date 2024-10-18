import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from aerobeamlib.aerodynamics_mesh_generator import Segment

class X56A_aero():
    def __init__(self, num_chord, num_wake, num_span1, num_span2, num_span3, ds):    
        # Simulation properties
        self.ref_chord = 0.2743
        num_span = num_span1 + num_span2 + num_span3
        
        # Segment 1
        root_chord1 = 0.2743
        root_LE1 = 0
        tip_chord1 = 0.1446
        tip_LE1 = 0.1052
        segment_span1 = 0.0607
        segment_root1 = 0

        segment_1 = Segment(root_chord1, root_LE1, tip_chord1, tip_LE1, segment_span1, segment_root1, num_span1, num_chord, num_wake, self.ref_chord, ds)

        # Segment 2
        root_chord2 = 0.1446
        root_LE2 = 0.1052
        tip_chord2 = 0.0957
        tip_LE2 = 0.1295
        segment_span2 = 0.0607
        segment_root2 = 0.0607

        segment_2 = Segment(root_chord2, root_LE2, tip_chord2, tip_LE2, segment_span2, segment_root2, num_span2, num_chord, num_wake, self.ref_chord, ds)

        # Segment 3
        root_chord3 = 0.0957
        root_LE3 = 0.1295
        tip_chord3 = 0.0957
        tip_LE3 = 0.2673
        segment_span3 = 0.3786
        segment_root3 = 0.1214

        segment_3 = Segment(root_chord3, root_LE3, tip_chord3, tip_LE3, segment_span3, segment_root3, num_span3, num_chord, num_wake, self.ref_chord, ds)

        panel_vortex1, collocation1, normal1, quarterchord1, tau_i1, tau_j1, delta_c1, delta_b1 = segment_1.bound_vortex_ring()
        panel_vortex2, collocation2, normal2, quarterchord2, tau_i2, tau_j2, delta_c2, delta_b2 = segment_2.bound_vortex_ring()
        panel_vortex3, collocation3, normal3, quarterchord3, tau_i3, tau_j3, delta_c3, delta_b3 = segment_3.bound_vortex_ring()

        wake_vortex1 = segment_1.wake_vortex_ring()
        wake_vortex2 = segment_2.wake_vortex_ring()
        wake_vortex3 = segment_3.wake_vortex_ring()

        self.X56A_geometry_x = np.concatenate((segment_1.geometry_x, segment_2.geometry_x, segment_3.geometry_x), axis=1)
        self.X56A_geometry_y = np.concatenate((segment_1.geometry_y, segment_2.geometry_y, segment_3.geometry_y), axis=1)
        self.X56A_bound_vortex_x = np.concatenate((segment_1.bound_vortex_x, segment_2.bound_vortex_x, segment_3.bound_vortex_x), axis=1)
        self.X56A_bound_vortex_y = np.concatenate((segment_1.bound_vortex_y, segment_2.bound_vortex_y, segment_3.bound_vortex_y), axis=1)
        self.X56A_wake_vortex_x = np.concatenate((segment_1.wake_vortex_x, segment_2.wake_vortex_x, segment_3.wake_vortex_x), axis=1)
        self.X56A_wake_vortex_y = np.concatenate((segment_1.wake_vortex_y, segment_2.wake_vortex_y, segment_3.wake_vortex_y), axis=1)

        self.panel_vortex = np.concatenate((panel_vortex1, panel_vortex2, panel_vortex3), axis=1)
        self.panel_vortex = np.reshape(self.panel_vortex, (num_chord*num_span, 4, 3))
        
        self.collocation = np.concatenate((collocation1, collocation2, collocation3), axis=1)
        self.collocation = np.reshape(self.collocation, (num_chord*num_span, 3))
        
        self.normal = np.concatenate((normal1, normal2, normal3), axis=1)
        self.normal = np.reshape(self.normal, (num_chord*num_span, 3))
        
        self.quarterchord = np.concatenate((quarterchord1, quarterchord2, quarterchord3), axis=1)
        self.quarterchord = np.reshape(self.quarterchord, (num_chord*num_span, 3))
        
        self.tau_i = np.concatenate((tau_i1, tau_i2, tau_i3), axis=1)
        self.tau_i = np.reshape(self.tau_i, (num_chord*num_span, 3))
        
        self.tau_j = np.concatenate((tau_j1, tau_j2, tau_j3), axis=1)
        self.tau_j = np.reshape(self.tau_j, (num_chord*num_span, 3))
        
        self.delta_c = np.concatenate((delta_c1, delta_c2, delta_c3), axis=1)
        self.delta_c = np.reshape(self.delta_c, (num_chord*num_span))
        
        self.delta_b = np.concatenate((delta_b1, delta_b2, delta_b3), axis=1)
        self.delta_b = np.reshape(self.delta_b, (num_chord*num_span))
        
        self.wake_vortex = np.concatenate((wake_vortex1, wake_vortex2, wake_vortex3), axis=1)
        self.wake_vortex = np.reshape(self.wake_vortex, (num_wake*num_span, 4, 3))

        self.cs1 = np.zeros((num_chord*num_span,1))
        self.cs1[(num_chord-1)*num_span-num_span2-num_span3:(num_chord-1)*num_span-num_span3, 0] = 1
        self.cs1[num_chord*num_span-num_span2-num_span3:num_chord*num_span-num_span3, 0] = 1
        
        self.cs2 = np.zeros((num_chord*num_span,1))
        self.cs2[(num_chord-1)*num_span-num_span3:(num_chord-1)*num_span-num_span3+4, 0] = 1
        self.cs2[num_chord*num_span-num_span3:num_chord*num_span-num_span3+4, 0] = 1

        self.cs3 = np.zeros((num_chord*num_span,1))
        self.cs3[num_chord*num_span-5:-1, 0] = 1
        self.cs3[(num_chord-1)*num_span-5:(num_chord-1)*num_span-1, 0] = 1
        

    def plot_grid(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')

        for i in range(self.X56A_geometry_x.shape[0]):
            plt.plot(self.X56A_geometry_x[i,:], self.X56A_geometry_y[i,:], color='k')
            plt.plot(self.X56A_bound_vortex_x[i,:], self.X56A_bound_vortex_y[i,:], color='r')
            
        for j in range(self.X56A_geometry_x.shape[1]):
            plt.plot(self.X56A_geometry_x[:,j], self.X56A_geometry_y[:,j], color='k')
            plt.plot(self.X56A_bound_vortex_x[:,j], self.X56A_bound_vortex_y[:,j], color='r')
            
        for i in range(self.X56A_wake_vortex_x.shape[0]):
            plt.plot(self.X56A_wake_vortex_x[i,:], self.X56A_wake_vortex_y[i,:], color='b')
            
        for j in range(self.X56A_wake_vortex_x.shape[1]):
            plt.plot(self.X56A_wake_vortex_x[:,j], self.X56A_wake_vortex_y[:,j], color='b')

        for i, panel in enumerate(self.collocation):
            if self.cs1[i,0] == 1:
                plt.scatter(panel[0], panel[1], color='orange', marker='x')
            elif self.cs2[i,0] == 1:
                plt.scatter(panel[0], panel[1], color='green', marker='x')
            elif self.cs3[i,0] == 1:
                plt.scatter(panel[0], panel[1], color='purple', marker='x')
            # else:
            #     plt.scatter(panel[0], panel[1], color='r', marker='x')

        # plt.plot([0.172,0.172], [0, 0.5])

        plt.xlabel('x')
        plt.ylabel('y')         
            
        plt.show()

    

        
        
