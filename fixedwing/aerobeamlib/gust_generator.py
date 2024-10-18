import numpy as np
from matplotlib import pyplot as plt

def discrete_gust(V_inf, aerogrid_collocation, dt, timesteps, H):

    Uds = 0.05 * V_inf    # Max gust velocity
    H = 5      # Gust penetration (half of gust profile)
    
    upwash = np.zeros((len(aerogrid_collocation), timesteps))
    for j, t in enumerate(range(timesteps)):
        for i, collocation in enumerate(aerogrid_collocation):
            s = V_inf*dt*t - collocation[0]
            
            if s < 2*H and s > 0:
                upwash[i,j] = Uds/2*(1-np.cos(np.pi*s/H))
            else:
                upwash[i,j] = 0

    return upwash