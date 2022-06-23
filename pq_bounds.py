import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

def pq_upper(Eg,gamma_grid):
    
    Eg_grid = [Eg(gamma) for gamma in gamma_grid]
    
    # We are ready to compute the new (p3,q3) pairs, which is unifom on log(gamma)
    aa = (Eg_grid[2:-1]-Eg_grid[1:-2])/(gamma_grid[2:-1]-gamma_grid[1:-2])
    bb = (Eg_grid[1:-2]-Eg_grid[0:-3])/(gamma_grid[1:-2]-gamma_grid[0:-3])
    
    init_slope = (Eg_grid[0]-1)/gamma_grid[0]
    q = aa-bb
    q = [1+init_slope,bb[0]-init_slope,q2,-aa[-1]];
    p = [0,q[1:]*gamma_grid]
    q = [q,0]
    p = [p,max(0,1-sum(p))]
    return p,q
    
def pq_lower(Eg,Egslope,gamma_grid):
    
    f_grid = np.zeros(np.shape(gamma_grid))
    f_grid[0] = Eg(gamma_grid[0])
    f_grid[-1] = 0
    
    for i in range(1,len(f_grid)-2):
        myfunc = lambda gamma: (Eg(gamma)-f_grid[i-1])/(gamma-gamma_grid[i-1])-Egslope(gamma)
        tmin = gamma_grid[i-1]+np.spacing(gamma_grid[i-1])
        tmax = gamma_grid[i]
        
        if myfunc(tmax) >= 0:
            f_grid[i]=Eg(tmax)
        else:
            if myfunc(tmin)<=0:
                slp=Egslope(gamma_grid2[i-1])
            else:
                t = fslove(myfunc,[tmin,tmax])
                slp = (Eg(t)-f_grid[i-1])/(t-gamma_grid[i-1])
            f_grid[i] = max(0,f_grid[i-1]+slp*(gamma_grid[i]-gamma_grid[i-1]))
    gamma_grid_ext = [0,gamma_grid]
    f_grid_ext = [1,f_grid]
    K = ConvexHull([gamma_grid_ext;f_grid_ext])
    
    slope = ([f_grid_ext[i] for i in K[1:-1]]-[f_grid_ext[i] for i in K[0:-2]])/([gamma_grid_ext[i] for i in K[1:-1]]-[gamma_grid_ext[i] for i in K[0:-2]])
    q = np.zeros(np.shape(gamma_grid))
    for 