import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

def pq_upper(Eg,gamma_grid):
    
    Eg_grid = [Eg(gamma) for gamma in gamma_grid]
    Eg_grid = np.array(Eg_grid)
    
    # We are ready to compute the new (p3,q3) pairs, which is unifom on log(gamma)
    aa = np.array((Eg_grid[2:]-Eg_grid[1:-1])/(gamma_grid[2:]-gamma_grid[1:-1]))
    bb = np.array((Eg_grid[1:-1]-Eg_grid[0:-2])/(gamma_grid[1:-1]-gamma_grid[0:-2]))
    
    init_slope = (Eg_grid[0]-1)/gamma_grid[0]
    q = aa-bb
    q = np.concatenate(([1+init_slope],[bb[0]-init_slope],q,[-aa[-1]]))
    p = np.concatenate(([0],q[1:]*gamma_grid))
    q = np.concatenate((q,[0]))
    p = np.concatenate((p,[np.maximum(0,1-sum(p))]))
    return p,q
    
def pq_lower(Eg,Egslope,gamma_grid):
    
    f_grid = np.zeros(np.shape(gamma_grid))
    f_grid[0] = Eg(gamma_grid[0])
    f_grid[-1] = 0
    
    for i in range(1,len(f_grid)-2):
        myfunc = lambda gamma: (Eg(gamma)-f_grid[i-1])/(gamma-gamma_grid[i-1])-Egslope(gamma)
        tmin = gamma_grid[i-1]+np.spacing(gamma_grid[i-1])
        tmax = gamma_grid[i]
        if i%100==0:
            print(i)
        
        if myfunc(tmax) >= 0:
            f_grid[i]=Eg(tmax)
        else:
            if myfunc(tmin)<=0:
                slp=Egslope(gamma_grid[i-1])
            else:          
                t = fsolve(myfunc,tmin,xtol=1e-3)
                slp = (Eg(t)-f_grid[i-1])/(t-gamma_grid[i-1])
            f_grid[i] = max(0,f_grid[i-1]+slp*(gamma_grid[i]-gamma_grid[i-1]))
    gamma_grid_ext = np.concatenate(([0],gamma_grid))
    print(np.shape(gamma_grid_ext))
    f_grid_ext = np.concatenate(([1],f_grid))
    print(np.shape(f_grid_ext))
    hull = ConvexHull(np.stack((gamma_grid_ext,f_grid_ext)))
    K=hull.simplices
    print("K",K)
    slope = np.array([f_grid_ext[i] for i in K[1:-1]]-[f_grid_ext[i] for i in K[0:-2]])/np.array([gamma_grid_ext[i] for i in K[1:-1]]-[gamma_grid_ext[i] for i in K[0:-2]])
    q = np.zeros(np.shape(gamma_grid))
    for j in range(1,len(K)-2):
        q[K[j]] = slope[j]-slope[j-1]
    q[-1]=-slope[-2]
    p = q*gamma_grid
    return p,q