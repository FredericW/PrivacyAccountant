'''
Note that the moment accountant/rdp_accountant function from tensorflow_privacy 
is an analytic method specifically designed for gaussian pdf.
Here in this code, we follow the definitions of rdp strictly, 
that is, we perform integrations to obtain the moments.
'''

import tensorflow as tf
import numpy as np
import mpmath as mp
import scipy as sp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

def compute_rdp_general(noise_type,orders,c_type,C,sen,q):
  
  # note that here the universal parameter for all distributions is the sigma,
  # so if you'd like to use L1-norm, you must choose the approperate sigma!!!
  
 
    if noise_type=="cactus":  
        # For cactus, we directly use the discrete pmf from file to compute the renyi-divergence with given order

        filename = ('TestData/cactusM_s%.1f_%s_c%.2f_x.csv' %(sen,c_type,C))
        x = np.genfromtxt(filename, dtype=np.float32)
        x = x.reshape(-1)

        filename = ('TestData/cactusM_s%.1f_%s_c%.2f_p.csv' %(sen,c_type,C))
        p = np.genfromtxt(filename, dtype=np.float32)
        p = p.reshape(len(p),1)

        xmax=max(x)
        n = len(x)/2//xmax 
        shift_n = int(n*sen)

        ps = p[shift_n:].reshape(-1)
        x=np.linspace(-xmax+sen,xmax,len(ps))

        l1 = sum(ps*np.abs(x))
        print("l1 cost =", l1)
        l2 = sum(ps*np.power(x,2))
        print("l2 cost =", l2)

        qs = q*p[:len(p)-shift_n].reshape(-1)+(1-q)*ps
    
    else: # For all other distributions, we sample the pdf to generate the Renyi-divergence!
    
        n=500.0 # quantization rate
        xmax=max(10,10*np.sqrt(C)) # the domin limit
        x=np.linspace(-xmax, xmax, int(2*xmax*n)) # domain of the distribution
        if noise_type=="gaussian":
            sigma=np.sqrt(C) if c_type=="l2" else C*np.sqrt(np.pi/2)
            f = lambda x: 1/np.sqrt(2*np.pi)/sigma*np.exp(-x**2/2/sigma**2)
            print("here",sigma)

        if noise_type == "laplace":
            b = np.sqrt(C/2) if c_type=="l2" else C
            f = lambda x:1/2/b*np.exp(-np.abs(x)/b)

        if noise_type == "airy":
            a0=1.01879
            f = lambda x: np.array(sp.special.airy(2*a0/3/C*np.abs(x)-a0)[0])**2.0/3.0/C/np.array(sp.special.airy(-a0)[0])**2.0
      
        ps = np.array([f(X) for X in x])/n
        l1 = sum(ps*np.abs(x))
        print("l1 cost =", l1)
        l2 = sum(ps*np.power(x,2))
        print("l2 cost =", l2)
    
        qs = q*np.array([f(X) for X in x-sen])/n+(1-q)*ps
        ps= ps.reshape(-1)
        qs = qs.reshape(-1)
    
    logmgf = lambda t: np.log(sum(qs*np.power(ps/qs,t)))
    return [logmgf(alpha)/(alpha-1) for alpha in orders]

def compute_epsilon(steps,orders,rdp_array,delta):
    rdp_array = rdp_array*np.array(steps)
    return get_privacy_spent(orders, rdp_array, target_delta=delta)[0]

def compute_delta(steps,orders,rdp_array,eps):
    rdp_array = rdp_array*np.array(steps)
    return get_privacy_spent(orders, rdp_array, target_eps=eps)[1]