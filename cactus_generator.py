'''
This function solves the optimization problem
minimize max_{|a|<1} D(P(x)||P(x-a))
subject to E c(X) <= C
Solved using an interior point second-order method.

It actually solves a finite dimensional approximation, which is roughly
minimize max_{|a|\le n} \sum_{i=-xmax*n}^{(xmax-1)*n} p(i)*log(p(i)/p(i+a))
subject to \sum_i p(i) = 1
           \sum_i (i*n)^2*p(i) <= D
In the above, n is the parameter that controls the quantization, and xmax
determines the range of the distribution that is computed. 

It actually solves a modification of the above optimization
problem by replacing the max in the objective with a soft-max given by a
log-sum-exp fnuction. That is, the objective function is replaced with
log(\sum_{a=1}^n \exp(\sum_{i=-xmax*n}^{(xmax-1)*n} t*p(i)*log(p(i)/p(i+a))))/t
This remains a convex function, and for large t, it gives the same
solution as the above. The basic technique is an equality-constrained (to 
deal with the two constraints) Newton minimization, and then when close 
to optimal the t value is increased.

The inputs are:
 1. n is the quantization level. Bins are of length 1/n
 2. xmax is the size of the interval (from -xmax to +xmax). In the
    terminology from the paper, N=xmax*n.
 3. C is the cost value
 4. c_exp is the cost exponent. That is, the cost function is
       c(x)=abs(x)^c_exp (Default is 2)
 5. opt_type should be either 1 or 2. 1 solves the problem where the
    objective will always be a true achievable bound. 2 solves a simpler
    problem, which actually gives a lower bound on the continuous
    optimization problem. (Default is 1)
 6. r is the geometric fall-off constant (Default is 0.9)
 7. tol is the tolerance for the solver. (Default is 1e-8)
 8. verbose controls how much information is displayed. 0 does nothing. 
    1 prints some data. 2 also plots the candidate distribution. (Default is 2)

The outputs are:
 1. primobj is the primal objective value.
 2. x is the vector of values at the centers of the quantization bins.
 3. p is the solution to the optimization problem.
 4. samplefnc is a function that will produce one random sample from the
    distribution when called with no arguments.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import math 
import matplotlib.pyplot as plt
import numpy as np 
from numpy.random import rand 
import scipy 
from scipy.special import gamma 
from scipy.special import gammaincc


def getallobj(p,n,R):
  # calculate all objective values for different shifts
  allobj=zeros(n)
  l=len(p)
  if opt_type == 1:
    for k in range(n):
      o1=np.sum(multiply((p[1:l-k-1] - p[1+k:l-1]), np.log(p[1:l-k-1] / p[1+k:l-1])))           
      q_plus= p[l-1]*np.power(R, arange(0,k+1,1))
      q_minus=p[0]*np.power(R, arange(k,-1,- 1))
      o2=sum(multiply((p[l-k-1:l] - q_plus), np.log(p[l-k-1:l] / q_plus))) + sum(multiply((p[1:2+k] - q_minus), np.log(p[1:2+k] / q_minus)))
      o3=(p[0] + p[l-1])*k*(1 - R ** (k+1)) / (1 - R)*(- log(R))

      allobj[k]=o1 + o2 + o3
  return allobj


def getgrad(p,k,R,opt_type):
  # Calculate gradient for a shift of k
  l=len(p)

  rat1=np.concatenate((p[2 + k:],p[l-1]*np.power(R, arange(1,k+1)))) / p[1:l-1]
  rat2=np.concatenate((p[0]*np.power(R,arange(k,0,-1)),p[0:l-k-2])) / p[1:l-1]
  gradmid=(- np.log(rat1) + 1 - rat1) + (- np.log(rat2) + 1 - rat2)

  Nfactor=(1+k)*(1 - R ** (1+k)) / (1 - R)*(- log(R))
  gradN=sum(multiply(np.power(R,arange(0,k+1)), np.log(rat1[-k-1:]) + 1 - 1.0 / rat1[-k-1:])) + Nfactor
  gradmN=sum(multiply(np.power(R,arange(k,-1,-1)),(np.log(rat2[:k+1]) + 1 - 1.0 / rat2[:k+1]))) + Nfactor

  grad=np.concatenate(([gradmN],gradmid,[gradN]))

  return grad

def getH(p,n,evals,R,opt_type):
  # Calculate the Hessian matrix
  l=len(p)

  d=zeros(l)
    for k in range(n):
      dN = sum(np.power(R, arange(0,k+1)))/p[l-1] + sum(p[-k-1:]) / p[l-1] ** 2
      dmN=sum(np.power(R,arange(k-1,-1,-1))) / p[0] + sum(p[1:1+k]) / p[0] ** 2
      dmid=2.0/p[1:l-1] + (np.concatenate((p[2+k:], np.power(R, arange(1,k+1))*p[l-1])) + np.concatenate((np.power(R, arange(k,0,-1))*p[0], p[:l-2-k] )))/ p[1:l-1]**2
      d=d + evals[k]*np.concatenate(([dmN],dmid,[dN]))
    
    H=np.diagflat(d)

    q=zeros((n,1))
    for k in range(n):
      j = 1 + k
      q[k]=sum(multiply(evals[k:n],(- np.power(R,arange(0,n-k))) / p[j] - 1 / p[0]))
    H[0,1:n+1]=q.reshape(-1)
    H[1:n+1,0]=q.reshape(-1)


    # deal with i=l
    q=zeros((n,1))
    for k in range(n):
      i = l-k-1
      q[k]=sum(multiply(evals[k:n], -np.power(R,arange(0,n-k))/p[i] - 1.0/p[l-1]))
    H[l-1,range(l-2,l-n-2,-1) ]=q.reshape(-1)
    H[range(l-2,l-n-2,-1),l-1]=q.reshape(-1)  

    for k in range(1,l-2):
      kmax = min(n,l-k-1)
      q=multiply(evals[:kmax], (-1/p[k]-1.0/p[k+1:k+1+kmax]))
      H[k, k+1:k+1+kmax]=q.reshape(-1)
      H[k+1:k+1+kmax,k]=q.reshape(-1)           
  
  return H


def cactus_generator(n, xmax, C, c_exp=2, opt_type=1, r=0.9, tol=1e-08, verbose=2):

  N = math.ceil(n*xmax)
  xmax = N/n

  x=np.arange(-N,N+1)/n
  l=len(x)

  # The two constraints (probability normalization, and the variance constraint) are handled by A*p=b as follows. 
  # Note that even though the variance constraint is an inequality, it will always be tight at optimal, so we treat it as an equality constraint.

  c=np.zeros(l)
  for i in range(l):
    if x[i]>0:
      c[i] = ((x[i]+1/2/n)**(c_exp+1)-(x[i]-1/2/n)**(c_exp + 1))*(n/(c_exp+1))
    elif x[i]==0:
      c[i] = (1 / 2 / n) ** c_exp / (1 + c_exp)
    else:
      c[i]=np.dot(((abs(x[i]) + 1 / 2 / n) ** (c_exp + 1) - (abs(x[i]) - 1 / 2 / n) ** (c_exp + 1)),(n / (c_exp + 1)))

  # slight over-estimate, but getting the exact value requires a nasty infinite sum. This upper bounds the sum by an integral. 
  # This still gives a true bound, without much loss
  cN = n*r**(-N-1/2)*(-n*np.log(r))**(-c_exp-1)*gamma(c_exp+1)*gammaincc(c_exp+1,-np.log(r)*(N-1/2))
  c[0]=cN
  c[l-1]=cN
  d = np.ones(l)
  d[0] = d[l-1] = 1/(1-r)  
  A = np.stack((c,d),axis = 0) 

  b = np.stack(([C],[1]),axis = 0)
  # We start with a guess for the distribution p.
  # This is a distribution designed to have farily heavy tails (which seems to make the solver work better), and to roughly satisfy the cost constraint.
  q = C ** (- 1 - 2 / c_exp)
  p = 1/(1+q*abs(x)**(c_exp+2))
  # make sure that p is normalized (it actually doesn't need to satisfy the cost constraint exactly initially -- the algorithm with ensure that)
  p = p/np.dot(A[1],p) 
  
  iter=0

  t=1

  isfeasible = False

  allobj = getallobj(p,n,r,opt_type)
  
  primobj=np.amax(allobj)
  fval=np.log(sum(np.exp(t*(allobj - primobj)))) / t + primobj

  while 1:

    iter = iter + 1
    eta = np.exp(t*(allobj - primobj))
    eta=eta / sum(eta)
    allgrads=np.zeros((l,n))

    for k in range(n):
      allgrads[:,k]=getgrad(p,k+1,r,opt_type).reshape(-1)
    print(np.shape(allgrads))
    print(np.shape(eta.reshape(n,1)))
    grad=np.matmul(allgrads,eta.reshape(n,1))
    print(grad[0:15])

    rpri=np.matmul(A,p.reshape(len(p),1))-b

    # The next few lines are typically the majority of the run-time.
    # Compute Hessian matrix
    H=getH(p,n,eta,r,opt_type)
    # strictly positive definite. This doesn't much change overall performance
    fullH = H + np.matmul(np.matmul(allgrads,np.diagflat(t*eta)),allgrads.T)-np.matmul(t*(1-1e-05)*grad,grad.T)
    
    try:
      R=np.linalg.cholesky(fullH)

      gtilde=np.linalg.solve(R.T,grad)

      AR=np.matmul(A, np.linalg.inv(R))
      U,S,V = np.linalg.svd(AR,full_matrices=False)
      s=S
      vtilde = - gtilde - np.matmul(V.T,(1.0/s.reshape(2,1)*np.matmul(U.T,rpri))) + np.matmul(V.T,(np.matmul(V,gtilde)))
      v=np.linalg.solve(R,vtilde)

    except np.linalg.LinAlgError as err:
      if verbose >=1:
          print('Chlesky failed')
      A01=np.concatenate((fullH, A.T),axis=1)
      A02=np.concatenate((A, np.zeros((2,2))),axis=1)
      A1 = np.concatenate((A01,A02),axis=0)
      b1=np.concatenate((-grad, -rpri))
      vw = np.linalg.solve(A1,b1)
      v = vw[0:l]
    nwt_dec=-sum(grad*v) / 2

    # implementing a line search. We move to p+dst*v
    dst=1
    while 1:
      pnew = p + dst*v.reshape(-1)
      if min(pnew) > 0:
        allobj=getallobj(pnew,n,r,opt_type)
        primobj=max(allobj)
        newfval=np.log(sum(np.exp(t*(allobj - primobj)))) / t + primobj
        if not isfeasible:
          break
        if newfval < fval + 0.1*dst*(np.matmul(grad.T,v)):
          break
        if dst < 1e-08:
          break
      dst=dst / 2

    if not isfeasible and dst == 1:
      isfeasible=True
      if verbose >= 1:
          print('Feasible!')
    p=pnew
    fval=newfval
    if verbose >= 1:
      print('iter=%d  dst=%1.1e  primobj=%1.4f  nwt_dec=%1.2e  t=%1.1e \n' %(iter,dst,primobj,nwt_dec,t));

    if isfeasible:
      # n/e/t is an estimate for the duality gap, if we are at exact optimality for the soft-max problem. 
      # Thus nwt_dec+n/e/1 is an estimate for the total duality gap
      if nwt_dec + n / exp(1) / t < tol:
          break
      # if we take a full Newton step or get extremely close to optimal, then increase t.
      if dst == 1 or nwt_dec < tol / 2:
          # This is a fairly modest increase in t.
          # This seems to make the solver the most robust, if not necessarily the most efficient.
          t=t*1.25
    
  plt.plot(x,p)
  filename=('cactus_x_d1.0_l%d_v%.2f.csv' %(c_exp,C)
  np.savetxt(filename, x, delimiter=",")
      
  filename=('cactus_p_d1.0_l%d_v%.2f.csv' %(c_exp,C))
  np.savetxt(filename, p, delimiter=",")
      
  return x, p 