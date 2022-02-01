
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from mpmath import polylog
from scipy.special import gamma

class archimedean():
    
    def __init__(self, d, family):
        """
            Creates an Archimedean copula.
            bounds_param = 
            #x0 = Initial guess. Array of real elements of size (n,), where 'n' is the number of independent variables.
        """

        self.family = family
        self.d = d

        if family == 'AMH':
            self.bounds_param = [(0+ 1e-6, 1-1e-6)]
            self.theta_start = np.array(0.5)
            self.phit = lambda theta, t : (1-theta)/(np.exp(t)-theta)
            self.phit_inv = lambda theta, t : np.log((1-theta+theta*t)/t)
            self.phit_invp = lambda theta, t : theta/(theta*t-theta+1)
            self.phit_d = lambda theta, d, t :  ((1-theta)/theta)* float(polylog(-d,theta*np.exp(-t)).real)
        
        if family == 'clayton':
            self.bounds_param = [(0+ 1e-6, None)]
            self.theta_start = np.array(0.5)
            self.phit = lambda theta, t : (1+ t)**(-1/theta)
            self.phit_inv = lambda theta, t : (1/theta)*(t**(-theta)-1)
            self.phitp = lambda theta, t :(-1/theta)*(1+t)**(-1/theta-1)
            self.phit_d = lambda theta, d, t :  np.prod([k+(1/theta) for k in range(0,d) ])*(1+t)**(-(d+1/theta))#(gamma(d+1/theta)/gamma(1/theta) )

        if family == 'gumbel':
            self.bounds_param = [(1+ 1e-6, None)]
            self.theta_start = np.array(0.5)
            
            self.phit = lambda theta, t : 2
            self.phit_inv = lambda theta, t : 2
            self.phit_d = lambda theta, d, t,  :  2


        self.cdf = lambda theta, u : self.phit(theta, sum([self.phit_inv(theta, ui) for ui in u]))
        self.pdf = lambda theta, u : self.phit_d(theta, self.d, sum([self.phit_inv(theta, ui) for ui in u]) )  /np.prod([self.phitp(theta, self.phit_inv(theta, ui)) for ui in u])

        self.logl = lambda theta, uj : np.log( ((-1)**d) *self.phit_d(theta, self.d, sum([self.phit_inv(theta, uij) for uij in uj])) ) + sum([np.log(-self.phit_invp(theta, uij)) for uij in uj])


