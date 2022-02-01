import numpy as np
import pandas as pd
from scipy import stats

from copula import copula



class archimedean(copula):

    Archimedean_families = ['clayton', 'gumbel', 'frank', 'joe', 'galambos', 'fgm', 'plackett', 'rgumbel', 'rclayton', 'rjoe', 'rgalambos', 'BB1', 'BB2']
    
    def __init__(self, family):
        """
            Creates an Archimedean copula.
            bounds_param = 
            #x0 = Initial guess. Array of real elements of size (n,), where 'n' is the number of independent variables.

        """
        super().__init__()
        self.family = family

        if family  in ['clayton', 'galambos', 'plackett', 'rclayton', 'rgalambos'] :
            self.bounds_param = [(0+ 1e-6, None)]
            self.theta_start = np.array(0.5)

        elif family in ['gumbel', 'joe', 'rgumbel', 'rjoe'] :
            self.bounds_param = [(1, None)]
            self.theta_start = np.array(1.5)

        elif family == 'frank':
            self.bounds_param = [(None)]
            self.theta_start = np.array(2)

        elif family == 'fgm':
            self.bounds_param = [(-1, 1-1e-6)]
            self.theta_start = np.array(0)

        elif family  in ['BB1'] :
            self.bounds_param1 = [(1, None)]
            self.bounds_param2 = [(0, None)]
            self.theta_start1 = np.array(1.5)
            self.theta_start2 = np.array(0.5)

        elif family  in ['BB2'] :
            self.bounds_param1 = [(0, None)]
            self.bounds_param2 = [(0, None)]
            self.theta_start1 = np.array(0.5)
            self.theta_start2 = np.array(0.5)
        else:
            print("family \"%s\" not in list: %s" % (family, archimedean.Archimedean_families) )
            raise ValueError

    def cdf(self, u, v, param):
        """
            returns the cumulative distribution function for the respective archimedean copula family
        """

        if self.family == 'clayton':
            return (u**(-param[0])+v**(-param[0])-1)**(-1/param[0])

        elif self.family == 'rclayton':
            return (u + v - 1 + archimedean(family='clayton').cdf((1-u),(1-v), param) )

        elif self.family == 'gumbel':
            return np.exp(  -( (-np.log(u))**param[0] + (-np.log(v))**param[0] )**(1/param[0]) )

        elif self.family == 'rgumbel':
            return (u + v - 1 + archimedean(family='gumbel').cdf((1-u),(1-v), param) )

        elif self.family == 'frank':
            a = (np.exp(-param[0]*u) -1)*(np.exp(-param[0]*v)-1)
            return (-1/param[0])*np.log(1+a/(np.exp(-param[0])-1))

        elif self.family == 'joe':
            u_ = (1 - u)**param[0]
            v_ = (1 - v)**param[0]
            return 1-(u_+v_-u_*v_)**(1/param[0])

        elif self.family == 'rjoe':
            return (u + v - 1 + archimedean(family='joe').cdf((1-u),(1-v), param) )

        elif self.family == 'galambos':
            return u*v*np.exp(((-np.log(u))**(-param[0])+(-np.log(v))**(-param[0]))**(-1/param[0]) )

        elif self.family == 'rgalambos':
            return (u + v - 1 + archimedean(family='galambos').cdf((1-u),(1-v), param) )

        elif self.family == 'fgm':
            return u*v*(1+param[0]*(1-u)*(1-v))

        elif self.family == 'plackett':
            eta = param[0]-1
            return 0.5*eta**(-1) * (1+eta*(u+v)-( (1+eta*(u+v) )**2 -4*param[0]*eta*u*v)**0.5 )

        elif self.family == 'BB1':
            return (1+( (u**(-param[0]) -1)**param[1] +  (v**(-param[0]) -1)**param[1] )**(1/param[1]) )**(-1/param[0])

        elif self.family == 'BB2':
            u_ = np.exp(param[1]*(u**(-param[0])-1))
            v_ = np.exp(param[1]*(v**(-param[0])-1))

            return (1+ (1/param[1])*np.log(u_+v_-1))**(-1/param[0])
        

    def pdf(self, u, v, param):
        """
            returns the density
        """

        if self.family == 'clayton':
            return ((param[0]+1)*(u*v)**(-param[0]-1))*((u**(-param[0])+v**(-param[0])-1)**(-2-1/param[0]))

        if self.family == 'rclayton':
            return archimedean(family='clayton').pdf((1-u),(1-v), param)

        elif self.family == 'gumbel':
            a = np.power(np.multiply(u, v), -1)
            tmp = np.power(-np.log(u), param[0]) + np.power(-np.log(v), param[0])
            b = np.power(tmp, -2 + 2.0 / param[0])
            c = np.power(np.multiply(np.log(u), np.log(v)), param[0] - 1)
            d = 1 + (param[0] - 1) * np.power(tmp, -1.0 / param[0])
            return archimedean(family='gumbel').cdf(u,v, param) * a * b * c * d

        if self.family == 'rgumbel':
            return archimedean(family='gumbel').pdf((1-u),(1-v), param)

        elif self.family == 'frank':
            a = param[0]*(1-np.exp(-param[0]))*np.exp(-param[0]*(u+v))
            b = (1-np.exp(-param[0])-(1-np.exp(-param[0]*u))*(1-np.exp(-param[0]*v)))**2
            return a/b

        elif self.family == 'joe':
            u_ = (1 - u)**param[0]
            v_ = (1 - v)**param[0]
            a = (u_+v_-u_*v_)**(-2+1/param[0])
            b = ((1-u)**(param[0]-1))*((1-v)**(param[0]-1))
            c = param[0]-1+u_+v_+u_*v_
            return a*b

        if self.family == 'rjoe':
            return archimedean(family='joe').pdf((1-u),(1-v), param)

        elif self.family == 'galambos':
            x = -np.log(u)
            y = -np.log(v)
            return (self.cdf(u,v, param)/(v*u))*(1-((x**(-param[0]) +y**(-param[0]))**(-1-1/param[0]))*(x**(-param[0]-1) +y**(-param[0]-1))
            + ((x**(-param[0]) +y**(-param[0]))**(-2-1/param[0]))*((x*y)**(-param[0]-1))*(1+param[0]+(x**(-param[0]) +y**(-param[0]))**(-1/param[0])))

        if self.family == 'rgalambos':
            return archimedean(family='galambos').pdf((1-u),(1-v), param)

        elif self.family == 'fgm':
            return 1+param[0]*(1-2*u)*(1-2*v)

        elif self.family == 'plackett':
            eta = (param[0]-1)
            a = param[0]*(1+eta*(u+v-2*u*v))
            b = ((1+eta*(u+v))**2-4*param[0]*eta*u*v)**(3/2)
            return a/b

        elif self.family == 'BB1':
            x = (u**(-param[0])-1)**(param[1])
            y = (v**(-param[0])-1)**(param[1])
            return ((1+(x+y)**(1/param[1]))**(-1/param[0]-2))*((x+y)**(1/param[1]-2))*(param[0]*(param[1]-1)+(param[0]*param[1]+1)*(x+y)**(1/param[1]))*((x*y)**(1-(1/param[1])))*((u*v)**(-param[0]-1))

        elif self.family == 'BB2':
            x = np.exp(param[1]*(u**(-param[0])-1))
            y = np.exp(param[1]*(v**(-param[0])-1))

            A = (1+(param[1]**-1)*np.log(x + y - 1))**(-1/(param[0]-2))
            B = (x + y - 1)**(-2)
            C= (1 + param[0] + param[0]*param[1] + param[0]*np.log(x+y-1))
            D = (x)*(y)*((u*v)**(-param[0]-1))
            return np.dot(A*B*C,D)

    def LTDC(self, theta):
        """
            Returns the lower tail dependence coefficient for a given self.theta
        """

        if self.family  in ['gumbel', 'joe', 'frank', 'galambos', 'fgm', 'plackett', 'rclayton']:
            return 0

        elif self.family  in ['rgalambos', 'clayton'] :
            return 2**(-1/theta)

        elif self.family  in ['rgumbel', 'rjoe'] :
            return 2-2**(1/theta)

    def UTDC(self, theta):
        """
            Returns the upper tail dependence coefficient for a given self.theta
        """

        if self.family  in ['clayton', 'frank', 'fgm', 'plackett', 'rgumbel', 'rjoe', 'rgalambos']:
            return 0

        elif self.family  in ['galambos', 'rclayton'] :
            return 2**(-1/theta)

        elif self.family  in ['gumbel', 'joe'] :
            return 2-2**(1/theta)


