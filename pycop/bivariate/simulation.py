import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from scipy.stats import norm, t

###
# See  Genest and MacKay (1986) The joy of copulas: bivariate distributions with uniform marginals
### General algorithm to generate pairs of random variables whose distribution function is given by an Archimedean Copula


def simu_gaussian(num, rho):
    """ 
    # Gaussian Copula with correlation rho

    """
    v1 = np.random.normal(0,1,num)
    v2 = np.random.normal(0,1,num)

    RHO = [[1,rho],[rho, 1]]
    L = linalg.cholesky(RHO, lower=True)
    y1, y2 = np.dot(L, [v1, v2])
    u1 = norm.cdf(y1, 0, 1)
    u2 = norm.cdf(y2, 0, 1)

    return u1, u2

def simu_tstudent(num, nu, rho):
    """ 
    # Bivariate student Copula with nu degrees of freedom and correlation rho

    """
    v1 = np.random.normal(0,1,num)
    v2 = np.random.normal(0,1,num)
    RHO = [[1,rho],[rho, 1]]
    L = linalg.cholesky(RHO, lower=True)
    y1, y2 = np.sqrt(nu/np.random.chisquare(df=nu,size=num) )*np.dot(L, [v1, v2])

    u1 = t.cdf(y1, df=nu, loc=0, scale=1)
    u2 = t.cdf(y2, df=nu, loc=0, scale=1)

    return u1, u2



def simu_clayton(num, theta):
    """
    # Clayton copula
    # Devroye, L. (1986) Non-uniform Random Variate Generation.
    # Devroye (1986) has proposed a simpler method for Clayton's copula
    """

    v1 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    v2 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    x = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0,num)])
    u1 = (1 + v1/x)**(-1/theta)
    u2 = (1 + v2/x)**(-1/theta)
    return u1, u2


def simu_frank(num, theta):
    """
    # Frank's copula
    """
    v1 = np.random.uniform(0,1,num)
    v2 = np.random.uniform(0,1,num)

    u1 = v1

    u2 = (-1/theta)*np.log(1+(v2*(np.exp(-theta)-1))/(v2 + (1-v2)*np.exp(-theta*v1) ))

    return u1, u2

def simu_fgm(num, theta):
    """
    # FGM's copula
    """
    v1 = np.random.uniform(0,1,num)
    v2 = np.random.uniform(0,1,num)
    
    A = 1 +theta*(1-2*v1)
    B = np.sqrt(A**2 -4*(A-1)*v2)

    u1 = v1
    u2 = (2*v2)/(A+B)

    return u1, u2

from scipy.stats import levy_stable

def simu_gumbel(num, theta):
    """
    # Gumbel copula
    """
    # https://cran.r-project.org/web/packages/gumbel/gumbel.pdf
    # https://cran.r-project.org/web/packages/gumbel/vignettes/gumbel.pdf

    d = theta
    alpha = 1/theta
    beta = 1
    gamma =1
    delta = 0
    X = levy_stable.rvs(alpha=1/theta, beta=1,scale=(np.cos(np.pi/(2*theta)))**theta,loc=0, size=num)

    v1 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    v2 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    def phi_1(t): return np.exp(-t**(1/theta))

    u1 = phi_1(v1/X)
    u2 = phi_1(v2/X)

    return u1, u2

from scipy.special import gamma, comb


def SimuSibuya(alpha, m):
    """
        Sibuya distribution Sibuya(α)
        Used for sampling F=Sibuya(α) for Joe copula 
        The algorithm in given in Proposition 3.2 in Hofert (2011) "Efficiently sampling nested Archimedean copulas"
    """

    G_1 = lambda y: ((1-y)*gamma(1-alpha) )**(-1/alpha)
    F = lambda n: 1- ((-1)**n)*comb(n,alpha-1)

    X = np.random.uniform(0,1,m)

    for i in range(0,len(X)):
        if X[i] <= alpha:
            X[i] = 1

        if F(np.floor(G_1(X[i]))) < X[i]:
            X[i] = np.ceil(G_1(X[i]))
        else:
            X[i] = np.floor(G_1(X[i]))
    return X


def simu_joe(num, theta):
    """
    # Joe copula
    Requires:
        n = number of variables to generate
        m = sample size
        theta = Joe copula parameter
    """

    alpha = 1/theta
    X = SimuSibuya(alpha, num)

    v = [np.random.uniform(0,1,num) for i in range(0,2)]

    phi_t = lambda t: (1-(1-np.exp(-t))**(1/theta))

    u = [phi_t(-np.log(v[i])/X) for i in range(0,2)]

    return u