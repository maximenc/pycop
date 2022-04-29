# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy.stats import norm, t, levy_stable, logser
from scipy.special import gamma, comb
from distutils.log import error

def simu_gaussian(n, m, corrMatrix):
    """ 
    # Gaussian Copula simulations with a given correlation matrix

    Parameters
    ----------
    n : int
        the dimension number of simulated variables
    m : int
        the sample size 
    corrMatrix : array
        the correlation matrix

    Returns
    -------
    u : array
        the simulated sample 

    """

    # Generate n independent standard Gaussian random variables V = (v1 ,..., vn):
    v = [np.random.normal(0,1,m) for i in range(0,n)]

    # Compute the lower triangular Cholesky factorization of the correlation matrix:
    L = linalg.cholesky(corrMatrix, lower=True)
    y = np.dot(L, v)
    u = norm.cdf(y, 0, 1 )

    return u


def simu_tstudent(n, m, corrMatrix, nu):
    """
    # Student Copula with k degrees of freedom and a given correlation matrix

    Parameters
    ----------
    n : int
        the dimension number of simulated variables
    m : int
        the sample size 
    corrMatrix : array
        the correlation matrix
    nu : float
        the degree of freedom 

    Returns
    -------
    u : array
        the simulated sample 

    """

    # Generate n independent standard Gaussian random variables  V = (v1 ,..., vn):
    v = [np.random.normal(0,1,m) for i in range(0,n)]

    # Compute the lower triangular Cholesky factorization of rho:
    L = linalg.cholesky(corrMatrix, lower=True)
    z = np.dot(L, v)

    # generate a random variable r, following a chi2-distribution with nu degrees of freedom
    r = np.random.chisquare(df=nu,size=m)

    y = np.sqrt(nu/ r)*z
    u = t.cdf(y, df=nu, loc=0, scale=1)

    return u


def SimuSibuya(alpha, m):
    """
    # Sibuya distribution Sibuya(α)
    Used for sampling F=Sibuya(α) for Joe copula 
    The algorithm is given in Proposition 3.2 in Hofert (2011) "Efficiently sampling nested Archimedean copulas"

    Parameters
    ----------
    alpha : float
        the alpha parameter, α = 1/θ
    m : int
        the sample size

    Returns
    -------
    X : array
        the simulated sample 

    """

    G_1 = lambda y: ((1-y)*gamma(1-alpha) )**(-1/alpha)
    F = lambda n: 1- ((-1)**n)*comb(n,alpha-1)

    X = np.random.uniform(0,1,m)

    for i in range(0,len(X)):
        if X[i] <= alpha:
            X[i] = 1
        elif F(np.floor(G_1(X[i]))) < X[i]:
            X[i] = np.ceil(G_1(X[i]))
        else:
            X[i] = np.floor(G_1(X[i]))

    return X



def simu_archimedean(familly, n, m, theta):
    """
    Archimedean copula simulation

    Parameters
    ----------

    familly : str
        type of the distribution
    n : int
        the dimension number of simulated variables
    m : int
        the sample size
    theta : float
        copula parameter
        Clayton: θ ∈ [0, inf)
        Gumbel: θ ∈ [1, +inf)
        Frank:  θ ∈ (-inf, +inf)
        Joe: θ ∈ [1, +inf)
        AMH: θ ∈ [0, 1)

    Returns
    -------
    u : array
        the simulated sample, array matrix with dim (m, n)

    """

    if familly == "clayton":
        # Generate n independent standard uniform random variables  V = (v1 ,..., vn):
        v = [np.random.uniform(0,1,m) for i in range(0,n)]
        # generate a random variable x following the gamma distribution gamma(theta**(-1), 1)
        X = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0,m)])
        phi_t = lambda t:  (1+t)**(-1/theta)
        u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]

    elif familly == "gumbel":
        v = [np.random.uniform(0,1,m) for i in range(0,n)]
        X = levy_stable.rvs(alpha=1/theta, beta=1,scale=(np.cos(np.pi/(2*theta)))**theta,loc=0, size=m)

        phi_t = lambda t:  np.exp(-t**(1/theta))

        u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]

    elif familly == "frank":

        v = [np.random.uniform(0,1,m) for i in range(0,n)]
        p = 1-np.exp(-theta)
        X = logser.rvs(p, loc=0, size=m, random_state=None)

        phi_t = lambda t:  -np.log(1-np.exp(-t)*(1-np.exp(-theta)))/theta
        u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]
    
    elif familly == "joe":

        alpha = 1/theta
        X = SimuSibuya(alpha, m)

        v = [np.random.uniform(0,1,m) for i in range(0,n)]

        phi_t = lambda t: (1-(1-np.exp(-t))**(1/theta))

        u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]


    elif familly == "amh":

        v = [np.random.uniform(0,1,m) for i in range(0,n)]

        X = np.random.geometric(p=1-theta, size=m)

        phi_t = lambda t:  (1-theta)/(np.exp(t)-theta)

        u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]

    else:
        raise error

    return u

def simu_mixture(n, m, combination):
    """
    Mixture copula simulation

    Parameters
    ----------

    n : int
        the dimension number of simulated variables
    m : int
        the sample size

    combination : list
        A list of dictionaries that contains information on the copula to combine.

        exemple:
        combination =[
            {
                "type": "clayton",
                "weight": 0.5,
                "theta": 4
            },
            {
                "type": "student",
                "weight": 0.5,
                "corrMatrix": corrMatrix,
                "nu":2
            }
        ]

    Returns
    -------
    u : array
        the simulated sample, array matrix with dim (m, n)

    """
    

    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    weights = [comb["weight"] for comb in combination]

    #Generate a random sample of indexes of combination types 
    y = np.array([np.where(ls==1)[0][0] for ls in np.random.multinomial(n=1, pvals=weights, size=m)])

    for i in range(0,len(combination)):
        combinationsize = len(v[0][y==i])

        if combination[i]["type"] == "gaussian":
            corrMatrix = combination[i]["corrMatrix"]

            vi = simu_gaussian(n, combinationsize, corrMatrix)
            for j in range(0,len(vi)):
                    v[j][y==i] = vi[j]
        elif combination[i]["type"] == "student":
            corrMatrix = combination[i]["corrMatrix"]
            nu = combination[i]["nu"]
            vi = simu_tstudent(n, combinationsize, corrMatrix, nu)

        elif combination[i]["type"] in ["clayton", "gumbel", "frank", "joe", "amh"]:
            vi = simu_archimedean(combination[i]["type"], n, combinationsize, combination[i]["theta"])

            for j in range(0,len(vi)):
                    v[j][y==i] = vi[j]
        else:
            raise error

    return v

