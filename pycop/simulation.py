# -*- coding: utf-8 -*-
import numpy as np
from scipy import linalg
from scipy.stats import norm, t, levy_stable, logser
from scipy.special import gamma, comb
from distutils.log import error
from typing import List


def simu_gaussian(n: int, m: int, corr_matrix: np.array):
    """ 
    # Gaussian Copula simulations with a given correlation matrix

    Parameters
    ----------
    n : int
        the dimension number of simulated variables
    m : int
        the sample size 
    corr_matrix : array
        the correlation matrix

    Returns
    -------
    u : array
        the simulated sample 

    """
    if not all(isinstance(v, int) for v in [n, m]):
        raise TypeError("The 'n' and 'm' arguments must both be integer types.")
    if not isinstance(corr_matrix, np.ndarray):
        raise TypeError("The 'corr_matrix' argument must be a numpy array.")
    # Generate n independent standard Gaussian random variables V = (v1 ,..., vn):
    v = [np.random.normal(0, 1, m) for i in range(0, n)]

    # Compute the lower triangular Cholesky factorization of the correlation matrix:
    l = linalg.cholesky(corr_matrix, lower=True)
    y = np.dot(l, v)
    u = norm.cdf(y, 0, 1)

    return u


def simu_tstudent(n: int, m: int, corr_matrix: np.array, nu: float):
    """
    # Student Copula with k degrees of freedom and a given correlation matrix

    Parameters
    ----------
    n : int
        the dimension number of simulated variables
    m : int
        the sample size 
    corr_matrix : array
        the correlation matrix
    nu : float
        the degree of freedom 

    Returns
    -------
    u : array
        the simulated sample 

    """
    if not all(isinstance(v, int) for v in [n, m]):
        raise TypeError("The 'n' and 'm' arguments must both be integer types.")
    if not isinstance(corr_matrix, np.ndarray):
        raise TypeError("The 'corr_matrix' argument must be a numpy array.")
    if not isinstance(nu, (int, float)):
        raise TypeError("The 'nu' argument must be a float type.")

    # Generate n independent standard Gaussian random variables  V = (v1 ,..., vn):
    v = [np.random.normal(0, 1, m) for i in range(0, n)]

    # Compute the lower triangular Cholesky factorization of rho:
    l = linalg.cholesky(corr_matrix, lower=True)
    z = np.dot(l, v)

    # generate a random variable r, following a chi2-distribution with nu degrees of freedom
    r = np.random.chisquare(df=nu,size=m)

    y = np.sqrt(nu/ r)*z
    u = t.cdf(y, df=nu, loc=0, scale=1)

    return u


def SimuSibuya(alpha: float, m: int):
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
    if not isinstance(alpha, (int, float)):
        raise TypeError("The 'alpha' argument must be a float type.")
    if not isinstance(m, int):
        raise TypeError("The 'm' argument must be an integer type.")

    G_1 = lambda y: ((1-y)*gamma(1-alpha))**(-1/alpha)
    F = lambda n: 1 - ((-1)**n)*comb(n, alpha-1)

    X = np.random.uniform(0, 1, m)

    for i in range(0, len(X)):
        if X[i] <= alpha:
            X[i] = 1
        elif F(np.floor(G_1(X[i]))) < X[i]:
            X[i] = np.ceil(G_1(X[i]))
        else:
            X[i] = np.floor(G_1(X[i]))

    return X


def simu_archimedean(family: str, n: int, m: int, theta: float):
    """
    Archimedean copula simulation

    Parameters
    ----------

    family : str
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
    if family not in ["clayton", "gumbel", "frank", "joe", "amh"]:
        raise ValueError("The family argument must be one of 'clayton', 'gumbel', 'frank', 'joe' or 'amh'.")
    if not all(isinstance(v, int) for v in [n, m]):
        raise TypeError("The 'n' and 'm' arguments must both be integer types.")
    if not isinstance(theta, (int, float)):
        raise TypeError("The 'theta' argument must be a float type.")

    if family == "clayton":
        # Generate n independent standard uniform random variables  V = (v1 ,..., vn):
        v = [np.random.uniform(0, 1, m) for i in range(0, n)]
        # generate a random variable x following the gamma distribution gamma(theta**(-1), 1)
        X = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0, m)])
        phi_t = lambda t:  (1+t)**(-1/theta)
        u = [phi_t(-np.log(v[i])/X) for i in range(0, n)]

    elif family == "gumbel":
        v = [np.random.uniform(0, 1, m) for i in range(0, n)]
        X = levy_stable.rvs(alpha=1/theta, beta=1, scale=(np.cos(np.pi/(2*theta)))**theta, loc=0, size=m)
        phi_t = lambda t:  np.exp(-t**(1/theta))
        u = [phi_t(-np.log(v[i])/X) for i in range(0, n)]

    elif family == "frank":
        v = [np.random.uniform(0, 1, m) for i in range(0, n)]
        p = 1-np.exp(-theta)
        X = logser.rvs(p, loc=0, size=m, random_state=None)
        phi_t = lambda t: -np.log(1-np.exp(-t)*(1-np.exp(-theta)))/theta
        u = [phi_t(-np.log(v[i])/X) for i in range(0, n)]
    
    elif family == "joe":
        alpha = 1/theta
        X = SimuSibuya(alpha, m)
        v = [np.random.uniform(0, 1, m) for i in range(0, n)]
        phi_t = lambda t: (1-(1-np.exp(-t))**(1/theta))
        u = [phi_t(-np.log(v[i])/X) for i in range(0, n)]

    elif family == "amh":
        v = [np.random.uniform(0, 1, m) for i in range(0, n)]
        X = np.random.geometric(p=1-theta, size=m)
        phi_t = lambda t:  (1-theta)/(np.exp(t)-theta)
        u = [phi_t(-np.log(v[i])/X) for i in range(0, n)]
    return u


def simu_mixture(n: int, m: int, combination: List[dict]):
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

        example:
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
    if not all(isinstance(v, int) for v in [n, m]):
        raise TypeError("The 'n' and 'm' arguments must both be integer types.")
    if not isinstance(combination, list):
        raise TypeError("The 'combination' argument must be a list type")
    if not all(isinstance(v, dict) for v in combination):
        raise TypeError("Each element of the 'combination' argument must be a dict type.")

    v = [np.random.uniform(0, 1, m) for i in range(0, n)]
    weights = [comb["weight"] for comb in combination]
    #Generate a random sample of indexes of combination types 
    y = np.array([np.where(ls == 1)[0][0] for ls in np.random.multinomial(n=1, pvals=weights, size=m)])

    for i in range(0, len(combination)):
        combinationsize = len(v[0][y == i])

        if combination[i]["type"] == "gaussian":
            corr_matrix = combination[i]["corrMatrix"]

            vi = simu_gaussian(n, combinationsize, corr_matrix)
            for j in range(0, len(vi)):
                    v[j][y == i] = vi[j]
        elif combination[i]["type"] == "student":
            corr_matrix = combination[i]["corrMatrix"]
            nu = combination[i]["nu"]
            vi = simu_tstudent(n, combinationsize, corr_matrix, nu)

        elif combination[i]["type"] in ["clayton", "gumbel", "frank", "joe", "amh"]:
            vi = simu_archimedean(combination[i]["type"], n, combinationsize, combination[i]["theta"])

            for j in range(0, len(vi)):
                    v[j][y == i] = vi[j]
        else:
            raise error

    return v

