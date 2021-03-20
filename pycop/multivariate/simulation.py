from scipy import linalg
from scipy.stats import norm, t, levy_stable, logser
import numpy as np


def SimuGaussian(n, m, corrMatrix):
    """ 
    # Gaussian Copula simulations with a given correlation matrix
    Requires:
        n = number of variables
        m = sample size
        corrMatrix = correlation matrix
    """

    # Generate n independent standard Gaussian random variables  U = (u1 ,..., un):
    v = [np.random.normal(0,1,m) for i in range(0,n)]

    # Compute the lower triangular Cholesky factorization of the correlation matrix:
    L = linalg.cholesky(corrMatrix, lower=True)
    y = np.dot(L, v)
    u = norm.cdf(y, 0, 1 )

    return u


def SimuStudent(n, m, corrMatrix, k):
    """
    # Student Copula with k degrees of freedom and a given correlation matrix
    Requires:
        n = number of variables
        m = sample size
        corrMatrix = correlation matrix
        k = degree of freedom 
    """

    # Generate n independent standard Gaussian random variables  U = (u1 ,..., un):
    v = [np.random.normal(0,1,m) for i in range(0,n)]

    # Compute the lower triangular Cholesky factorization of rho:
    L = linalg.cholesky(corrMatrix, lower=True)
    z = np.dot(L, v)

    # generate a random variable r, following a chi2-distribution with nu degrees of freedom
    r = np.random.chisquare(df=k,size=m)

    y = np.sqrt(k/ r)*z
    u = t.cdf(y, df=k, loc=0, scale=1)

    return u


def SimuClayton(n, m, theta):
    """
    # Clayton copula
    # Devroye, L. (1986) Non-uniform Random Variate Generation.
    Requires:
        n = number of variables
        m = sample size
        theta = Clayton copula parameter
    """

    # Generate n independent standard Gaussian random variables  U = (u1 ,..., un):
    v = [np.array([np.random.exponential(scale=1.0) for i in range(0,m)]) for j in range(0,n)]

    # generate a random variable x following the gamma distribution gamma(theta**(-1), 1)
    x = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0,m)])

    u = [(1 + vi/x)**(-1/theta) for vi in v]

    return u


def SimuGumbel(n, m, theta):
    """
    # Gumbel copula
    Requires:
        n = number of variables to generate
        m = sample size
        theta = Gumbel copula parameter
    """
    # https://cran.r-project.org/web/packages/gumbel/gumbel.pdf
    # https://cran.r-project.org/web/packages/gumbel/vignettes/gumbel.pdf

    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    X = levy_stable.rvs(alpha=1/theta, beta=1,scale=(np.cos(np.pi/(2*theta)))**theta,loc=0, size=m)

    phi_t = lambda t:  np.exp(-t**(1/theta))

    u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]
    return u


def SimuFrank(n, m, theta):
    """
    # Frank copula
    Requires:
        n = number of variables to generate
        m = sample size
        theta = Frank copula parameter
    """

    v = [np.random.uniform(0,1,m) for i in range(0,n)]
    p = 1-np.exp(-theta)
    X = logser.rvs(p, loc=0, size=m, random_state=None)

    phi_t = lambda t:  -np.log(1-np.exp(-t)*(1-np.exp(-theta)))/theta
    u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]
    return u


def SimuJoe(n, m, theta):
    """
    # Joe copula
    Requires:
        n = number of variables to generate
        m = sample size
        theta = Joe copula parameter
    """

    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    X = 

    phi_t = lambda t: (1-(1-np.exp(-t))**(1/theta))

    u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]

    return u

