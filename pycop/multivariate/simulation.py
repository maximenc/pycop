import numpy as np
from scipy import linalg
from scipy.stats import norm, t, levy_stable, logser
from scipy.special import gamma, comb


def SimuGaussian(n, m, corrMatrix):
    """ 
    # Gaussian Copula simulations with a given correlation matrix
    Requires:
        n = number of variables
        m = sample size
        corrMatrix = correlation matrix
    """

    # Generate n independent standard Gaussian random variables V = (v1 ,..., vn):
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

    # Generate n independent standard Gaussian random variables  V = (v1 ,..., vn):
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
    Requires:
        n = number of variables
        m = sample size
        theta = Clayton copula parameter
    """

    # Generate n independent standard uniform random variables  V = (v1 ,..., vn):
    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    # generate a random variable x following the gamma distribution gamma(theta**(-1), 1)
    X = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0,m)])

    phi_t = lambda t:  (1+t)**(-1/theta)

    u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]

    return u


def SimuGumbel(n, m, theta):
    """
    # Gumbel copula
    Requires:
        n = number of variables to generate
        m = sample size
        theta = Gumbel copula parameter
    """

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

def SimuJoe(n, m, theta):
    """
    # Joe copula
    Requires:
        n = number of variables to generate
        m = sample size
        theta = Joe copula parameter
    """

    alpha = 1/theta
    X = SimuSibuya(alpha, m)

    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    phi_t = lambda t: (1-(1-np.exp(-t))**(1/theta))

    u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]

    return u


def SimuAMH(n, m, theta):
    """
    # Ali-Mikhail-Haq copula
    Requires:
        n = number of variables
        m = sample size
        theta = Ali-Mikhail-Haq copula parameter
    """

    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    X = np.random.geometric(p=1-theta, size=m)

    phi_t = lambda t:  (1-theta)/(np.exp(t)-theta)

    u = [phi_t(-np.log(v[i])/X) for i in range(0,n)]
    return u

from scipy.stats import bernoulli


combination = [
    {"type": "clayton", "weight": 0.33, "theta": 3},
    {"type": "student", "weight": 0.33, "corrMatrix": 1, "k": 2},
    {"type": "gumbel", "weight": 0.33, "theta": 3}
]

def SimuMx(n, m, combination):
    v = [np.random.uniform(0,1,m) for i in range(0,n)]

    weights = [comb["weight"] for comb in combination]

    #Generate m random sample of category labels 
    y = np.array([np.where(ls==1)[0][0] for ls in np.random.multinomial(n=1, pvals=weights, size=m)])

    for i in range(0,len(combination)):

        combinationsize = len(v[0][y==i])
        if combination[i]["type"] == "clayton":
            theta = combination[i]["theta"]
            X = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0,combinationsize)])
            phi_t = lambda t:  (1+t)**(-1/theta)
            for j in range(0,len(v)):
                v[j][y==i] = phi_t(-np.log(v[j][y==i])/X)       

        elif combination[i]["type"] == "gumbel":
            theta = combination[i]["theta"]
            X = levy_stable.rvs(alpha=1/theta, beta=1,scale=(np.cos(np.pi/(2*theta)))**theta,loc=0, size=combinationsize)
            phi_t = lambda t:  np.exp(-t**(1/theta))
            for j in range(0,len(v)):
                v[j][y==i] = phi_t(-np.log(v[j][y==i])/X)  

        elif combination[i]["type"] == "gaussian":
            corrMatrix = combination[i]["corrMatrix"]

            v2 = np.random.uniform(0,1,combinationsize)

            L = linalg.cholesky(corrMatrix, lower=True)

            for j in range(0,len(v)):
                v[j][y==i] = np.sqrt(-2*np.log(v[j][y==i]))*np.cos(2*np.pi*v2) 

            y = np.dot(L, [vk[y==i] for vk in v])
            Y = norm.cdf(y, loc=0, scale=1)

            for j in range(0,len(v)):
                v[j][y==i] = np.array(Y[j])
        
        elif combination[i]["type"] == "student":
            corrMatrix = combination[i]["corrMatrix"]
            k = combination[i]["k"]

            v2 = np.random.uniform(0,1,combinationsize)
            L = linalg.cholesky(corrMatrix, lower=True)
            r = np.random.chisquare(df=k,size=combinationsize)

            for j in range(0,len(v)):
                v[j][y==i] = np.sqrt(-2*np.log(v[j][y==i]))*np.cos(2*np.pi*v2) 

            z = np.dot(L, [vk[y==i] for vk in v])
            Y = t.cdf(np.sqrt(k/ r)*z, df=k, loc=0, scale=1)

            for j in range(0,len(v)):
                v[j][y==i] = np.array(Y[j])


    return v

