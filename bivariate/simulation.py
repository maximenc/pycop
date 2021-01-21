import numpy as np
import matplotlib.pyplot as plt



###
# See  Genest and MacKay (1986) The joy of copulas: bivariate distributions with uniform marginals
### General algorithm to generate pairs of random variables whose distribution function is given by an Archimedean Copula

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

def simu_stable(alpha, beta, gamma, delta, num):
    # REF : Univariate Stable Distributions: Models for Heavy Tailed Data
    T = np.random.uniform(-np.pi/2,np.pi/2,num)
    W = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    T_0 = np.arctan(beta * np.tan(np.pi *alpha*0.5) )/alpha

    if alpha == 1:
        Z = 2/np.pi *( (2/np.pi + beta*T)*np.tan(T) - beta*np.log( ( 2/np.pi * W * np.cos(T) )/( 2/np.pi + beta*T) ) )

    else:
        Z = ( (np.sin(alpha)*(T_0+T)) /( (np.cos(alpha)*T_0*np.cos(T) )**(1/alpha) ) )  * ((np.cos(alpha*T_0+(1-alpha)*T)/W )**((1-alpha)/alpha))

    return Z

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
    X = simu_stable(alpha,beta,gamma,delta, num)

    v1 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    v2 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
    def phi_1(t): return np.exp(-t**(1/theta))

    u1 = phi_1(v1/X)
    u2 = phi_1(v2/X)

    return u1, u2


u1, u2 = simu_clayton(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

u1, u2 = simu_frank(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

u1, u2 = simu_gumbel(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

