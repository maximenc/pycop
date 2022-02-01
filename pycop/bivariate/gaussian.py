from scipy.stats import norm, multivariate_normal
import numpy as np

from copula import copula


class gaussian(copula):

    def __init__(self):
        """
            Creates a gaussian copula

            corrMatrix: length determines the dimension of random variable

            the correlation matrix must be squared + symetric 
            and definite positive 

        """
        super().__init__()


    def cdf(self, u, v, rho):
        """
            returns the cumulative distribution
            d = (U1, ..., Un)

        """
        y1 = norm.ppf(u, 0, 1)
        y2 = norm.ppf(v, 0, 1)

        return multivariate_normal.cdf((y1,y2), mean=None, cov=[[1,rho],[rho,1]])

    def pdf(self, u, v, rho):
        """
            returns the density
        """
        y1 = norm.ppf(u, 0, 1)
        y2 = norm.ppf(v, 0, 1)

        return multivariate_normal.pdf((y1,y2), mean=None, cov=[[1,rho],[rho,1]])/(norm.pdf(y1)*norm.pdf(y2))


