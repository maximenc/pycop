

from scipy.stats import t
from scipy.special import gamma
from scipy.stats import norm, multivariate_normal
import numpy as np


class gaussian():

    def __init__(self, corrMatrix):
        """
            Creates a gaussian copula

            corrMatrix: length determines the dimension of random variable

            the correlation matrix must be squared + symetric 
            and definite positive 

        """
        self.corrMatrix = np.asarray(corrMatrix)
        self.n = len(corrMatrix)

    def cdf(self, d):
        """
            returns the cumulative distribution
            d = (U1, ..., Un)

        """
        y = norm.ppf(d, 0, 1)

        return multivariate_normal.cdf(y, mean=None, cov=self.corrMatrix)

    def pdf(self, d):
        """
            returns the density
        """
        y = norm.ppf(d, 0, 1)
    
        rho_det = np.linalg.det(self.corrMatrix)
        rho_inv = np.linalg.inv(self.corrMatrix)

        return rho_det**(-0.5) * np.exp(-0.5 * np.dot(y, np.dot(rho_inv - np.identity(self.n), y))) 

