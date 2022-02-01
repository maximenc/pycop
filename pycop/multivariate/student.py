from scipy.stats import t
from scipy.special import gamma
from scipy.stats import norm, multivariate_normal
import numpy as np


class student():

    def __init__(self, corrMatrix, nu):
        self.corrMatrix = np.asarray(corrMatrix)
        self.nu = nu
        self.n = len(corrMatrix)

        """
            The multivariate student distribution CDF has no analytic expression but it can be approximated numerically
        """

    def pdf(self, d):
        """
            returns the density
        """
        y = t.ppf(d, df=self.nu)
    
        rho_det = np.linalg.det(self.corrMatrix)
        rho_inv = np.linalg.inv(self.corrMatrix)

        A = gamma((self.nu+self.n)/2)*( gamma(self.nu/2)**(self.n-1) )
        B = gamma((self.nu+1)/2)**self.n
        C = (1 + (np.dot(y, np.dot(rho_inv, y)))/self.nu)**((self.nu+self.n)/2)

        [ (1 + yi**2/self.nu)**((self.nu+1)/2) for yi in y]

        prod = 1
        for comp in [ (1 + yi**2/self.nu)**((self.nu+1)/2) for yi in y]:
            prod *=comp

        return rho_det**(-0.5) *(A*prod)/(B*C)
