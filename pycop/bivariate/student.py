from scipy.stats import t
from scipy.special import gamma
import numpy as np

from copula import copula



class student(copula):

    def __init__(self):
        super().__init__()

        """
            The multivariate student distribution CDF has no analytic expression but it can be approximated numerically
        """

    def pdf(self, u, v, param):
        """
            returns the density
        """
        rho = param[0]
        nu = param[1]
        y = t.ppf((u, v), df=nu)
    
        rho_det = np.linalg.det([[1,rho],[rho,1]])
        rho_inv = np.linalg.inv([[1,rho],[rho,1]])

        A = gamma((nu+2)/2)*( gamma(nu/2) )
        B = gamma((nu+1)/2)**2
        C = (1 + (np.dot(y, np.dot(rho_inv, y)))/nu)**((nu+2)/2)

        prod = 1
        for comp in [ (1 + yi**2/nu)**((nu+1)/2) for yi in y]:
            prod *=comp
        return rho_det**(-0.5) *(A*prod)/(B*C)


