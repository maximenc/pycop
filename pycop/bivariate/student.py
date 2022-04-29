import numpy as np
from scipy.stats import t
from scipy.special import gamma
from pycop.bivariate.copula import copula
class student(copula):
    """
    # Creates a student copula object

    The multivariate student CDF has no analytic expression but it can be
    approximated numerically

    ...

    Attributes
    ----------
    family : str
        = "student"

    Methods
    -------

    get_pdf(u, v, rho)
        Computes the Probability Density Function (PDF).
    """

    
    def __init__(self):
        # the `student` copula class inherit the `copula` class
        super().__init__()
        self.family = "student"

    def get_pdf(self, u, v, param):
        """
        # Computes the PDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        param : list
            A list that contains the correlation coefficient rho ∈ [-1,1] and
            nu > 0, the degrees of freedom.
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


