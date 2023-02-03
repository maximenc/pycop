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
    bounds_param : list
        A list that contains the domain of the parameter(s) in a tuple.
        Exemple : [(lower, upper)]
    parameters_start : array 
        Value(s) of the initial guess when estimating the copula parameter(s).
        It represents the parameter `x0` in the `scipy.optimize.minimize` function.

    Methods
    -------

    get_pdf(u, v, rho)
        Computes the Probability Density Function (PDF).
    """

    def __init__(self):
        # the `student` copula class inherit the `copula` class
        super().__init__()
        self.family = "student"
        self.bounds_param = [(-1+1e-6, 1-1e-6), (1e-6, None)]
        self.parameters_start = (np.array(0), np.array(1))

    def get_pdf(self, u, v, param):
        """
        # Computes the PDF

        # Source:
        Joe, H. (2014). Dependence modeling with copulas. CRC press.
        4.13 Multivariate t - Student's Copula p.181 Equation (4.32)

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

        term1 = gamma((nu + 2) / 2) * gamma(nu / 2) 
        term2 = gamma((nu + 1) / 2) ** 2

        u_ = t.ppf(u, df=nu)
        v_ = t.ppf(v, df=nu)

        det_rho = 1-rho**2
        multid = (-2 * u_ * v_ * rho + (u_ ** 2) + (v_ ** 2) ) / det_rho
        term3 = (1 + multid / nu) ** ((nu + 2) / 2)

        prod1 = (1 + (u_ ** 2) / nu) ** ((nu + 1) / 2) 
        prod2 = (1 + (v_ ** 2) / nu) ** ((nu + 1) / 2)
        prod = prod1 * prod2

        return (1/np.sqrt(det_rho)) * (term1 * prod) / (term2 * term3)

