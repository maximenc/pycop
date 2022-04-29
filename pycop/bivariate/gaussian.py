from scipy.stats import norm, multivariate_normal
from pycop.bivariate.copula import copula
class gaussian(copula):
    """
    # Creates a gaussian copula object

    ...

    Attributes
    ----------
    family : str
        = "gaussian"

    Methods
    -------
    get_cdf(u, v, rho)
        Computes the Cumulative Distribution Function (CDF).
    get_pdf(u, v, rho)
        Computes the Probability Density Function (PDF).
    """

    def __init__(self):
        # the `gaussian` copula class inherit the `copula` class
        super().__init__()
        self.family = "gaussian"

    def get_cdf(self, u, v, rho):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        rho : float
            The correlation coefficient rho ∈ [-1,1].
            Used to defined the correlation matrix (squared, symetric and definite positive)
        """

        y1 = norm.ppf(u, 0, 1)
        y2 = norm.ppf(v, 0, 1)

        return multivariate_normal.cdf((y1,y2), mean=None, cov=[[1,rho[0]],[rho[0],1]])

    def get_pdf(self, u, v, rho):
        """
        # Computes the PDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        rho : float
            The correlation coefficient rho ∈ [-1,1].
            Used to defined the correlation matrix (squared, symetric and definite positive)
        """

        y1 = norm.ppf(u, 0, 1)
        y2 = norm.ppf(v, 0, 1)

        return multivariate_normal.pdf((y1,y2), mean=None, cov=[[1,rho[0]],[rho[0],1]])/(norm.pdf(y1)*norm.pdf(y2))


