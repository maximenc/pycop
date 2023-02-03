import numpy as np
from pycop.bivariate.copula import copula

class archimedean(copula):
    """
    # Creates an Archimedean copula objects
    Source for the CDF and PDF functions:
    Joe, H. (2014). Dependence modeling with copulas. CRC press.
    Chapter 4: Parametric copula families and properties (p.159)

    ...

    Attributes
    ----------
    family : str
        The name of the Archimedean copula function.
    type : str
        The type of copula = "archimedean".
    bounds_param : list
        A list that contains the domain of the parameter(s) in a tuple.
        Exemple : [(lower, upper)]
    parameters_start : array 
        Value(s) of the initial guess when estimating the copula parameter(s).
        It represents the parameter `x0` in the `scipy.optimize.minimize` function.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the Cumulative Distribution Function (CDF).
    get_pdf(u, v, param)
        Computes the Probability Density Function (PDF).
    LTDC(theta)
        Computes the Lower Tail Dependence Coefficient (TDC).
    UTDC(theta)
        Computes the upper TDC.
    """

    Archimedean_families = [
        'clayton', 'gumbel', 'frank', 'joe', 'galambos','fgm', 'plackett',
        'rgumbel', 'rclayton', 'rjoe','rgalambos', 'BB1', 'BB2']
    

    def __init__(self, family):
        """
        Parameters
        ----------
        family : str
            The name of the Archimedean copula function.

        Raises
        ------
        ValueError
            If the given `family` is not supported.
        """

        # the `archimedean` copula class inherit the `copula` class
        super().__init__()
        self.family = family
        self.type = "archimedean"

        if family  in ['clayton', 'galambos', 'plackett', 'rclayton', 'rgalambos'] :
            self.bounds_param = [(1e-6, None)]
            self.parameters_start = np.array(0.5)

        elif family in ['gumbel', 'joe', 'rgumbel', 'rjoe'] :
            self.bounds_param = [(1, None)]
            self.parameters_start = np.array(1.5)

        elif family == 'frank':
            self.bounds_param = [(None, None)]
            self.parameters_start = np.array(2)

        elif family == 'fgm':
            self.bounds_param = [(-1, 1-1e-6)]
            self.parameters_start = np.array(0)

        elif family  in ['BB1'] :
            self.bounds_param = [(1e-6, None), (1, None)]
            self.parameters_start = (np.array(.5), np.array(1.5))

        elif family  in ['BB2'] :
            self.bounds_param = [(1e-6, None), (1e-6, None)]
            self.parameters_start = (np.array(1), np.array(1))
        else:
            print("family \"%s\" not in list: %s" % (family, archimedean.Archimedean_families) )
            raise ValueError

    def get_cdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        param : list
            A list that contains the copula parameter(s) (float)
        """

        if self.family == 'clayton':
            return (u ** (-param[0]) + v ** (-param[0]) - 1) ** (-1 / param[0])

        elif self.family == 'rclayton':
            return (u + v - 1 + archimedean(family='clayton').get_cdf((1 - u),(1 - v), param) )

        elif self.family == 'gumbel':
            return np.exp(-((-np.log(u)) ** param[0] + (-np.log(v)) ** param[0] ) ** (1 / param[0]))

        elif self.family == 'rgumbel':
            return (u + v - 1 + archimedean(family='gumbel').get_cdf((1-u),(1-v), param) )

        elif self.family == 'frank':
            a = (np.exp(-param[0] * u) - 1) * (np.exp(-param[0] * v) - 1)
            return (-1 / param[0]) * np.log(1 + a / (np.exp(-param[0]) - 1))

        elif self.family == 'joe':
            u_ = (1 - u) ** param[0]
            v_ = (1 - v) ** param[0]
            return 1 - (u_ + v_ - u_ * v_) ** (1 / param[0])

        elif self.family == 'rjoe':
            return (u + v - 1 + archimedean(family='joe').get_cdf((1 - u),(1 - v), param) )

        elif self.family == 'galambos':
            return u * v * np.exp(((-np.log(u)) ** (-param[0]) + (-np.log(v)) ** (-param[0])) ** (-1 / param[0]) )

        elif self.family == 'rgalambos':
            return (u + v - 1 + archimedean(family='galambos').get_cdf((1 - u),(1 - v), param) )

        elif self.family == 'fgm':
            return u * v * (1 + param[0] * (1 - u) * (1 - v))

        elif self.family == 'plackett':
            eta = param[0] - 1
            term1 = 0.5 * eta ** -1
            term2 = 1 + eta * (u + v)
            term3 = (1 + eta * (u + v)) ** 2
            term4 = 4 * param[0] * eta * u * v
            return term1 * (term2 - (term3 - term4) ** 0.5)

        elif self.family == 'BB1':
            term1 = (u ** (-param[1]) - 1) ** param[0]
            term2 = (v ** (-param[1]) - 1) ** param[0]
            term3 = (1 + term1 + term2) ** (1 / param[0])
            return (term3) ** (-1 / param[1])

        elif self.family == 'BB2':
            u_ = np.exp(param[0] * (u ** (-param[1]) - 1))
            v_ = np.exp(param[0] * (v ** (-param[1]) - 1))
            return (1 + (1 / param[0]) * np.log(u_ + v_ - 1)) ** (-1 / param[1])
        

    def get_pdf(self, u, v, param):
        """
        # Computes the PDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        param : list
            A list that contains the copula parameter(s) (float)
        """

        if self.family == 'clayton':
            term1 = (param[0] + 1) * (u * v) ** (-param[0] - 1)
            term2 = (u ** (-param[0]) + v ** (-param[0]) - 1) ** (-2 - 1 / param[0])
            return term1 * term2
    
        if self.family == 'rclayton':
            return archimedean(family='clayton').get_pdf((1 - u),(1 - v), param)

        elif self.family == 'gumbel':
            term1 = np.power(np.multiply(u, v), -1)
            tmp = np.power(-np.log(u), param[0]) + np.power(-np.log(v), param[0])
            term2 = np.power(tmp, -2 + 2.0 / param[0])
            term3 = np.power(np.multiply(np.log(u), np.log(v)), param[0] - 1)
            term4 = 1 + (param[0] - 1) * np.power(tmp, -1 / param[0])
            return archimedean(family='gumbel').get_cdf(u,v, param) * term1 * term2 * term3 * term4

        if self.family == 'rgumbel':
            return archimedean(family='gumbel').get_pdf((1 - u), (1 - v), param)

        elif self.family == 'frank':
            term1 = param[0] * (1 - np.exp(-param[0])) * np.exp(-param[0] * (u + v))
            term2 = (1 - np.exp(-param[0]) - (1 - np.exp(-param[0] * u)) \
                    * (1 - np.exp(-param[0] * v))) ** 2
            return term1 / term2

        elif self.family == 'joe':
            u_ = (1 - u) ** param[0]
            v_ = (1 - v) ** param[0]
            term1 = (u_ + v_ - u_ * v_) ** (-2 + 1 / param[0])
            term2 = ((1 - u) ** (param[0] - 1)) * ((1 - v) ** (param[0] - 1))
            term3 = param[0] - 1 + u_ + v_ + u_ * v_
            return term1 * term2 * term3

        if self.family == 'rjoe':
            return archimedean(family='joe').get_pdf((1 - u),(1 - v), param)

        elif self.family == 'galambos':
            x = -np.log(u)
            y = -np.log(v)
            term1 = self.get_cdf(u, v, param) / (v * u)
            term2 = 1 - ((x ** (-param[0]) + y ** (-param[0])) ** (-1 - 1 / param[0])) \
                    * (x ** (-param[0] - 1) + y ** (-param[0] - 1))
            term3 = ((x ** (-param[0]) + y ** (-param[0])) ** (-2 - 1 / param[0])) \
                    * ((x * y) ** (-param[0] - 1))
            term4 = 1 + param[0] + ((x ** (-param[0]) + y ** (-param[0])) ** (-1 / param[0]))
            return term1 * term2 + term3 * term4

        if self.family == 'rgalambos':
            return archimedean(family='galambos').get_pdf((1 - u),(1 - v), param)

        elif self.family == 'fgm':
            return 1 + param[0] * (1 - 2 * u) * (1 - 2 * v)

        elif self.family == 'plackett':
            eta = (param[0] - 1)
            term1 = param[0] * (1 + eta * (u + v - 2 * u * v))
            term2 = (1 + eta * (u + v)) ** 2 
            term3 = 4 * param[0] * eta * u * v
            return term1 / (term2 - term3) ** (3 / 2)

        elif self.family == 'BB1':
            theta, delta = param[0], param[1]
            x = (u ** (-theta) - 1) ** (delta)
            y = (v ** (-theta) - 1) ** (delta)
            term1 = (1 + (x + y) ** (1 / delta)) ** (-1 / theta - 2)
            term2 = (x + y) ** (1 / delta - 2)
            term3 = theta * (delta - 1) + (theta * delta + 1) * (x + y) ** (1 / delta)
            term4 = (x * y) ** (1 - 1 / delta) * (u * v) ** (-theta - 1)
            return term1 * term2 * term3 * term4

        elif self.family == 'BB2':
            theta, delta = param[0], param[1]
            x = np.exp(delta * (u ** (-theta) )) - 1
            y = np.exp(delta * (v ** (-theta) )) - 1
            term1 = (1 + (delta ** (-1)) * np.log(x + y - 1)) ** (-2 -1 / theta)
            term2 = (x + y - 1) ** (-2)
            term3 = 1 + theta + theta * delta + theta * np.log(x + y - 1)
            term4 = x * y * (u * v) ** (-theta - 1)
            return term1 * term2 * term3 * term4

    def LTDC(self, theta):
        """
        # Computes the lower TDC for a given theta

        Parameters
        ----------
        theta : float
            The copula parameter
        """

        if self.family  in ['gumbel', 'joe', 'frank', 'galambos', 'fgm', 'plackett', 'rclayton']:
            return 0

        elif self.family  in ['rgalambos', 'clayton'] :
            return 2 ** (-1 / theta)

        elif self.family  in ['rgumbel', 'rjoe'] :
            return 2 - 2 ** (1 / theta)

    def UTDC(self, theta):
        """
        # Computes the upper TDC for a given theta

        Parameters
        ----------
        theta : float
            The copula parameter
        """

        if self.family  in ['clayton', 'frank', 'fgm', 'plackett', 'rgumbel', 'rjoe', 'rgalambos']:
            return 0

        elif self.family  in ['galambos', 'rclayton'] :
            return 2 ** (-1 / theta)

        elif self.family  in ['gumbel', 'joe'] :
            return 2 - 2 ** (1 / theta)
