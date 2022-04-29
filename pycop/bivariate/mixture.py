from distutils.log import error
import numpy as np
from pycop.bivariate.copula import copula
from pycop.bivariate.archimedean import archimedean
from pycop.bivariate.gaussian import gaussian
from pycop.bivariate.student import student


class mixture(copula):
    """
    # Creates an mixture copula objects

    ...

    Attributes
    ----------
    dim : int
        The number of copula combined, only 2 or 3 supported
    mixture_type : str
        The type of mixture as bundle of the combinated copula
    cop : list
        A list that contains the copula objects to combine
    bounds_param : list
        A list that contains the domain of the parameter(s) in a tuple.
        Example : [(lower, upper), (lower, upper)]
    parameters_start : array 
        Value(s) of the initial guess when estimating the copula parameter(s).
        It represents the parameter `x0` in the `scipy.optimize.minimize` function.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the Cumulative Distribution Function (CDF).
    get_pdf(u, v, param)
        Computes the Probability Density Function (PDF).
    LTDC(w1, theta1)
        Computes the Lower Tail Dependence Coefficient (TDC).
    UTDC(w1, theta2)
        Computes the upper TDC.
    """

    def __init__(self, copula_list):
        """
        Parameters
        ----------
        copula_list : list
            A list of string of the type of copula to combined
            Example : ["clayton", "gumbel"]

        Raises
        ------
        ValueError
            dim : the lenght of `copula_list` must be equal to 2 or 3.
            Mixtures are only available for a combination of 2 or 3 copulas

            copula_list : element must be supported functions.
            Mixtures are only available for archimedean and gaussian.

        """
        # the `student` copula class inherit the `copula` class
        super().__init__()

        Archimedean_families = [
            'clayton', 'gumbel', 'frank', 'joe', 'galambos','fgm', 'plackett',
            'rgumbel', 'rclayton', 'rjoe','rgalambos']

        self.dim = len(copula_list)

        if self.dim != 2 and self.dim != 3:
            print("Mixture supported only for combinaison of 2 or 3 copulas")
            raise ValueError

        self.cop = []
        mixture_type = copula_list[0].capitalize()

        for cop in copula_list[1:]:
            mixture_type+= "-"+cop.capitalize()

        self.family = mixture_type + " mixture"
        if self.dim ==2:
            self.bounds_param = [(0,1)]
            self.parameters_start = [np.array(1/self.dim)]     
        else:
            self.bounds_param = [(0,1) for i in range(0, 3)]
            self.parameters_start = [np.array(1/self.dim) for i in range(0, 3)]    

        for i in range(0,self.dim):

            if copula_list[i] == "gaussian":
                self.cop.append(gaussian())
                self.bounds_param.append((-1,1))
                self.parameters_start.append(np.array(0))

            elif copula_list[i] in Archimedean_families:
                cop_mixt = archimedean(family=copula_list[i])
                self.cop.append(cop_mixt)
                self.bounds_param.append(cop_mixt.bounds_param[0])
                self.parameters_start.append(cop_mixt.parameters_start)
            else:
                print("Mixture only supported for archimedean and gaussian mixture only")
                print("Archimedean copula available are: ", Archimedean_families)
                raise ValueError
        self.parameters_start = tuple(self.parameters_start)

    def get_cdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        param : list
            A list that contains the parameters of the mixture and the copula.
            The element of the list must be ordered as follow, for 2-dimensional mixture :
                [
                    weight1 : float, weight1 ∈ [-1,1]
                        The weight given in the first copula.
                    theta1 : float
                        The theta parameter of the first copula.
                    theta2 : float
                        " second.
                ]
            For a 3-dimensional mixture :
                [
                    weight1 : float
                        the weight given in the first copula.
                    weight2 : float
                        " second.
                    weight3 : float
                        " third.
                    theta1 : float
                        The theta parameter of the first copula.
                    theta2 : float
                        " second.
                    theta3 : float
                        " third.
                ]
            The sum of the weights must be equal to 1.
        """
        if self.dim == 2:
            cdf = param[0]*(self.cop[0].get_cdf(u,v,[param[1]])) \
            +(1-param[0])*(self.cop[1].get_cdf(u,v,[param[2]]))
        else:
            cdf= param[0]*(self.cop[0].get_cdf(u,v,[param[3]])) \
            + param[1]*(self.cop[1].get_cdf(u,v,[param[4]])) \
            + param[2]*(self.cop[2].get_cdf(u,v,[param[5]]))
        return cdf

    def get_pdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs 
        param : list
            A list that contains the parameters of the mixture and the copula.
            See how to order the list in the above method `get_cdf`
        """
        if self.dim == 2:
            pdf = param[0]*(self.cop[0].get_pdf(u,v,[param[1]])) \
            +(1-param[0])*(self.cop[1].get_pdf(u,v,[param[2]]))
        else:
            pdf = param[0]*(self.cop[0].get_pdf(u,v,[param[3]])) \
            + param[1]*(self.cop[1].get_pdf(u,v,[param[4]])) \
            + param[2]*(self.cop[2].get_pdf(u,v,[param[5]]))

        return pdf

    def LTDC(self, w1, theta1):
        """
        # Computes the upper TDC

        Parameters
        ----------
        w1 : float
            The weight associated to the copula with Lower Tail Dependence 
        theta1 : float
            The parameter of the copula with Lower Tail Dependence
        """
        return self.cop1.LTDC(theta1)*w1

    def UTDC(self, w2, theta2):
        """
        # Computes the upper TDC

        Parameters
        ----------
        w2 : float
            The weight associated to the copula with Upper Tail Dependence 
        theta2 : float
            The parameter of the copula with Upper Tail Dependence
        """
        return self.cop2.UTDC(theta2)*w2


