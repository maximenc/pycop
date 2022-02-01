import numpy as np


class mix2Copula():
    """
        Creates a Copula from a mix of 2 families of Archimedean Copulas
    """
    # Mettre une condition sur la family1 avec lower tail dependence (clayton rgumbel etc)
    def __init__(self, family1, family2):
        self.family1 = family1
        self.family2 = family2
        self.cop1 = archimedean(family=family1)
        self.cop2 = archimedean(family=family2)
        self.bounds_param = ((0,1), self.cop1.bounds_param[0], self.cop2.bounds_param[0])
        self.theta_start = (np.array(0.5), self.cop1.theta_start, self.cop2.theta_start)

    def cdf(self, u, v, param):
        return param[0]*(self.cop1.cdf(u,v,param[1]))+(1-param[0])*(self.cop2.cdf(u,v,param[2]))

    def pdf(self, u, v,param):
        return param[0]*(self.cop1.pdf(u,v,param[1]))+(1-param[0])*(self.cop2.pdf(u,v,param[2]))

    def LTDC(self, w1, theta1):
        return self.cop1.LTDC(theta1)*w1

    def UTDC(self, w1, theta2):
        return self.cop2.UTDC(theta2)*(1-w1)

    def plot_pdf(self, param, Nsplit):
        U_grid = np.linspace(0, 1, Nsplit)
        V_grid = np.linspace(0, 1, Nsplit)
        U_grid, V_grid = np.meshgrid(U_grid, V_grid)
        Z = np.array( [self.pdf(uu, vv, param) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)

    def plot_cdf(self, param, Nsplit):
        U_grid = np.linspace(0, 1, Nsplit)
        V_grid = np.linspace(0, 1, Nsplit)
        U_grid, V_grid = np.meshgrid(U_grid, V_grid)
        Z = np.array( [self.cdf(uu, vv, param) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)

class mix3Copula():
    """
        Creates a Copula from a mix of 3 families of Archimedean Copulas
        Only family 1 should have lower tail dependence and only family 3 should have upper tail dependence
    """

    def __init__(self, family1, family2, family3):
        self.family1 = family1
        self.family2 = family2
        self.family3 = family3
        self.cop1 = archimedean(family=family1)
        self.cop2 = archimedean(family=family2)
        self.cop3 = archimedean(family=family3)

        self.bounds_param = ((0,1), (0,1), self.cop1.bounds_param[0], self.cop2.bounds_param[0], self.cop3.bounds_param[0])
        self.theta_start = (np.array(0.33), np.array(0.33), self.cop1.theta_start, self.cop2.theta_start, self.cop3.theta_start)

    def cdf(self, u, v, param):
        return param[0]*(self.cop1.cdf(u,v,param[2])) + param[1]*(self.cop2.cdf(u,v,param[3])) + (1-param[0]-param[1])*(self.cop3.cdf(u,v,param[4]))

    def pdf(self, u, v,param):
        return param[0]*(self.cop1.pdf(u,v,param[2])) + param[1]*(self.cop2.pdf(u,v,param[3])) + (1-param[0]-param[1])*(self.cop3.pdf(u,v,param[4]))

    def LTDC(self, w1, theta1):
        return self.cop1.LTD(theta1)*w1

    def UTDC(self, w3, theta3):
        return self.cop3.UTD(theta3)*w3
