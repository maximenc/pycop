import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

def plot_bivariate(U,V,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U,V,Z)
    plt.show()


class Empirical():
    """
        Empirical copula 
        takes pandas Dataframe lenght x rows * 2 columns as Arguments 

    """
    def __init__(self, data):
        self.data = data
        self.n = len(data)

    def cdf(self, i_n, j_n):
        i = int(self.n * i_n)
        j = int(self.n * j_n)
        col = [self.data.columns.values[0], self.data.columns.values[1]]
        ith_order_u = np.partition(np.asarray(self.data[col[0]].values), i)[i]
        jth_order_v = np.partition(np.asarray(self.data[col[1]].values), j)[j]
        return (1/self.n) * len(self.data.loc[(self.data[col[0]] <= ith_order_u ) & (self.data[col[1]] <= jth_order_v) ])

    def LTD_(self, i0_n):
        return self.cdf(i0_n,i0_n)/i0_n

    def UTD_(self, i0_n):
        return ((1-2*i0_n)+self.cdf(i0_n,i0_n))/(1-i0_n)
       #df = df.fillna(0)

    def plot_cdf(self, Nsplit):
        U_grid = np.linspace(0, 1, Nsplit)[:-1]
        V_grid = np.linspace(0, 1, Nsplit)[:-1]
        U_grid, V_grid = np.meshgrid(U_grid, V_grid)
        Z = np.array( [self.cdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)


    def plot_pdf(self, Nsplit):

        U_grid = np.linspace(min(self.data.iloc[:,0]), max(self.data.iloc[:,0]), Nsplit)
        V_grid = np.linspace(min(self.data.iloc[:,1]), max(self.data.iloc[:,1]), Nsplit)

        df = pd.DataFrame(index=U_grid, columns=V_grid)
        df = df.fillna(0)
        for i in range(0,Nsplit-1):
            for j in range(0,Nsplit-1):
                Xa = U_grid[i]
                Xb = U_grid[i+1]
                Ya = V_grid[j]
                Yb = V_grid[j+1]
                df.at[Xa,Ya] = len(self.data.loc[( Xa <= self.data.iloc[:,0]) & (self.data.iloc[:,0] < Xb) & ( Ya <= self.data.iloc[:,1]) & (self.data.iloc[:,1]< Yb)])
        df.index=df.index+(U_grid[0]-U_grid[1])/2
        df.columns = df.columns+ (V_grid[0]-V_grid[1])/2 
        print(df)
        df = df /len(self.data)

        U, V = np.meshgrid(U_grid, V_grid)   # Create coordinate points of X and Y
        Z = np.array(df)

        plot_bivariate(U,V,Z)



class Archimedean():

    Archimedean_families = ['clayton', 'gumbel', 'frank', 'joe', 'galambos', 'fgm', 'plackett', 'rgumbel', 'rclayton', 'rjoe', 'rgalambos']
    
    def __init__(self, family):
        """
            Creates an Archimedean copula.
            bounds_param = 
            #x0 = Initial guess. Array of real elements of size (n,), where ‘n’ is the number of independent variables.
            theta_start #x0 = Initial guess. Array of real elements of size (n,), where ‘n’ is the number of independent variables.
        """
        self.family = family
        if family  in ['clayton', 'galambos', 'plackett', 'rclayton', 'rgalambos'] :
            self.bounds_param = [(0+ 1e-6, None)]
            self.theta_start = np.array(0.5)
        elif family in ['gumbel', 'joe', 'rgumbel', 'rjoe'] :
            self.bounds_param = [(1, None)]
            self.theta_start = np.array(1.5)
        elif family == 'frank':
            self.bounds_param = [(None)]
            self.theta_start = np.array(2)
        elif family == 'fgm':
            self.bounds_param = [(-1, 1-1e-6)]
            self.theta_start = np.array(0)

    def cdf(self, u, v, theta):
        """
            returns the cumulative distribution function for the respective archimedean copula
        """

        if self.family == 'clayton':
            return (u**(-theta)+v**(-theta)-1)**(-1/theta)

        elif self.family == 'rclayton':
            return (u + v - 1 + Archimedean(family='clayton').cdf((1-u),(1-v), theta) )

        elif self.family == 'gumbel':
            return np.exp(  -( (-np.log(u))**theta + (-np.log(v))**theta )**(1/theta) )

        elif self.family == 'rgumbel':
            return (u + v - 1 + Archimedean(family='gumbel').cdf((1-u),(1-v), theta) )

        elif self.family == 'frank':
            a = 1-np.exp(-theta)-(1-np.exp(-theta*u)*(1-np.exp(-theta*v) ) )
            return (-theta**(-1))*np.log(a/(1-np.exp(-theta)))

        elif self.family == 'joe':
            u_ = (1 - u)
            v_ = (1 - v)
            return 1-(u_**theta+v_**theta-(u_**theta)*(v_**theta))**(1/theta)

        elif self.family == 'rjoe':
            return (u + v - 1 + Archimedean(family='joe').cdf((1-u),(1-v), theta) )

        elif self.family == 'galambos':
            return u*v*np.exp(((-np.log(u))**(-theta)+(-np.log(v))**(-theta))**(-1/theta) )

        elif self.family == 'rgalambos':
            return (u + v - 1 + Archimedean(family='galambos').cdf((1-u),(1-v), theta) )

        elif self.family == 'fgm':
            return u*v*(1+theta*(1-u)*(1-v))

        elif self.family == 'plackett':
            eta = theta-1
            return 0.5*eta**(-1) * (1+eta*(u+v)-( (1+eta*(u+v) )**2 -4*theta*eta*u*v)**0.5 )

    def pdf(self, u, v, theta):
        """
            returns the density
        """

        if self.family == 'clayton':
            return ((theta+1)*(u*v)**(-theta-1))*((u**(-theta)+v**(-theta)-1)**(-2-1/theta))

        if self.family == 'rclayton':
            return Archimedean(family='clayton').pdf((1-u),(1-v), theta)

        elif self.family == 'gumbel':
            a = np.power(np.multiply(u, v), -1)
            tmp = np.power(-np.log(u), theta) + np.power(-np.log(v), theta)
            b = np.power(tmp, -2 + 2.0 / theta)
            c = np.power(np.multiply(np.log(u), np.log(v)), theta - 1)
            d = 1 + (theta - 1) * np.power(tmp, -1.0 / theta)
            return Archimedean(family='gumbel').cdf(u,v, theta) * a * b * c * d

        if self.family == 'rgumbel':
            return Archimedean(family='gumbel').pdf((1-u),(1-v), theta)

        elif self.family == 'frank':
            a = theta*(1-np.exp(-theta))*np.exp(-theta*(u+v))
            b = (1-np.exp(-theta)-(1-np.exp(-theta*u))*(1-np.exp(-theta*v)))**2
            return a/b

        elif self.family == 'joe':
            u_ = (1 - u)
            v_ = (1 - v)
            a = (u_**theta+v_**theta-(u_**theta*v_**theta))**(-2+1/theta)
            b = (u_**(theta-1)*u_**(theta-1))*(theta-1+u_**theta+v_**theta-(u_**theta*v_**theta))
            return a*b

        if self.family == 'rjoe':
            return Archimedean(family='joe').pdf((1-u),(1-v), theta)

        elif self.family == 'galambos':
            x = -np.log(u)
            y = -np.log(v)
            return (self.cdf(u,v,theta)/(v*u))*(1-((x**(-theta) +y**(-theta))**(-1-1/theta))*(x**(-theta-1) +y**(-theta-1))
            + ((x**(-theta) +y**(-theta))**(-2-1/theta))*((x*y)**(-theta-1))*(1+theta+(x**(-theta) +y**(-theta))**(-1/theta)))

        if self.family == 'rgalambos':
            return Archimedean(family='galambos').pdf((1-u),(1-v), theta)

        elif self.family == 'fgm':
            return 1+theta*(1-2*u)*(1-2*v)

        elif self.family == 'plackett':
            eta = (theta-1)
            a = theta*(1+eta*(u+v-2*u*v))
            b = ((1+eta*(u+v))**2-4*theta*eta*u*v)**(3/2)
            return a/b

    def LTD(self, theta):
        """
            returns the lower tail dependence coefficient for the given theta
        """
        if self.family  in ['gumbel', 'joe', 'frank', 'galambos', 'fgm', 'plackett', 'rclayton']:
            return 0
        elif self.family  in ['rgalambos', 'clayton'] :
            return 2**(-1/theta)
        elif self.family  in ['rgumbel', 'rjoe'] :
            return 2-2**(1/theta)

    def UTD(self, theta):
        """
            returns the upper tail dependence coefficient 
        """
        if self.family  in ['clayton', 'frank', 'fgm', 'plackett', 'rgumbel', 'rjoe', 'rgalambos']:
            return 0
        elif self.family  in ['galambos', 'rclayton'] :
            return 2**(-1/theta)
        elif self.family  in ['gumbel', 'joe'] :
            return 2-2**(1/theta)

    def plot_pdf(self, theta, Nsplit):
        U_grid = np.linspace(0, 1, Nsplit)
        V_grid = np.linspace(0, 1, Nsplit)
        U_grid, V_grid = np.meshgrid(U_grid, V_grid)
        Z = np.array( [self.pdf(uu, vv, theta) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)

    def plot_cdf(self, theta, Nsplit):
        U_grid = np.linspace(0, 1, Nsplit)
        V_grid = np.linspace(0, 1, Nsplit)
        U_grid, V_grid = np.meshgrid(U_grid, V_grid)
        Z = np.array( [self.cdf(uu, vv, theta) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)


class Mix2Copula():
    """
        Creates a Copula from a mix of 2 families of Archimedean Copulas
    """
    # Mettre une condition sur la family1 avec lower tail dependence (clayton rgumbel etc)
    def __init__(self, family1, family2):
        self.family1 = family1
        self.family2 = family2
        self.cop1 = Archimedean(family=family1)
        self.cop2 = Archimedean(family=family2)
        self.bounds_param = ((0,1), self.cop1.bounds_param[0], self.cop2.bounds_param[0])
        self.theta_start = (np.array(0.5), self.cop1.theta_start, self.cop2.theta_start)

    def cdf(self, u, v, param):
        return param[0]*(self.cop1.cdf(u,v,param[1]))+(1-param[0])*(self.cop2.cdf(u,v,param[2]))

    def pdf(self, u, v,param):
        return param[0]*(self.cop1.pdf(u,v,param[1]))+(1-param[0])*(self.cop2.pdf(u,v,param[2]))

    def LTD(self, w1, theta1):
        return self.cop1.LTD(theta1)*w1

    def UTD(self, w1, theta2):
        return self.cop2.UTD(theta2)*(1-w1)

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

class Mix3Copula():
    """
        Creates a Copula from a mix of 3 families of Archimedean Copulas
    """

    def __init__(self, family1, family2, family3):
        self.family1 = family1
        self.family2 = family2
        self.family3 = family3
        self.cop1 = Archimedean(family=family1)
        self.cop2 = Archimedean(family=family2)
        self.cop3 = Archimedean(family=family3)

        self.bounds_param = ((0,1), (0,1), self.cop1.bounds_param[0], self.cop2.bounds_param[0], self.cop3.bounds_param[0])
        self.theta_start = (np.array(0.33), np.array(0.33), self.cop1.theta_start, self.cop2.theta_start, self.cop3.theta_start)

    def cdf(self, u, v, param):
        return param[0]*(self.cop1.cdf(u,v,param[2])) + param[1]*(self.cop2.cdf(u,v,param[3])) + (1-param[0]-param[1])*(self.cop3.cdf(u,v,param[4]))

    def pdf(self, u, v,param):
        return param[0]*(self.cop1.pdf(u,v,param[2])) + param[1]*(self.cop2.pdf(u,v,param[3])) + (1-param[0]-param[1])*(self.cop3.pdf(u,v,param[4]))

    def LTD(self, w1, theta1):
        return self.cop1.LTD(theta1)*w1

    def UTD(self, w3, theta3):
        return self.cop3.UTD(theta3)*w3

