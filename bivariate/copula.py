import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

pd.options.mode.chained_assignment = None  # default='warn'

def plot_bivariate(U,V,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U,V,Z)
    plt.show()


class Empirical():
    """
        Bivariate Empirical Copula 
        Takes a pandas Dataframe which first 2 columns are  

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

    def LTDC_(self, i_n):
        """
            Lower Tail Dependence Coefficient for a given threshold i/n

        """
        return self.cdf(i_n,i_n)/i_n

    def UTDC_(self, i_n):
        """
            Upper Tail Dependence Coefficient for a given threshold i/n

        """
        return ((1-2*i_n)+self.cdf(i_n,i_n))/(1-i_n)

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
    
    def optimal_tdc(self, case):
        """
            Returns optimal Empirical Tail Dependence coefficient (TDC)
            Based on the heuristic plateau-finding algorithm from Frahm et al (2005) "Estimating the tail-dependence coefficient: properties and pitfalls"

        """

        data = self.data #creates a copy 
        data.reset_index(inplace=True,drop=True)
        #data["TDC"] = np.NaN
        #data["TDC_smoothed"] = np.NaN
        n = len(self.data)
        b = int(n/200) # length to apply a moving average on 2b + 1 consecutive points
        # b is chosen such that ~1% of the data fall into the mooving average

        if case == "upper":
            # Compute the Upper TDC for every possible threshold i/n
            for i in range(1,n-1):
                data.at[i,"TDC"] = self.UTDC_(i_n=i/n)        
            data = data.iloc[::-1] # Reverse the order, the plateau finding algorithm starts with lower values
            data.reset_index(inplace=True,drop=True)

        elif case =="lower":
            # Compute the Lower TDC for every possible threshold i/n
            for i in range(1,n-1):
                data.at[i,"TDC"] = self.LTDC_(i_n=i/n)
        else:
            print("Takes \"upper\" or \"lower\" argument only")
            return None 

        # Smooth the TDC with a mooving average of lenght 2*b+1
        for i in range(b,n-b):
            TDC_smooth = 0
            for j in range(i-b,i+b+1):
                TDC_smooth = TDC_smooth +data.at[j,"TDC"]
            TDC_smooth = TDC_smooth/(2*b+1)  
            data.at[i,"TDC_smoothed"] = TDC_smooth

        m = int(np.sqrt(n-2*b) ) # lenght of the plateau
        std = data["TDC_smoothed"].std() # the standard deviation of the smoothed series
        data["cond"] = np.NaN # column that will contains the condition: plateau - 2*sigma

        for k in range(0,n-2*b-m+1):
            plateau = 0
            for i in range(k+1,k+m-1+1):
                plateau = plateau + np.abs(data.at[i,"TDC_smoothed"] - data.at[k,"TDC_smoothed"])
            data.at[k,"cond"] = plateau - 2*std

        # Finding the first k such that: plateau - 2*sigma <= 0
        k = data.loc[ data.cond <= 0,:].index[0]

        # The optimal TDC is defined as the average of the corresponding plateau
        plateau_k = 0
        for i in range(1,m-1+1):
            plateau_k = plateau_k + data.at[k+i-1,"TDC_smoothed"] 
        TDC_ = plateau_k/m

        print("Optimal threshold: ", k/n)
        return TDC_


class Archimedean():

    Archimedean_families = ['clayton', 'gumbel', 'frank', 'joe', 'galambos', 'fgm', 'plackett', 'rgumbel', 'rclayton', 'rjoe', 'rgalambos']
    
    def __init__(self, family):
        """
            Creates an Archimedean copula.
            bounds_param = 
            #x0 = Initial guess. Array of real elements of size (n,), where 'n' is the number of independent variables.

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
            returns the cumulative distribution function for the respective archimedean copula family
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
            Returns the lower tail dependence coefficient for a given theta
        """
        if self.family  in ['gumbel', 'joe', 'frank', 'galambos', 'fgm', 'plackett', 'rclayton']:
            return 0
        elif self.family  in ['rgalambos', 'clayton'] :
            return 2**(-1/theta)
        elif self.family  in ['rgumbel', 'rjoe'] :
            return 2-2**(1/theta)

    def UTD(self, theta):
        """
            Returns the upper tail dependence coefficient for a given theta
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

