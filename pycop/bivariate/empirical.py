import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_bivariate(U,V,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U,V,Z)
    plt.show()

class empirical():
    """
    # A class used to create an Empirical copula object 

    ...

    Attributes
    ----------
    data : pandas dataframe
        A dataframe with two columns that corresponds to the observations
    n : int
        The number of observations

    Methods
    -------
    cdf(i_n, j_n)
        Compute the empirical Cumulative Distribution Function (CDF)
    pdf(i_n, j_n)
        Compute the empirical Probability Density Function (PDF)
    plot_cdf(Nsplit)
        Plot the empirical CDF
    plot_pdf(Nsplit)
        Plot the empirical PDF
    LTDC(i_n)
        Compute the lower Tail Dependence Coefficient (TDC) for a given threshold i/n
    UTDC(i_n)
        Compute the upper Tail Dependence Coefficient (TDC) for a given threshold i/n
    optimal_tdc(case)
        Compute the lower or upper TDC accoring to Frahm et al (2005) algorithm.
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : pandas dataframe
            A dataframe with two columns that corresponds to the observations.
        """
        self.data = data
        self.n = len(data)
        

    def cdf(self, i_n, j_n):
        """
        # Compute the CDF

        Parameters
        ----------
        i_n : float
            The threshold to compute the univariate distribution for the first vector.
        j_n : float
            The threshold to compute the univariate distribution for the second vector.
        """

        i = int(round(self.n * i_n))
        j = int(round(self.n * j_n))
        ith_order_u = sorted(np.asarray(self.data.iloc[:,0].values))[i-1]
        jth_order_v = sorted(np.asarray(self.data.iloc[:,1].values))[j-1]
        return (1/self.n) * len(self.data.loc[(self.data.iloc[:,0] <= ith_order_u ) & (self.data.iloc[:,1] <= jth_order_v) ])

    def LTDC(self, i_n):
        """
        # Compute the empirical lower TDC for a given threshold i/n

        Parameters
        ----------
        i_n : float
            The threshold to compute the TDC
        """

        i = int(round(self.n * i_n))

        if i == 0:
            return 0
        return self.cdf(i_n,i_n)/(i/self.n)

    def UTDC(self, i_n):
        """
        # Compute the empirical upper TDC for a given threshold i/n

        Parameters
        ----------
        i_n : float
            The threshold to compute the TDC
        """

        i = int(round(self.n * i_n))
        if i == 0:
            return 0
        return (1- 2*(i/self.n) + self.cdf(i_n,i_n) ) / (1-(i/self.n))

    def plot_cdf(self, Nsplit):
        """
        # Plot the empirical CDF

        Parameters
        ----------
        Nsplit : The number of splits used to compute the grid
        """
        U_grid = np.linspace(0, 1, Nsplit)[:-1]
        V_grid = np.linspace(0, 1, Nsplit)[:-1]
        U_grid, V_grid = np.meshgrid(U_grid, V_grid)
        Z = np.array( 
            [self.cdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)

    def plot_pdf(self, Nsplit):
        """
        # Plot the empirical PDF
        
        Parameters
        ----------
        Nsplit : The number of splits used to compute the grid
        """

        U_grid = np.linspace(
            min(self.data.iloc[:,0]),
            max(self.data.iloc[:,0]),
            Nsplit)
        V_grid = np.linspace(
            min(self.data.iloc[:,1]),
            max(self.data.iloc[:,1]),
            Nsplit)

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
        df = df /len(self.data)

        U, V = np.meshgrid(U_grid, V_grid)   # Create coordinate points of X and Y
        Z = np.array(df)

        plot_bivariate(U,V,Z)
    
    def optimal_tdc(self, case):
        """
        # Compute the optimal Empirical Tail Dependence coefficient (TDC)
        
        The algorithm is based on the heuristic plateau-finding algorithm 
        from Frahm et al (2005) "Estimating the tail-dependence coefficient:
        properties and pitfalls"

        Parameters
        ----------
        case: str
            takes "upper" or "lower" for upper TDC or lower TDC
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
                data.at[i,"TDC"] = self.UTDC(i_n=i/n)        
            data = data.iloc[::-1] # Reverse the order, the plateau finding algorithm starts with lower values
            data.reset_index(inplace=True,drop=True)

        elif case =="lower":
            # Compute the Lower TDC for every possible threshold i/n
            for i in range(1,n-1):
                data.at[i,"TDC"] = self.LTDC(i_n=i/n)
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
        #data["cond"] = np.NaN # column that will contains the condition: plateau - 2*sigma

        for k in range(0,n-2*b-m+1): #change the range from 0 ?
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