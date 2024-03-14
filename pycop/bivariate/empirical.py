import numpy as np
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
    data : numpy array
        A numpy array with two vectors that corresponds to the observations
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
        data : numpy array
        """
        self.data = data
        self.n = len(data[0])

    def get_cdf(self, i_n, j_n):
        """
        # Compute the CDF

        Parameters
        ----------
        i_n : float
            The threshold to compute the univariate distribution for the first vector.
        j_n : float
            The threshold to compute the univariate distribution for the second vector.
        """

        # Calculate rank indices for both vectors
        i = int(round(self.n * i_n))
        j = int(round(self.n * j_n))

        ith_order_u = sorted(self.data[0])[i-1]
        ith_order_v = sorted(self.data[1])[j-1]

        # Find indices where both vectors are less than the corresponding rank indices
        mask_x = self.data[0] <= ith_order_u
        mask_y = self.data[1] <= ith_order_v

        return np.sum(np.logical_and(mask_x, mask_y)) / self.n

    def LTDC(self, i_n):
        """
        # Compute the empirical lower TDC for a given threshold i/n

        Parameters
        ----------
        i_n : float
            The threshold to compute the lower TDC.
        """
        
        if int(round(self.n * i_n)) == 0:
            return 0
        
        return self.get_cdf(i_n, i_n) / i_n

    def UTDC(self, i_n):
        """
        # Compute the empirical upper TDC for a given threshold i/n

        Parameters
        ----------
        i_n : float
            The threshold to compute the upper TDC.
        """
        
        return (1 - 2 * i_n + self.get_cdf(i_n, i_n) ) / (1-i_n)


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
            [self.get_cdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)
        plot_bivariate(U_grid,V_grid,Z)

    def plot_pdf(self, Nsplit):
        """
        # Plot the empirical PDF
        
        Parameters
        ----------
        Nsplit : The number of splits used to compute the grid
        """

        U_grid = np.linspace(self.data[0].min(), self.data[0].max(), Nsplit)
        V_grid = np.linspace(self.data[1].min(), self.data[1].max(), Nsplit)

        # Initialize a matrix to hold the counts
        counts = np.zeros((Nsplit, Nsplit))

        for i in range(Nsplit-1):
            for j in range(Nsplit-1):
                # Define the edges of the bin
                Xa, Xb = U_grid[i], U_grid[i + 1]
                Ya, Yb = V_grid[j], V_grid[j + 1]

                # Use boolean indexing to count points within the bin
                mask = (Xa <= self.data[0]) & (self.data[0] < Xb) & (Ya <= self.data[1]) & (self.data[1] < Yb)
                counts[i, j] = np.sum(mask)
        # Adjust the grid centers for plotting
        U_grid_centered = U_grid + (U_grid[1] - U_grid[0]) / 2
        V_grid_centered = V_grid + (V_grid[1] - V_grid[0]) / 2
        
        U, V = np.meshgrid(U_grid_centered, V_grid_centered)   # Create coordinate points of X and Y
        Z = counts / np.sum(counts)

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

        #### 1) The series of TDC is smoothed using a box kernel with bandwidth b ∈ N
        # Consists in applying a moving average on 2b + 1 consecutive points
        tdc_array = np.zeros((self.n,))
        
        # b is chosen such that ~1% of the data fall into the mooving average
        b = int(np.ceil(self.n/200)) 
        
        if case == "upper":
            # Compute the Upper TDC for every possible threshold i/n
            for i in range(1, self.n-1):
                tdc_array[i] = self.UTDC(i_n=i/self.n)
            # We reverse the order, the plateau finding algorithm starts with lower values
            tdc_array = tdc_array[::-1]

        elif case =="lower":
            # Compute the Lower TDC for every possible threshold i/n
            for i in range(1, self.n-1):
                tdc_array[i] = self.LTDC(i_n=i/self.n)
        else:
            print("Takes \"upper\" or \"lower\" argument only")
            return None 

        # Smooth the TDC with a mooving average of lenght 2b+1
        # The total lenght = n-2b-1 because i ∈ [1, n−2b]
        tdc_smooth_array = np.zeros((self.n-2*b-1,))
        
        for i, j in zip(range(b+1, self.n-b), range(0, self.n-2*b-1)):
            tdc_smooth_array[j] = sum(tdc_array[i-b:i+b+1]) / (2*b+1)  

        #### 2) We select a vector of m consecutive estimates that satisfies a plateau condition
        
        # m = lenght of the plateau = number of consecutive smoothed TDC estimates
        m = int(np.floor(np.sqrt(self.n-2*b))) 
        # The standard deviation of the smoothed TDC series
        std_tdc_smooth = tdc_smooth_array.std() 

        # We iterate k from 0 to n-2b-m because k ∈ [1, n−2b−m+1]
        for k in range(0,self.n-2*b-m): 
            plateau = 0
            for i in range(1,m-1):
                plateau = plateau + np.abs(tdc_smooth_array[k+i] - tdc_smooth_array[k])
            # if the plateau satisfies the following condition:
            if plateau <= 2*std_tdc_smooth:
                #### 3) Then, the TDC estimator is defined as the average of the estimators in the corresponding plateau
                avg_tdc_plateau = np.mean(tdc_smooth_array[k:k+m-1])
                print("Optimal threshold: ", k/self.n)
                return avg_tdc_plateau
        
        # In case the condition is not satisfied the TDC estimate is set to zero
        return 0