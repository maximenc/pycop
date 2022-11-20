import numpy as np
import matplotlib.pyplot as plt


def plot_bivariate_3d(X, Y, Z, bounds, title, **kwargs):
    """
    # Plot the 3D surface

    Parameters
    ----------
    X, Y, Z : array
        Positions of data points.
    bounds : list
        A list that contains the `xlim` and `ylim` of the graph.
    title : str
        A string for the title of the plot.
        
    **kwargs
        Additional keyword arguments passed to `ax.plot_surface`.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(np.linspace(bounds[0],bounds[1],6))
    ax.set_yticks(np.linspace(bounds[0],bounds[1],6))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.plot_surface(X,Y,Z, **kwargs)
    plt.title(title)
    plt.show()

def plot_bivariate_contour(X, Y, Z, bounds, title, **kwargs):
    """
    # Plot the contour surface

    Parameters
    ----------
    X, Y, Z : array
        Positions of data points.
    bounds : list
        A list that contains the `xlim` and `ylim` of the graph.
    title : str
        A string for the title of the plot.
    
    **kwargs
        Additional keyword arguments passed to `plt.contour`.
    """
    plt.figure()
    CS = plt.contour(X, Y, Z, colors='k', linewidths=1., linestyles=None, **kwargs)
    plt.clabel(CS, fontsize=8, inline=1)
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.title(title)
    plt.show()

class copula():
    """
    # A class used to create a Copula object 

    Set attributes and methods common to all copula objects (elliptical and Archimedean).

    ...

    Attributes
    ----------

    Methods
    -------
    plot_cdf(param, plot_type,  Nsplit=50, **kwargs)
        Plot the bivariate Cumulative Distribution Function (CDF)
    plot_pdf(param, plot_type,  Nsplit=50, **kwargs)
        Plot the Probability Density Function (PDF)
    plot_mpdf(param, margin, plot_type,  Nsplit=50, **kwargs)
        Plot the PDF with given marginal distributions
    """
    
    def __init__(self):
        pass

    def plot_cdf(self, param, plot_type, Nsplit=50, **kwargs):
        """
        # Plot the bivariate CDF

        Parameters
        ----------
        param : list
            A list of the copula parameter(s)
        plot_type : str
            The type of the plot either "3d" or "contour"
        Nsplit : int, optional
            The number of points plotted (Nsplit*Nsplit) (default is 50)

        **kwargs
            Additional keyword arguments passed to either `plot_bivariate_3d` or
            `plot_bivariate_contour`.
            Examples :
            - `colormap` can be passed in to change the default color 
            of the 3d plot.
            - `levels` can be passed to determine the positions of the contour lines.

        """
        title = self.family.capitalize() + " Copula CDF" 

        bounds = [0+1e-2, 1-1e-2]
        U_grid, V_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], Nsplit),
            np.linspace(bounds[0], bounds[1], Nsplit))
    
        Z = np.array(
            [self.get_cdf(uu, vv,  param) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        
        Z = Z.reshape(U_grid.shape)

        if plot_type == "3d":
            plot_bivariate_3d(U_grid,V_grid,Z, [0,1], title, **kwargs)
        elif plot_type == "contour":
            plot_bivariate_contour(U_grid,V_grid,Z, [0,1], title, **kwargs)
        else:
            print("only \"contour\" or \"3d\" arguments supported for type")
            raise ValueError

    def plot_pdf(self, param, plot_type, Nsplit=50, **kwargs):
        """
        # Plot the bivariate PDF

        Parameters
        ----------
        param : list
            A list of the copula parameter(s)
        plot_type : str
            The type of the plot either "3d" or "contour"
        Nsplit : int, optional
            The number of points plotted (Nsplit*Nsplit) (default is 50)

        **kwargs
            Additional keyword arguments passed to either `plot_bivariate_3d` or
            `plot_bivariate_contour`.
            Examples :
            - `colormap` can be passed in to change the default color 
            of the 3d plot.
            - `levels` can be passed to determine the positions of the contour lines.
        """

        title = self.family.capitalize() + " Copula PDF" 

        if plot_type == "3d":
            bounds = [0+1e-1/2, 1-1e-1/2]

        elif plot_type == "contour":
            bounds = [0+1e-2, 1-1e-2]

        U_grid, V_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], Nsplit),
            np.linspace(bounds[0], bounds[1], Nsplit))
        
        Z = np.array( 
            [self.get_pdf(uu, vv,  param) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
       
        Z = Z.reshape(U_grid.shape)

        if plot_type == "3d":

            plot_bivariate_3d(U_grid,V_grid,Z, [0,1], title, **kwargs)
        elif plot_type == "contour":
            plot_bivariate_contour(U_grid,V_grid,Z, [0,1], title, **kwargs)
        else:
            print("only \"contour\" or \"3d\" arguments supported for type")
            raise ValueError


    def plot_mpdf(self, param, margin, plot_type, Nsplit=50, **kwargs):
        """
        # Plot the bivariate PDF with given marginal distributions.
        
        The method supports only scipy distribution with `loc` and `scale`
        parameters as marginals.

        Parameters
        ----------
        - param : list
            A list of the copula parameter(s)
        - margin : list
            A list of dictionaries that contains the scipy distribution and 
            the location and scale parameters. 

            Examples :
            marginals = [
                {
                    "distribution": norm, "loc" : 0, "scale" : 1,
                },
                {
                    "distribution": norm, "loc" : 0, "scale": 1,
                }]
        - plot_type : str
            The type of the plot either "3d" or "contour"
        - Nsplit : int, optional
            The number of points plotted (Nsplit*Nsplit) (default is 50)
            
        **kwargs
            Additional keyword arguments passed to either `plot_bivariate_3d` or
            `plot_bivariate_contour`.
            exemples :
            - `colormap` can be passed in to change the default color 
            of the 3d plot.
            - `levels` can be passed to determine the positions of the contour lines.

        """
        
        title = self.family.capitalize() + " Copula PDF" 

        # We retrieve the univariate marginal distribution from the list
        univariate1 = margin[0]["distribution"]
        univariate2 = margin[1]["distribution"]
        
        bounds = [-3, 3]

        U_grid, V_grid = np.meshgrid(
            np.linspace(bounds[0], bounds[1], Nsplit),
            np.linspace(bounds[0], bounds[1], Nsplit))
    
        mpdf = lambda uu, vv : self.get_pdf(
            univariate1.cdf(uu, margin[0]["loc"], margin[0]["scale"]), \
            univariate2.cdf(vv, margin[1]["loc"], margin[1]["scale"]), param) \
            * univariate1.pdf(uu, margin[0]["loc"], margin[0]["scale"]) \
            * univariate2.pdf(vv, margin[1]["loc"], margin[1]["scale"])

        Z = np.array(
            [mpdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)

        if plot_type == "3d":
            plot_bivariate_3d(U_grid,V_grid,Z, bounds, title, **kwargs)
        elif plot_type == "contour":
            plot_bivariate_contour(U_grid,V_grid,Z, bounds, title, **kwargs)
        else:
            print("only \"contour\" or \"3d\" arguments supported for type")
            raise ValueError