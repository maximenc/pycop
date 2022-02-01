import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab

import matplotlib
from scipy.stats import norm

def plot_bivariate_3d(U,V,Z, bounds, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(np.linspace(bounds[0],bounds[1],6))
    ax.set_yticks(np.linspace(bounds[0],bounds[1],6))
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.plot_surface(U,V,Z, **kwargs)
    plt.show()

def plot_bivariate_contour(U,V,Z,bounds, lvls=None, **kwargs):
    plt.figure()
    CS = plt.contour(U, V, Z, levels=lvls, colors='k', linewidths=0.8, linestyles=None, **kwargs)
    plt.clabel(CS, fontsize=8, inline=1)
    plt.xlim(bounds)
    plt.ylim(bounds)
    plt.title('Single color - negative contours solid')
    plt.show()

class copula():
    
    def __init__(self):
        pass

    def plot_cdf(self, param, type,  Nsplit=50, **kwargs):

        bounds = [0+1e-2, 1-1e-2]
        U_grid, V_grid = np.meshgrid(np.linspace(bounds[0], bounds[1], Nsplit), np.linspace(bounds[0], bounds[1], Nsplit))
        Z = np.array( [self.cdf(uu, vv,  param) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)

        if type == "3d":
            plot_bivariate_3d(U_grid,V_grid,Z, [0,1], **kwargs)
        elif type == "contour":
            plot_bivariate_contour(U_grid,V_grid,Z, [0,1], **kwargs)
        else:
            print("only \"contour\" or \"3d\" arguments supported for type")
            raise ValueError

    def plot_pdf(self, param, type,  Nsplit=50, **kwargs):

        if type == "3d":
            bounds = [0+1e-1/2, 1-1e-1/2]

        elif type == "contour":
            bounds = [0+1e-2, 1-1e-2]

        U_grid, V_grid = np.meshgrid(np.linspace(bounds[0], bounds[1], Nsplit), np.linspace(bounds[0], bounds[1], Nsplit))
        Z = np.array( [self.pdf(uu, vv,  param) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)

        if type == "3d":

            plot_bivariate_3d(U_grid,V_grid,Z, [0,1], **kwargs)
        elif type == "contour":
            plot_bivariate_contour(U_grid,V_grid,Z, [0,1], **kwargs)
        else:
            print("only \"contour\" or \"3d\" arguments supported for type")
            raise ValueError


    def plot_mpdf(self, param, margin, type,  Nsplit=50, **kwargs):

        if margin[0] == "gaussian":
            univariate1 = norm

        if margin[1] == "gaussian":
            univariate2 = norm
        
        bounds = [-3, 3]

        U_grid, V_grid = np.meshgrid(np.linspace(bounds[0], bounds[1], Nsplit), np.linspace(bounds[0], bounds[1], Nsplit))
        Z = np.array( [self.pdf(univariate1.cdf(uu), univariate2.cdf(vv), param)*univariate1.pdf(uu,0,1)*univariate2.pdf(vv,0,1) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
        Z = Z.reshape(U_grid.shape)

        if type == "3d":
            plot_bivariate_3d(U_grid,V_grid,Z, bounds, **kwargs)
        elif type == "contour":
            plot_bivariate_contour(U_grid,V_grid,Z, bounds, **kwargs)
        else:
            print("only \"contour\" or \"3d\" arguments supported for type")
            raise ValueError

#matplotlib.rcParams['xtick.direction'] = 'out'
#matplotlib.rcParams['ytick.direction'] = 'out'



# You can set negative contours to be solid instead of dashed:
# matplotlib.rcParams['contour.negative_linestyle'] = 'solid'



