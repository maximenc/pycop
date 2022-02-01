import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab


import matplotlib

def plot_bivariate(U,V,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U,V,Z)
    plt.show()



def plot_pdf(cop, Nsplit=25):
    U_grid = np.linspace(1e-1, 1-1e-1, Nsplit)
    V_grid = np.linspace(1e-1, 1-1e-1, Nsplit) # bounded ]0,1[
    #U_grid = np.linspace(-3, 3, Nsplit)
    #V_grid = np.linspace(-3, 3, Nsplit) # bounded ]0,1[
    U_grid, V_grid = np.meshgrid(U_grid, V_grid)

    Z = np.array( [cop.pdf((norm.cdf(uu), norm.cdf(vv)))*norm.pdf(uu,0,1)*norm.pdf(vv,0,1) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
    #Z = np.array( [cop.pdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

    Z = Z.reshape(U_grid.shape)
    plot_bivariate(U_grid,V_grid,Z)

def plot_cdf(cop, Nsplit=25):
    U_grid = np.linspace(1e-1, 1-1e-1, Nsplit)
    V_grid = np.linspace(1e-1, 1-1e-1, Nsplit) # bounded ]0,1[
    U_grid, V_grid = np.meshgrid(U_grid, V_grid)
    Z = np.array( [cop.cdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

    Z = Z.reshape(U_grid.shape)

    plot_bivariate(U_grid,V_grid,Z)




def plot_contour(cop, lvls=None):

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    Nsplit = 50
    U_grid = np.linspace(0, 1, Nsplit)
    V_grid = np.linspace(0, 1, Nsplit)
    U_grid, V_grid = np.meshgrid(U_grid, V_grid)
    #Z = np.array( [cop.pdf((norm.cdf(uu), norm.cdf(vv)))*norm.pdf(uu,0,1)*norm.pdf(vv,0,1) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
    #Z = np.array( [cop.cdf((uu, vv)) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )
    Z = np.array( [cop.pdf(uu, vv) for uu, vv in zip(np.ravel(U_grid), np.ravel(V_grid)) ] )

    Z = Z.reshape(U_grid.shape)


    # You can set negative contours to be solid instead of dashed:
    # matplotlib.rcParams['contour.negative_linestyle'] = 'solid'


    plt.figure()
    CS = plt.contour(U_grid, V_grid, Z, levels=lvls,colors='k', linewidths=0.8, linestyles=None)
    plt.clabel(CS, fontsize=8, inline=1)
    plt.title('Single color - negative contours solid')
    plt.show()







"""
#################################################################################
########################## PLOT EMPIRICAL DENSITY ###############################
N = 30
U = np.linspace(0, 1, N)
V = np.linspace(0, 1, N)

df = pd.DataFrame(index=U, columns=V)
df = df.fillna(0)

for i in range(0,N-1):
    for j in range(0,N-1):
        Xa = U[i]
        Xb = U[i+1]
        Ya = V[j]
        Yb = V[j+1]
        #print((Xa,Xb), " - ", (Ya,Yb) )
        df.at[Xa,Ya] = len(data.loc[( Xa <= data[Asset1]) & (data[Asset1] < Xb) & ( Ya <= data[Asset2]) & (data[Asset2] < Yb)])
        #print(df.loc[Xa,Ya])

df = df/len(data[Asset1]*2)
print(df)

U, V = np.meshgrid(U, V)   # Create coordinate points of X and Y
Z = np.array( df )
print(Z)
Z = Z.reshape(U.shape)

print()
plot_bivariate_pdf(U,V,Z)"""