import matplotlib.pyplot as plt
import numpy as np


def empirical_density_contourplot(u, v, lims):
    
    res = 10

    pts = np.array([u,v])

    pts = (pts - lims[0]) * res / (lims[1] - lims[0])
    pts = np.round(pts).astype(int)
    pts[pts<0] = 0
    pts[pts>(res-1)] = res - 1

    Z = np.zeros((res,res))
    for i in range(0, len(u)):
        Z[pts[0,i],pts[1,i]] += 1.

    #Z /= len(u) * (lims[1]-lims[0])**2 / res**2

    lvls = np.percentile(Z.flatten(), (50, 80, 90,))
    x = np.linspace(lims[0],lims[1],res)
    y = np.linspace(lims[0],lims[1],res)
    X, Y = np.meshgrid(x,y)
    CS2 = plt.contour(X,Y,Z, levels=lvls, colors="k", linewidths=0.8)
    fmt = {}

    for l,s in zip( CS2.levels, [ "90%", "80%", "50%" ] ):
        fmt[l] = s

    plt.clabel(CS2, inline=1, inline_spacing=0, fmt=fmt, fontsize=8)


def empiricalplot(u, contour=True):

    minu = min([min(ui) for ui in u])
    maxu = max([max(ui) for ui in u])
    setplenght = maxu-minu
    str("{:.1f}".format(minu))
    lowerticks = [minu,minu+0.2*setplenght,minu+0.4*setplenght]
    upperticks = [minu+0.6*setplenght,minu+0.8*setplenght,maxu]
    limticks = [minu-0.1*setplenght, maxu+0.1*setplenght]

    n=len(u)
    for i in range(0,n):
        for j in range(0,n):
            if i == j:
                ax = plt.subplot(n, n, 1+(n+1)*i)
                #plt.text(0.5, 0.5, r"$u_%s$" % str(i+1))
                plt.xlim(limticks)
                plt.ylim(limticks)
                plt.xticks(lowerticks, [str("{:.1f}".format(tcks)) for tcks in lowerticks], ha='center')
                plt.yticks(upperticks, [str("{:.1f}".format(tcks)) for tcks in upperticks], va='center', ha='left') 
                ax.tick_params(axis="y",direction="in",  pad=-10)
                ax.tick_params(axis="x",direction="in",  pad=-15)
            
                ax2 = ax.twinx()
                plt.xlim(limticks)
                plt.ylim(limticks)
                plt.yticks(lowerticks, [str("{:.1f}".format(tcks)) for tcks in lowerticks], va='center', ha='right') 
                ax2.tick_params(axis="y",direction="in",  pad=-10)

                ax3 = ax.twiny()
                plt.xlim(limticks)
                plt.ylim(limticks)
                plt.xticks(upperticks, [str("{:.1f}".format(tcks)) for tcks in upperticks], va='center', ha='center') 
                ax3.tick_params(axis="x",direction="in",  pad=-12)
            elif i < j:
                if contour == True:
                    ax = plt.subplot(n, n, n*i+j+1)
                    empirical_density_contourplot(u[i], u[j], [minu, maxu])
                else:
                    pass

                plt.xlim(limticks)
                plt.ylim(limticks)
                plt.xticks([])
                plt.yticks([])
            else:
                ax = plt.subplot(n, n, n*i+j+1)
                plt.scatter(u[i], u[j], alpha=0.8, facecolors='none', edgecolors='b', s=2)
                plt.xlim(limticks)
                plt.ylim(limticks)
                plt.xticks([])
                plt.yticks([])

    plt.subplots_adjust(bottom=0.02, right=0.98, top=0.98, left=0.02, wspace=0.0, hspace=0.0)
    plt.show()



