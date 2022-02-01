import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import simulation

u1, u2 = simulation.simu_clayton(num=2500, theta=0.6)
from scipy.stats import norm
#apply distribution.ppf to transform uniform margin to the desired distribution in scipy.stats
u1 = norm.ppf(u1)
u2 = norm.ppf(u2)


# Block maxima 
block_lenght = 20 #number of trading days in one month
# Split data by block of equal lenght
u1_splitted = [u1[i*block_lenght:(i + 1) * block_lenght] for i in range((len(u1) + block_lenght - 1) // block_lenght )]  
u2_splitted = [u2[i*block_lenght:(i + 1) * block_lenght] for i in range((len(u2) + block_lenght - 1) // block_lenght )]  

block_maxima_u1 = []
block_maxima_u2 = []
for block_u1, block_u2 in zip(u1_splitted, u2_splitted):
    max_block_u1 = max(block_u1)
    max_block_u2 = max(block_u2)
    block_maxima_u1.append(max_block_u1)
    block_maxima_u2.append(max_block_u2)

plt.scatter(u1, u2, color="black", alpha=0.8)
plt.scatter(block_maxima_u1, block_maxima_u2, color="red", alpha=0.8)
plt.show()

def calc(i, UV):
    return np.log(np.sqrt(np.log(1/UV[0][i])*np.log(1/UV[1][i])) * np.log(1/(min(UV[0][i], UV[1][i])**2) ))
    
sumlog = sum([calc(i,[block_maxima_u1,block_maxima_u2]) for i in range(0, len(block_maxima_u1))])
TDC = 2 - 2*np.exp(sumlog/len(block_maxima_u1))
print(TDC)

from statsmodels.distributions.empirical_distribution import ECDF

ecdf1 = ECDF(block_maxima_u1)
ecdf2 = ECDF(block_maxima_u2)

n = len(block_maxima_u1)

block_maxima_u1 = [n*l/(n+1) for l in ecdf1(block_maxima_u1)]
block_maxima_u2 = [n*l/(n+1) for l in ecdf2(block_maxima_u2)]

def A_pickand(t,n, UV):
    Xi = lambda i, t : min( -np.log(UV[0][i])/(1-t), -np.log(UV[1][i])/t)
    Xi = sum([ Xi(i,t) for i in range(0,n)])
    Xi = Xi/(n)
    A_ = 1/(Xi)
    #A_ = min(max(A_, (1-t),t), 1)
    return A_

TDC = 2 - 2*A_pickand(t=0.5,n=n, UV=[block_maxima_u1,block_maxima_u2])

print("Pickand: ", TDC)

def A_cfg(t,n, UV):
    Xi = lambda i, t : min( -np.log(UV[0][i])/(1-t), -np.log(UV[1][i])/t)
    Xi = sum([-np.log(Xi(i, t)) for i in range(0,n) ] )/(n)
    A_ =np.exp(-0.57+ Xi) 
    #A_ = min(max(A_, (1-t),t), 1)
    return A_

TDC = 2 - 2*A_cfg(t=0.5,n=n, UV=[block_maxima_u1,block_maxima_u2])
print("CFG: ", TDC)

