import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

import simulation

u1, u2 = simulation.simu_gumbel(num=1000, theta=1.8)
from scipy.stats import norm
#apply distribution.ppf to transform uniform margin to the desired distribution in scipy.stats
u1 = norm.ppf(u1)
u2 = norm.ppf(u2)

# Block maxima 
block_lenght = 10 #number of trading days in one month
# Split data by block of equal lenght
u1_splitted = [u1[i*block_lenght:(i + 1) * block_lenght] for i in range((len(u1) + block_lenght - 1) // block_lenght )]  
u2_splitted = [u2[i*block_lenght:(i + 1) * block_lenght] for i in range((len(u2) + block_lenght - 1) // block_lenght )]  

block_maxima = [[],[]]
for block_u1, block_u2 in zip(u1_splitted, u2_splitted):
    max_block_u1 = max(block_u1)
    max_block_u2 = max(block_u2)
    block_maxima[0].append(max_block_u1)
    block_maxima[1].append(max_block_u2)

print(block_maxima)
scaledranks =block_maxima

# Transform into scaled ranks
for i in range(0,len(block_maxima[0])):
    scaledranks[0][i] = len([l for l in block_maxima[0] if l <= block_maxima[0][i]])/(len(block_maxima[0])+1)
    scaledranks[1][i] = len([l for l in block_maxima[1] if l <= block_maxima[1][i]])/(len(block_maxima[0])+1)
print(scaledranks)

def xi(i,t, UV):
    return min( -np.log(UV[0][i])/(1-t), -np.log(UV[1][i])/t)

def A_pickand(t, UV):
    sum_xi = sum([xi(i,t, UV) for i in range(0,len(UV[0]))])

    return 1/(sum_xi/len(UV[0]) )

def A_cfg(t, UV):
    sum_logxi = sum([np.log(xi(i,t, UV)) for i in range(0,len(UV[0]))])
    return np.exp(-0.57- 1/len(UV[0]) * sum_logxi) 


TDC = 2 - 2*A_cfg(t=0.5, UV=scaledranks)
print(TDC)

TDC = 2 - 2*A_pickand(t=0.5, UV=scaledranks)
print(TDC)

def calc(i, UV):
    return np.log(np.sqrt(np.log(1/UV[0][i])*np.log(1/UV[1][i])) * np.log(1/(max(UV[0][i], UV[1][i])**2) ))
sumlog = sum([calc(i,scaledranks) for i in range(0, len(scaledranks[0]))])
TDC = 2 - 2*np.exp(sumlog/len(scaledranks[0]))
print(TDC)


