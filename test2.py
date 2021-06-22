import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import minimize


def pseudo_obs(data):
    """
        take dataframe as argument and returns 
        Pseudo-observations from real data X
    """
    pseudo_obs = data
    for i in range(len(data.columns)):
        order = pseudo_obs.iloc[:,i].argsort()
        ranks = order.argsort()
        pseudo_obs.iloc[:,i] = [ (r + 1) / (len(data) + 1) for r in ranks ]
    return pseudo_obs


df = pd.read_csv("data/msci.csv")
df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.drop(["Date"], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))

df = df.dropna()




import pycop.multivariate.copula as cop


from pycop.bivariate.copula import archimedean
copl = archimedean(family="clayton")
copl.plot_cdf(theta=1.5, Nsplit=50)
from pycop.bivariate import estimation

#param, cmle = estimation.fit_cmle(copl, df[["US","UK"]])
#print(param)


cp = cop.archimedean(family="clayton", d=2)
cp.plot_cdf(theta=1.5, Nsplit=50)
cp.plot_pdf(theta=1.5, Nsplit=50)
"""
psd_obs = pseudo_obs(df[["US","UK"]])

bounded_opti_methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']

def log_likelihood(theta):
    return -sum([ np.log(cp.pdf(theta, [psd_obs.iloc[i,0],psd_obs.iloc[i,1]] ) ) for i in range(0,len(psd_obs))])
        

results = minimize(log_likelihood, cp.theta_start, method='L-BFGS-B', bounds=cp.bounds_param) #options={'maxiter': 300})#.x[0]
#print("method = ", opti_method, " - success = ", results.success, " - message: ", results.message)
if results.success == True:
    print(results.x, -results.fun)
else:
    print("optimization failed")

"""
