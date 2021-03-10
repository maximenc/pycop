import pandas as pd
import numpy as np
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

def fit_cmle(copula, data):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    """
    psd_obs = pseudo_obs(data)

    bounded_opti_methods = ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']

    def log_likelihood(theta):
        return -sum([ np.log(copula.pdf(psd_obs.iloc[i,0],psd_obs.iloc[i,1],theta)) for i in range(0,len(psd_obs))])
        
    """
    if copula.bounds_param[0] == (None):
        results = minimize(log_likelihood, copula.theta_start, method='Nelder-Mead')#.x[0]
        #print("method = Nelder-Mead - success = ", results.success, " - message: ", results.message)
        return (results.x, -results.fun)
    else:
    """
    for opti_method in bounded_opti_methods:
        results = minimize(log_likelihood, copula.theta_start, method=opti_method, bounds=copula.bounds_param) #options={'maxiter': 300})#.x[0]
        #print("method = ", opti_method, " - success = ", results.success, " - message: ", results.message)
        if results.success == True:
            return (results.x, -results.fun)

        print("optimization failed")
        return ([10e-6], 0)



def IAD_dist(copula, data, theta):

    ranks = data
    order_u = ranks.iloc[:,0].argsort()
    order_v = ranks.iloc[:,1].argsort()
    ranks.iloc[:,0] = order_u.argsort()
    ranks.iloc[:,1] = order_v.argsort()

    def C_emp(i,j,ranks):
        return 1/n * len(ranks.loc[(ranks.iloc[:,0]<= i ) & (ranks.iloc[:,1] <= j) ])

    n = len(data)
    IAD = 0
    for i in range(1,n):
        for j in range(1,n):
            C_ = C_emp(i,j,ranks)
            C = copula.cdf((i/n),(j/n),theta)
            IAD = IAD + ((C_-C)**2)/(C*(1-C))
    return IAD


def AD_dist(copula, data, theta):

    ranks = data
    order_u = ranks.iloc[:,0].argsort()
    order_v = ranks.iloc[:,1].argsort()
    ranks.iloc[:,0] = order_u.argsort()
    ranks.iloc[:,1] = order_v.argsort()

    def C_emp(i,j,ranks):
        return 1/n * len(ranks.loc[(ranks.iloc[:,0]<= i ) & (ranks.iloc[:,1] <= j) ])

    n = len(data)
    AD_lst = []
    for i in range(1,n):
        for j in range(1,n):
            C_ = C_emp(i,j,ranks)
            C = copula.cdf((i/n),(j/n),theta)
            term = abs(C_ - C)/np.sqrt(C*(1-C))
            AD_lst.append(term)
    return max(AD_lst)
