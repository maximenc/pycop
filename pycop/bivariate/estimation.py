import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, t

def pseudo_obs(data):

    ecdf1 = ECDF(data[0])
    ecdf2 = ECDF(data[1])

    n = len(data[0])

    scaled_ranks = np.array([[n*l/(n+1) for l in ecdf1(data[0])], [n*l/(n+1) for l in ecdf2(data[1])]])

    return scaled_ranks


def fit_mle(copula, data, marginals, known_parameters=None, opti_method='SLSQP'):

    marg_cdf = []
    marg_pdf = []

    if known_parameters != None:

        for i in range(2):
            if marginals[i] == "gaussian":

                mu = known_parameters[i]["mu"]
                sigma = known_parameters[i]["sigma"]
            
                marg_cdf.append(lambda i : norm.cdf(data[0][i],mu,sigma) )
                marg_pdf.append(lambda i : norm.pdf(data[0][i],mu,sigma) )

            elif marginals[i] == "student":
                mu = known_parameters[i]["mu"]
                sigma = known_parameters[i]["sigma"]
                nu = known_parameters[i]["nu"]

                marg_cdf.append(lambda i : t.cdf(data[0][i],mu,sigma, df=nu) )
                marg_pdf.append(lambda i : t.pdf(data[0][i],mu,sigma, df=nu) )

            logi= lambda  i, theta: np.log(copula.pdf(marg_cdf[0](i),marg_cdf[1](i),[theta]))+np.log(marg_pdf[0](i)) +np.log(marg_pdf[1](i))
            log_likelihood = lambda  theta:  -sum([logi(i, theta)  for i in range(0,len(data[0]))])

    else:
        for i in range(2):

            if marginals[i] == "gaussian":
            
                marg_cdf.append(lambda i,mu, sigma : norm.cdf(data[0][i],mu,sigma) )
                marg_pdf.append(lambda i, mu, sigma : norm.pdf(data[0][i],mu,sigma) )

            elif marginals[i] == "student":
                marg_cdf.append(lambda i, mu, sigma, nu : t.cdf(data[0][i],mu,sigma, df=nu) )
                marg_pdf.append(lambda i, mu, sigma, nu : t.pdf(data[0][i],mu,sigma, df=nu) )

            logi= lambda  i, theta: np.log(copula.pdf(marg_cdf[0](i),marg_cdf[1](i),[theta]))+np.log(marg_pdf[0](i)) +np.log(marg_pdf[1](i))
            log_likelihood = lambda  theta:  -sum([logi(i, theta)  for i in range(0,len(data[0]))])



        
    results = minimize(log_likelihood, copula.theta_start, method='Nelder-Mead') #options={'maxiter': 300})#.x[0]
    print("method = ", opti_method, " - termination = ", results.success, " - message: ", results.message)
    if results.success == True:
        return (results.x, -results.fun)

    print("optimization failed")
    return None

def fit_cmle(copula, data, opti_method='SLSQP'):
    """
    Uses the empirical cumulative distribution function (ecdf)  to estimate the margins instead of using a parametric model. 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    """
    psd_obs = pseudo_obs(data)

    def log_likelihood(theta):
        return -sum([ np.log(copula.pdf(psd_obs[0][i],psd_obs[1][i],[theta])) for i in range(0,len(psd_obs[0]))])
        

    if copula.bounds_param[0] == (None):
        results = minimize(log_likelihood, copula.theta_start, method='Nelder-Mead')
        print("method = Nelder-Mead - termination = ", results.success, " - message: ", results.message)
        return (results.x, -results.fun)
    else:

        results = minimize(log_likelihood, copula.theta_start, method=opti_method, bounds=copula.bounds_param) #options={'maxiter': 300})#.x[0]
        print("method = ", opti_method, " - termination = ", results.success, " - message: ", results.message)
        if results.success == True:
            return (results.x, -results.fun)

        print("optimization failed")
        return None



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
