from distutils.log import error
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, t

import warnings
# suppress warnings
warnings.filterwarnings('ignore')

def pseudo_obs(data):
    """
    # Transform the dataset to uniform margins.

    The pseudo-observations are the scaled ranks.

    Parameters
    ----------
    data : array like
        The dataset to transform.
    
    Returns
    -------
    scaled_ranks : array like

    """

    ecdf1 = ECDF(data[0])
    ecdf2 = ECDF(data[1])

    n = len(data[0])

    scaled_ranks = np.array(
        [[n*l/(n+1) for l in ecdf1(data[0])], [n*l/(n+1) for l in ecdf2(data[1])]])

    return scaled_ranks


def fit_cmle(copula, data, opti_method='SLSQP'):
    """
    # Compute the Canonical Maximum likelihood Estimator (CMLE) using the pseudo-observations

    Parameters
    ----------
    data : array like
        The dataset.

    copula : class
        The copula object 
    opti_method : str, optional
        The optimization method to pass to known_parametersscipy.optimize.minimize`.
        The default algorithm is set to `SLSQP`

    Returns
    -------
    Return  the estimated parameter(s) in a list

    """
    
    psd_obs = pseudo_obs(data)

    def log_likelihood(parameters):
        params = [parameters]
        logl = -sum([ np.log(copula.get_pdf(psd_obs[0][i],psd_obs[1][i],params)) for i in range(0,len(psd_obs[0]))])
        return logl


    if copula.bounds_param[0] == (None):
        results = minimize(log_likelihood, copula.parameters_start, method='Nelder-Mead')
        print("method = Nelder-Mead - termination = ", results.success, " - message: ", results.message)
        return (results.x, -results.fun)
    else:

        results = minimize(log_likelihood, copula.parameters_start, method=opti_method, bounds=copula.bounds_param) #options={'maxiter': 300})#.x[0]
        print("method = ", opti_method, " - termination = ", results.success, " - message: ", results.message)
        if results.success == True:
            return (results.x, -results.fun)

        print("optimization failed")
        return None


def fit_cmle_mixt(copula, data, opti_method='SLSQP'):
    """
    # Compute the CMLE for a mixture copula using the pseudo-observations

    Parameters
    ----------
    data : array like
        The dataset.

    copula : class
        The mixture copula object 
    opti_method : str, optional
        The optimization method to pass to known_parametersscipy.optimize.minimize`.
        The default algorithm is set to `SLSQP`

    Returns
    -------
    Return  the estimated parameter(s) in a list

    """
    psd_obs = pseudo_obs(data)

    def log_likelihood(parameters):
        if copula.dim == 2:
            w1, param1, param2 = parameters
            params = [w1, param1, param2]
        else: # dim = 3
            w1, w2, w3, param1, param2, param3 = parameters
            params = [w1, w2, w3, param1, param2, param3]
        logl = -sum([ np.log(copula.get_pdf(psd_obs[0][i],psd_obs[1][i],params)) for i in range(0,len(psd_obs[0]))])
        return logl

    def con(parameters):
        w1, w2, w3, param1, param2, param3 = parameters
        return 1 - w1 - w2 - w3

    if copula.dim == 2:
        results = minimize(log_likelihood, copula.parameters_start, method=opti_method, bounds=copula.bounds_param) #options={'maxiter': 300})#.x[0]
    else:
        cons = {'type':'eq', 'fun': con}
        results = minimize(log_likelihood, copula.parameters_start, method=opti_method, bounds=copula.bounds_param, constraints=cons) #options={'maxiter': 300})#.x[0]


    print("method = ", opti_method, " - termination = ", results.success, " - message: ", results.message)
    if results.success == True:
        return (results.x, -results.fun)

    print("optimization failed")
    return None


def fit_mle(data, copula, marginals, opti_method='SLSQP', known_parameters=False):
    """
    # Compute the Maximum likelihood Estimator (MLE)

    Parameters
    ----------
    data : array like
        The dataset.

    copula : class
        The copula object 
    marginals : list
        A list of dictionary that contains the marginal distributions and their
        `loc` and `scale` parameters when the parameters are known.
        Example:
        marginals = [
            {
                "distribution": norm, "loc" : 0, "scale" : 1,
            },
            {
                "distribution": norm, "loc" : 0, "scale": 1,
            }]
    opti_method : str, optional
        The optimization method to pass to known_parametersscipy.optimize.minimize`.
        The default algorithm is set to `SLSQP`
    known_parameters : bool
        If the variable is set to `True` then we estimate the `loc` and `scale`
        parameters of the marginal distributions.

    Returns
    -------
    Return  the estimated parameter(s) in a list

    """

    if copula.type == "mixture":
        print("estimation of mixture only available with CMLE try fit mle")
        raise error
    
    if known_parameters == True:

        marg_cdf1 = lambda i : marginals[0]["distribution"].cdf(data[0][i], marginals[0]["loc"], marginals[0]["scale"]) 
        marg_pdf1 = lambda i : marginals[0]["distribution"].pdf(data[0][i], marginals[0]["loc"], marginals[0]["scale"])

        marg_cdf2 = lambda i : marginals[1]["distribution"].cdf(data[1][i], marginals[1]["loc"], marginals[1]["scale"]) 
        marg_pdf2 = lambda i : marginals[1]["distribution"].pdf(data[1][i], marginals[1]["loc"], marginals[1]["scale"]) 

        logi = lambda  i, theta: np.log(copula.get_pdf(marg_cdf1(i),marg_cdf2(i),[theta]))+np.log(marg_pdf1(i)) +np.log(marg_pdf2(i))
        log_likelihood = lambda  theta:  -sum([logi(i, theta)  for i in range(0,len(data[0]))])


        results = minimize(log_likelihood, copula.parameters_start, method=opti_method, )# options={'maxiter': 300})#.x[0]

    else:
        marg_cdf1 = lambda i, loc, scale : marginals[0]["distribution"].cdf(data[0][i], loc, scale) 
        marg_pdf1 = lambda i, loc, scale : marginals[0]["distribution"].pdf(data[0][i], loc, scale)

        marg_cdf2 = lambda i, loc, scale : marginals[1]["distribution"].cdf(data[1][i], loc, scale) 
        marg_pdf2 = lambda i, loc, scale : marginals[1]["distribution"].pdf(data[1][i], loc, scale) 

        logi = lambda  i, theta, loc1, scale1, loc2, scale2: \
            np.log(copula.get_pdf(marg_cdf1(i, loc1, scale1),marg_cdf2(i, loc2, scale2),[theta])) \
            + np.log(marg_pdf1(i, loc1, scale1)) +np.log(marg_pdf2(i, loc2, scale2))
        
        def log_likelihood(params):
            theta, loc1, scale1, loc2, scale2 = params
            return  -sum([logi(i, theta, loc1, scale1, loc2, scale2)  for i in range(0,len(data[0]))])

        results = minimize(log_likelihood, (copula.parameters_start, np.array(0), np.array(1), np.array(0), np.array(1)), method=opti_method, )# options={'maxiter': 300})#.x[0]

    print("method = ", opti_method, " - termination = ", results.success, " - message: ", results.message)
    if results.success == True:
        return results.x

    print("Optimization failed")
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
