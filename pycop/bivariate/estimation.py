import numpy as np
from scipy.optimize import minimize
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

    n = len(data[0])  # Assuming data[0] and data[1] are of the same length

    scaled_rank = lambda values: (np.argsort(np.argsort(values)) + 1) / (n + 1)

    scaled_ranks = np.array([scaled_rank(data[0]), scaled_rank(data[1])])

    return scaled_ranks


def fit_cmle(copula, data, opti_method='SLSQP', options={}):
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
    options : dict, optional
        The dictionary that contains the options to pass to the scipy.minimize function
        options={'maxiter': 100000}

    Returns
    -------
    Return  the estimated parameter(s) in a list

    """

    psd_obs = pseudo_obs(data)

    def log_likelihood(parameters):
        """
        The number of parameters depends on the type of copule function
        """
        if len(copula.bounds_param) == 1:
            params = [parameters]
        else:
            param1, param2 = parameters
            params = [param1, param2]
        logl = -np.sum(np.log(copula.get_pdf(psd_obs[0], psd_obs[1], params)))
        return logl

    if (copula.bounds_param[0] == (None, None)):
        results = minimize(log_likelihood, copula.parameters_start, method='Nelder-Mead', options=options)
        # print("method: Nelder-Mead - success:", results.success, ":", results.message)
        return (results.x, -results.fun)

    else:
        results = minimize(log_likelihood, copula.parameters_start, method=opti_method, bounds=copula.bounds_param, options=options)
        # print("method:", opti_method, "- success:", results.success, ":", results.message)
        if results.success == True:
            return (results.x, -results.fun)
        else:
            print(results)
            print("optimization failed")
            return None


def fit_cmle_mixt(copula, data, opti_method='SLSQP', options={}):
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
        logl = -sum([ np.log(copula.get_pdf(psd_obs[0][i], psd_obs[1][i],params)) for i in range(0, len(psd_obs[0]))])
        return logl

    # copula.dim gives the number of weights to consider
    cons = [{'type': 'eq', 'fun': lambda parameters: np.sum(parameters[:copula.dim]) - 1}]

    results = minimize(log_likelihood,
                       copula.parameters_start,
                       method=opti_method,
                       bounds=copula.bounds_param,
                       constraints=cons,
                       options=options)

    #print("method:", opti_method, "- success:", results.success, ":", results.message)
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
        raise ValueError
    
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

    print("method:", opti_method, "- success:", results.success, ":", results.message)
    if results.success == True:
        return results.x

    print("Optimization failed")
    return None

def IAD_dist(copula, data, param):
    """
    Compute the Integrated Anderson-Darling (IAD) distance between 
    the parametric copula and the empirical copula with vectorization.

    Info:
        This function first computes the empirical copula. It then computes the 
        theoretical (parametric) copula values using the provided copula function 
        and parameters. The IAD distance is calculated based on the differences 
        between these two copulas.    
        Based on equation 9 in "Crash Sensitivity and the Cross Section of Expected
        Stock Returns" (2018) Journal of Financial and Quantitative Analysis

    Args:
        copula (function): The copula object, providing a method `get_cdf` to compute the CDF.
        data (array-like): The underlying data as a 2D array, where each row is a dimension.
        param (array-like): The parameters of the copula.

    Returns:
        float: The IAD distance between the empirical and the parametric copulas.
    """
    
    n = len(data[0])
    
    # Get the order statistics for each dimension
    sorted_u = np.sort(data[0])
    sorted_v = np.sort(data[1])

    # Create a grid of comparisons for each pair (u, v)
    u_grid, v_grid = np.meshgrid(sorted_u, sorted_v, indexing='ij')
    
    # Count the number of points below the threshold in both dimensions
    # Use broadcasting to compare all pairs and count
    counts = np.sum((data[0][:, None, None] <= u_grid) & (data[1][:, None, None] <= v_grid), axis=0)
    # Compute the empirical copula
    C_empirical = counts / n
    
    # Prepare the grid for computing the parametric copula
    x_values, y_values = np.linspace(1/n, 1-1/n, n), np.linspace(1/n, 1-1/n, n)

    # Compute the parametric (theoretical) copula values
    C_copula = np.array([[copula.get_cdf(x, y, param) for x in x_values] for y in y_values])

    # Calculate the Integrated Anderson-Darling distance
    IAD = np.sum(((C_empirical - C_copula) ** 2) / (C_copula - C_copula**2))

    return IAD


def AD_dist(copula, data, param):
    """
    Compute the Anderson-Darling (IAD) distance between the parametric
    copula and the empirical copula with vectorization.

    Same principle as IAD_dist()
    """
    
    n = len(data[0])
    
    sorted_u = np.sort(data[0])
    sorted_v = np.sort(data[1])

    u_grid, v_grid = np.meshgrid(sorted_u, sorted_v, indexing='ij')
    
    counts = np.sum((data[0][:, None, None] <= u_grid) & (data[1][:, None, None] <= v_grid), axis=0)
    C_empirical = counts / n
    
    x_values, y_values = np.linspace(1/n, 1-1/n, n), np.linspace(1/n, 1-1/n, n)
    C_copula = np.array([[copula.get_cdf(x, y, param) for x in x_values] for y in y_values])

    AD = np.max(((C_empirical - C_copula) ** 2) / (C_copula - C_copula**2))
    return AD