---
title: 'pycop: a Python package for dependence modeling with copulas'
tags:
  - copula
  - simulation
  - dependence
  - tail dependence
  - mixture
authors:
  - name: Maxime L. D. Nicolas
    orcid: 0000-0001-6135-3932
    affiliation: 1
affiliations:
 - name: Sorbonne School of Economics, Pantheon-Sorbonne University, Paris, France
   index: 1
date: 18 August 2022
bibliography: paper.bib
---

# Summary

Researchers and practitioners from various fields have been interested in quantifying the dependence between multiple random variables. Although simple statistics based on linear models, such as Pearson’s correlation coefficient, are often preferred to ease interpretation, more realistic nonlinear approaches are often needed to model complex dependence structures using the joint distribution function of random variables; in particular, this type of complex dependence can be defined by a copula, which is a marginal-free version of the joint distribution function [@nelsen2007introduction; @joe2014dependence].

Indeed, copulas are powerful statistical tools that allow the separate estimation of marginals and copula functions. Several parametric copula functions exist, where the parameters usually describe the intensity of the dependence, different families of copula functions are given in @joe1997multivariate. Copula methods are used in various fields, such as hydrology [@genest2007everything; @poulin2007importance], meteorology and climatology [@schoelzel2008multivariate], astronomy [@sato2011copula], and finance and economics [@cherubini2004copula].

The purpose of the package is to provide tools for copula modeling with Python. It provides all of the common parametric copula functions to model a variety of dependence structures. The package includes features such as Elliptical copulas (Gaussian & Student), common Archimedean copulas functions, mixture models defined as convex combinations of copulas, random sample generation models, empirical copula methods, and the computation of parametric and nonparametric estimation of tail dependence coefficients.

# Statement of need

Due to its universality and increasing popularity, many scientists now rely on Python for statistical analysis. However, compared to the R programming language, Python still lacks statistical tools. The most common copula tools rely on R [@yan2007enjoy; @kojadinovic2010modeling; @laverny2020empirical], making it inconvenient for researchers and practitioners who rely on Python to model multivariate dependencies. We aim to fill this gap by providing a package to model multivariate dependencies Python. The pycop package is created for researchers in broad fields who need to easily use copula modeling with Python.

Moreover, existing tools do not allow the modeling of convex combinations of multiple copula functions, namely, mixture copulas. The software pays particular attention to the modeling of mixture models, which are defined as convex combinations of copula functions [@mclachlan1988mixture]. Due to their flexibility in modeling unknown distributional shapes, mixture models are increasingly used in statistical literature [@mclachlan2019finite]. Mixture copula models are increasingly used in the literature [@weiss2015mixture; @kim2013mixture; @chabi2018crash]. We fill this gap in the statistical software literature by creating a tool that has the ability to plot, estimate and simulate a mixture copula model.

The `pycop` package is built on a user-friendly API to create a specific copula object for which we can estimate the parameters for a given dataset, generate random samples and plot the density and cumulative distribution functions. The library comes with documentation and several Jupyter notebook examples.

# Acknowledgments

I acknowledge the important contributions from David Woroniuk and Alexis Bogroff to the code. I did not receive any financial support for this project.

# References
