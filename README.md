<p align="center">
<img src="https://github.com/maximenc/pycop/raw/master/docs/images/logo_pycop.svg" width="40%" height="40%" />
</p>


[![PyPi version](https://badgen.net/pypi/v/pycop/)](https://pypi.org/project/pycop)
[![Downloads](https://pepy.tech/badge/pycop)](https://pepy.tech/project/pycop)
[![License](https://img.shields.io/pypi/l/pycop)](https://img.shields.io/pypi/l/pycop)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7030034.svg)](https://doi.org/10.5281/zenodo.7030034)

# Overview

Pycop is the most complete tool for modeling multivariate dependence with Python. The package provides methods such as estimation, random sample generation, and graphical representation for commonly used copula functions. The package supports the use of mixture models defined as convex combinations of copulas. Other methods based on the empirical copula such as the non-parametric Tail Dependence Coefficient are given.

Some of the features covered:
* Elliptical copulas (Gaussian & Student) and common Archimedean Copulas functions
* Mixture model of multiple copula functions (up to 3 copula functions)
* Multivariate random sample generation
* Empirical copula method
* Parametric and Non-parametric Tail Dependence Coefficient (TDC)


### Available copula function
<p align="center">

| Copula     |  Bivariate <br /> Graph &  Estimation | Multivariate <br /> Simulation  |
|---                | :-:      | :-:     |
| Mixture           | &check;  | &check; |
| Gaussian          | &check;  | &check; |
| Student           | &check;  | &check; |
| Clayton           | &check;  | &check; |
| Rotated Clayton   | &check;  | &check; |
| Gumbel            | &check;  | &check; |
| Rotated Gumbel    | &check;  | &check; |
| Frank             | &check;  | &check; |
| Joe               | &check;  | &check; |
| Rotated Joe       | &check;  | &check; |
| Galambos          | &check;  | &cross; |
| Rotated Galambos  | &check;  | &cross; |
| BB1               | &check;  | &cross; |
| BB2               | &check;  | &cross; |
| FGM               | &check;  | &cross; |
| Plackett          | &check;  | &cross; |
| AMH               | &cross;  | &check; |
</p>

# Usage

Install pycop using pip
```
pip install pycop
```

# Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/maximenc/pycop/blob/master/examples/example_estim.ipynb)
[Estimations on msci returns](https://github.com/maximenc/pycop/blob/master/examples/example_estim.ipynb)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/maximenc/pycop/blob/master/examples/example_plot.ipynb)
[Graphical Representations](https://github.com/maximenc/pycop/blob/master/examples/example_plot.ipynb)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/maximenc/pycop/blob/master/examples/example_simu.ipynb)
[Simulations](https://github.com/maximenc/pycop/blob/master/examples/example_simu.ipynb)



# Table of Contents

- [Graphical Representation](#Graphical-Representation)
  - [3d plot](#3d-plot)
  - [Contour plot](#Contour-plot)
  - [Mixture plot](#Mixture-plot)
- [Simulation](#Simulation)
  - [Gaussian](#Gaussian)
  - [Student](#Student)
  - [Archimedean](#Archimedean)
  - [High dimension](#High-dimension)
  - [Mixture simulation](#Mixture-simulation)
- [Estimation](#Estimation)
  - [Canonical Maximum Likelihood Estimation](#Canonical-Maximum-Likelihood-Estimation)
- [Tail Dependence Coefficient](#Tail-Dependence-Coefficient)
  - [Theoretical TDC](#Theoretical-TDC)
  - [Non-parametric TDC](#Non-parametric-TDC)
  - [Optimal Empirical TDC](#Optimal-Empirical-TDC)


# Graphical Representation

We first create a copula object by specifying the copula familly

```python
from pycop import archimedean
cop = archimedean(family="clayton")
```

Plot the cdf and pdf of the copula.


## 3d plot

```python
cop = archimedean(family="gumbel")

cop.plot_cdf([2], plot_type="3d", Nsplit=100 )
cop.plot_pdf([2], plot_type="3d", Nsplit=100, cmap="cividis" )
```


<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/gumbel_3d_cdf.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/gumbel_3d_pdf.svg" width="45%" />
</p>


## Contour plot

plot the contour

```python
cop = archimedean(family="plackett")

cop.plot_cdf([2], plot_type="contour", Nsplit=100 )
cop.plot_pdf([2], plot_type="contour", Nsplit=100, )
```


<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/plackett_contour_cdf.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/plackett_contour_pdf.svg" width="45%" />
</p>


It is also possible to add specific marginals

```python
cop = archimedean.archimedean(family="clayton")

from scipy.stats import norm


marginals = [
    {
        "distribution": norm, "loc" : 0, "scale" : 0.8,
    },
    {
        "distribution": norm, "loc" : 0, "scale": 0.6,
    }]

cop.plot_mpdf([2], marginals, plot_type="3d",Nsplit=100,
            rstride=1, cstride=1,
            antialiased=True,
            cmap="cividis",
            edgecolor='black',
            linewidth=0.1,
            zorder=1,
            alpha=1)

lvls = [0.02, 0.05, 0.1, 0.2, 0.3]

cop.plot_mpdf([2], marginals, plot_type="contour", Nsplit=100,  lvls=lvls)
```


<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/clayton_3d_mpdf.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/clayton_contour_mpdf.svg" width="45%" />
</p>

## Mixture plot

mixture of 2 copulas

```python
from pycop.mixture import mixture

cop = mixture(["clayton", "gumbel"])
cop.plot_pdf([0.2, 2, 2],  plot_type="contour", Nsplit=40,  lvls=[0.1,0.4,0.8,1.3,1.6] )
# plot with defined marginals
cop.plot_mpdf([0.2, 2, 2], marginals, plot_type="contour", Nsplit=50)
```
<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/2c_mixture_contour_pdf.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/2c_mixture_contour_mpdf.svg" width="45%" />
</p>


```python

cop = mixture(["clayton","gaussian", "gumbel"])
cop.plot_pdf([1/3, 1/3, 1/3, 2, 0.5, 4],  plot_type="contour", Nsplit=40,  lvls=[0.1,0.4,0.8,1.3,1.6] )
cop.plot_mpdf([1/3, 1/3, 1/3, 2, 0.5, 2], marginals, plot_type="contour", Nsplit=50)
```
<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/3c_mixture_contour_pdf.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/plot/3c_mixture_contour_mpdf.svg" width="45%" />
</p>


# Simulation

## Gaussian


```python
from scipy.stats import norm
from pycop import simulation

n = 2 # dimension
m = 1000 # sample size

corrMatrix = np.array([[1, 0.8], [0.8, 1]])
u1, u2 = simulation.simu_gaussian(n, m, corrMatrix)
```
Adding gaussian marginals, (using distribution.ppf from scipy.statsto transform uniform margin to the desired distribution)

```python
u1 = norm.ppf(u1)
u2 = norm.ppf(u2)
```

## Student
```python
u1, u2 = simulation.simu_tstudent(n, m, corrMatrix, nu=1)

```


<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/gaussian_simu.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/student_simu.svg" width="45%" />
</p>





## Archimedean

List of archimedean cop available

```python
u1, u2 = simulation.simu_archimedean("gumbel", n, m, theta=2)
u1, u2 = 1 - u1, 1 - u2
```

Rotated

```python
u1, u2 = 1 - u1, 1 - u2
```


<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/gumbel_simu.svg" width="45%" />
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/rgumbel_simu.svg" width="45%" />
</p>


## High dimension


```python

n = 3       # Dimension
m = 1000    # Sample size

corrMatrix = np.array([[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1]])
u = simulation.simu_gaussian(n, m, corrMatrix)
u = norm.ppf(u)
```
<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/gaussian_simu_n3.svg" width="45%" />
</p>


```python
u = simulation.simu_archimedean("clayton", n, m, theta=2)
u = norm.ppf(u)
```

<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/clayton_simu_n3.svg" width="45%" />
</p>

## Mixture simulation

Simulation from a mixture of 2 copulas

```python
n = 3
m = 2000

combination = [
    {"type": "clayton", "weight": 1/3, "theta": 2},
    {"type": "gumbel", "weight": 1/3, "theta": 3}
]

u = simulation.simu_mixture(n, m, combination)
u = norm.ppf(u)
```
<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/2c_mixture_simu.svg" width="45%" />
</p>

Simulation from a mixture of 3 copulas
```python
corrMatrix = np.array([[1, 0.8, 0], [0.8, 1, 0], [0, 0, 1]])


combination = [
    {"type": "clayton", "weight": 1/3, "theta": 2},
    {"type": "student", "weight": 1/3, "corrMatrix": corrMatrix, "nu":2},
    {"type": "gumbel", "weight": 1/3, "theta":3}
]

u = simulation.simu_mixture(n, m, combination)
u = norm.ppf(u)
```

<p align="center">
  <img src="https://github.com/maximenc/pycop/raw/master/docs/images/simu/3c_mixture_simu.svg" width="45%" />
</p>


# Estimation

Estimation available :
CMLE


## Canonical Maximum Likelihood Estimation (CMLE)

Import a sample with pandas

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/msci.csv")
df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.drop(["Date"], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))

df = df.dropna()
```


```python
from pycop import estimation, archimedean

cop = archimedean.archimedean("clayton")
param, cmle = estimation.fit_cmle(cop, df[["US","UK"]])

```
clayton  estim:  0.8025977727691012



# Tail Dependence coefficient

## Theoretical TDC

```python
cop.LTDC(theta=param)
cop.UTDC(theta=param)
```


## Non-parametric TDC
Create an empirical copula object
```python
from pycop import empirical

cop = empirical(df[["US","UK"]])
```
Compute the non-parametric Upper TDC (UTDC) or the Lower TDC (LTDC) for a given threshold:
```python
cop.LTDC(0.01) # i/n = 1%
cop.UTDC(0.99) # i/n = 99%
```

## Optimal Empirical TDC
Returns the optimal non-parametric TDC based on the heuristic plateau-finding algorithm from Frahm et al (2005) "Estimating the tail-dependence coefficient: properties and pitfalls"

```python
cop.optimal_tdc("upper")
cop.optimal_tdc("lower")
```
