# Copula and Tail Dependence

Install pycop using pip
```
pip install pycop
```

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

# Empirical Copula
Create an empirical copula object 
```python
from pycop.bivariate.copula import empirical

cop = empirical(df[["US","UK"]])
```

### Plot Empirical density
Plot the PDF or the CDF with a smoothing parameter "Nsplit":
```python
cop.plot_pdf(Nsplit=50)
cop.plot_cdf(Nsplit=50) 
```

### Non-parametric Tail Dependence Coefficient (TDC)
Compute the non-parametric Upper TDC (UTDC) or the Lower TDC (LTDC) for a given threshold:
```python
cop.LTDC(0.01) # i/n = 1%
cop.UTDC(0.99) # i/n = 99%
```

### Optimal Empirical Tail Dependence coefficient (TDC)
Returns the optimal non-parametric TDC based on the heuristic plateau-finding algorithm from Frahm et al (2005) "Estimating the tail-dependence coefficient: properties and pitfalls"

```python
cop.optimal_tdc("upper") 
cop.optimal_tdc("lower")
```

# Archimedean Copula
Returns the estimated parameter from CMLE.
Available archimedean copula functions are:
* clayton
* gumbel
* frank
* joe
* galambos
* fgm
* plackett
* rgumbel
* rclayton
* rjoe
* rgalambos

```python
from pycop.bivariate.copula import archimedean
cop = archimedean(family="clayton")
```

## Density graph
```python
cop.plot_pdf(theta=1.5, Nsplit=50)
cop.plot_cdf(theta=1.5, Nsplit=25)
```

## Canonical Maximum Likelihood Estimation (CMLE)
```python
from pycop.bivariate import estimation
param, cmle = estimation.fit_cmle(cop, df[["US","UK"]])
```

## Tail Dependence coefficient (TDC)
```python
cop.LTDC(theta=param)
cop.UTDC(theta=param)
```

## Combining archimedean copula
```python
from pycop.bivariate.copula import mix2Copula
cop = mix2Copula(family1="clayton", family2="gumbel")
param, cmle = estimation.fit_cmle(cop, data)

cop.LTDC(w1=param[0], theta1=param[1])
cop.UTDC(w1=param[0], theta2=param[2])
```

# Simulation
## Gaussian Copula
```python
from pycop.bivariate import simulation
import matplotlib.pyplot as plt

u1, u2 = simulation.simu_gaussian(num=2000, rho=0.5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
```

Adding gaussian marginals, (using distribution.ppf from scipy.statsto transform uniform margin to the desired distribution) 
```python
from scipy.stats import norm
u1 = norm.ppf(u1)
u2 = norm.ppf(u2)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
```

### Student Copula
```python
u1, u2 = simulation.simu_tstudent(num=3000, nu=1, rho=0.5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
```

### Archimedean Copula
Clayton Copula
```python
u1, u2 = simulation.simu_clayton(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
```

Frank Copula
```python
u1, u2 = simulation.simu_frank(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
```

Gumbel Copula
```python
u1, u2 = simulation.simu_gumbel(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
```
