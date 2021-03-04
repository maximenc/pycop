# Copula


## Simple Usage

```python
import pandas as pd
import numpy as np
from bivariate import copula, estimation


df = pd.read_csv("data/msci1.csv")
df.index = pd.to_datetime(df["Date"], format='%m/%d/%Y')
df = df.drop(['Date'], axis=1)

for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))
df = df.dropna()

data = df[["US","UK"]]

cop = copula.Archimedean(family='joe')
param_, cmle = estimation.fit_cmle(cop, data)
print(param_)

family1 = ['clayton', 'rgumbel','rjoe','rgalambos']
family2 = ['rclayton', 'gumbel','joe','galambos']

cop = copula.Mix2Copula(family1="clayton",family2="gumbel")


param_, cmle = estimation.fit_cmle(cop, data)

# getting parameters
print(param_)

ltd = cop.LTD(w1=param_[0], theta1=param_[1])
utd = cop.UTD(w1=param_[0], theta2=param_[2])

print(ltd, utd)

# Empirical

cop = copula.Empirical(data)
ltd_ = cop.LTD_(0.01)
utd_ = cop.UTD_(0.99)


print(ltd_, utd_)


# Simulation
import matplotlib.pyplot as plt
from bivariate import simulation

## Gaussian Copula
u1, u2 = simulation.simu_gaussian(num=2000, rho=0.5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

## Gaussian Copula with gaussian marginals
from scipy.stats import norm

u1, u2 = simulation.simu_gaussian(num=2000, rho=0.5)
#apply distribution.ppf to transform uniform margin to the desired distribution in scipy.stats
u1 = norm.ppf(u1)
u2 = norm.ppf(u2)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

## Plot Copula density
# Empirical
data = df[["Japan","US"]]
cop = copula.Empirical(data)
cop.plot_pdf(Nsplit=50)

# Clayton
cop = copula.Archimedean(family='clayton')
cop.plot_pdf(theta=1.5, Nsplit=25)
cop.plot_cdf(theta=1.5, Nsplit=25)

## Student Copula
u1, u2 = simulation.simu_tstudent(num=3000, nu=1, rho=0.5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

## Clayton Copula
u1, u2 = simulation.simu_clayton(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

## Frank Copula
u1, u2 = simulation.simu_frank(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()

## Gumbel Copula
u1, u2 = simulation.simu_gumbel(num=2000, theta=5)
plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()


```
