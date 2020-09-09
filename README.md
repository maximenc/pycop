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



p = cop.params
# cop.params = ...  # you can override parameters too, even after it's fitted!  

# get a summary of the copula. If it's fitted, fit details will be present too
cop.summary()

# overriding parameters, for Elliptical Copulae, you can override the correlation matrix
cop[:] = np.eye(8)  # in this case, this will be equivalent to an Independent Copula
```