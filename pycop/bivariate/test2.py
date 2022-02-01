import numpy as np

import gaussian, student, archimedean
import estimation 




import pandas as pd
import numpy as np

df = pd.read_csv("msci.csv")
df.index = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.drop(["Date"], axis=1)

for col in df.columns.values:
    df.loc[:,col] = np.log(df[col]) - np.log(df[col].shift(1))

df = df.dropna()

print(df)

data = np.array([df["US"],df["UK"]])


cop = archimedean.archimedean(family="frank")
#param, cmle = estimation.fit_cmle(cop, data)
#print(param)

data = np.array([df["US"],df["UK"]])

cop = archimedean.archimedean(family="frank")
param, cmle = estimation.fit_mle(cop, data, marginals=["gaussian", "student"], known_parameters = [{"mu":0,"sigma":0.02}, {"mu":0,"sigma":0.02, "nu":10}])
print(param)



