import numpy as np
import matplotlib.pyplot as plt
# Clayton copula


num = 2000
theta = 1.5

print("n = ", num, "   -  theta = ", theta, "   - LTD = ", 2**(-1/theta))

v1 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])
v2 = np.array([np.random.exponential(scale=1.0) for i in range(0,num)])

x = np.array([np.random.gamma(theta**(-1), scale=1.0) for i in range(0,num)])

u1 = (1 + v1/x)**(-1/theta)
u2 = (1 + v2/x)**(-1/theta)


plt.scatter(u1, u2, color="black", alpha=0.8)
plt.show()
