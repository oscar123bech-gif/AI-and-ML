import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,11)

y = [1,3,2,4,5,6,8,7,9,10]
y = np.array(y)

plt.scatter(x,y)
plt.show()

meanx = x.mean() 
meany = y.mean()
m = np.sum((x - meanx) * (y - meany))/np.sum((x-meanx)**2)
c = meany - m*meanx 

print(m,c)

predictedy = m*x+c
print(predictedy)

plt.scatter(x,y)
plt.plot(x,predictedy)
plt.show()

error = np.sqrt(np.mean((predictedy - y)**2))

print(error)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
print(x.shape)
x = x.reshape(-1,1)
print(x.shape)

lr = LinearRegression()
lr.fit(x,y)
print(lr.coef_)
print(lr.intercept_)
predictedy2 = lr.predict(x)
print(predictedy2)


error2 = root_mean_squared_error(y,predictedy2)
print(error2)