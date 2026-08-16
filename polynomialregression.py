import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X = np.arange(1,100)
y = X**5

plt.scatter(X,y)
plt.plot()
plt.show()


csv = pd.read_csv("HousingData.csv")

csv.info()
print(csv.isnull().sum())

# crimmean = csv["CRIM"].mean()
# csv["CRIM"] = csv["CRIM"].fillna(crimmean)
# print(csv.isnull().sum())

csv.dropna(inplace=True)
print(csv.isnull().sum())
csv.info()
print(csv.shape)


X = csv[["LSTAT","RM"]]
y = csv["MEDV"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import PolynomialFeatures
pl = PolynomialFeatures(degree=2)
X_trainpoly = pl.fit_transform(X_train)
X_testpoly = pl.fit_transform(X_test)
print(X_train)
print(X_trainpoly)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_trainpoly,y_train)
print(lr.coef_)
print(lr.intercept_)

predictedy = lr.predict(X_testpoly)
print(predictedy)
from sklearn.metrics import root_mean_squared_error
error = root_mean_squared_error(y_test,predictedy)
print(error)



