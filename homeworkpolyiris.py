import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



csv = pd.read_csv("iris.csv")

csv.info()
print(csv.isnull().sum())


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
csv["species"] = le.fit_transform(csv["species"])

X = csv[["sepal_length","sepal_width"]]
y = csv["species"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import PolynomialFeatures
pl = PolynomialFeatures(degree=2)
X_trainpoly = pl.fit_transform(X_train)
X_testpoly = pl.fit_transform(X_test)


print(X_train)
print(X_trainpoly)
from sklearn.linear_model import LinearRegression
pl = LinearRegression()
pl.fit(X_trainpoly,y_train)
print(pl.coef_)
print(pl.intercept_)

predictedy = pl.predict(X_testpoly)
print(predictedy)
from sklearn.metrics import root_mean_squared_error
error = root_mean_squared_error(y_test,predictedy)
print(error)



