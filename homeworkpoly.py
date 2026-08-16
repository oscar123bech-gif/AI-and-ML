import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv = pd.read_csv("iris.csv")

csv.info()
print(csv.isnull().sum())

X = csv[["petal_length","sepal_width"]]
y = csv["species"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import PolynomialFeatures
pl = PolynomialFeatures(degree=2)
X_trainpoly = pl.fit_transform(X_train)
print(X_train)
print(X_trainpoly)

