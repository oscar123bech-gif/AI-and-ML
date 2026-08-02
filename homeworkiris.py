import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

csv = pd.read_csv("iris.csv")

print(csv.isnull().sum())

X = csv[["sepal_length","petal_length"]]
print(X)

y = csv["species"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=50)