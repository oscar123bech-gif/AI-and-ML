import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd

csv = pd.read_csv("HousingData.csv")

csv.info()

print(csv.isnull().sum())

lstatmean = csv["LSTAT"].mean()
print(lstatmean)

csv["LSTAT"] = csv["LSTAT"].fillna(value=lstatmean)

print(csv.isnull().sum())


X = csv[["LSTAT","RM"]]
print(X)

