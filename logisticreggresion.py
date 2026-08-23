import pandas as pd


csv = pd.read_csv("titanic.csv")
csv.info()
print (csv.notnull().sum())

X = csv[["Sex","Fare","Pclass"]]
y = csv["Survived"]


print(csv["Sex"].value_counts())
