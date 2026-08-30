import pandas as pd


csv = pd.read_csv("titanic.csv")
csv.info()
print (csv.notnull().sum())

from sklearn.preprocessing import LabelEncoder
print(csv["Sex"].value_counts())
le = LabelEncoder()
csv["Sex"] = le.fit_transform(csv["Sex"])
csv.info()

X = csv[["Sex","Fare","Pclass"]]
y = csv["Survived"]



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.1)
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()

lg.fit(X_train,y_train)

predictedy = lg.predict(X_test)

print(predictedy,y_test)

from sklearn.metrics import confusion_matrix,classification_report

error = confusion_matrix(y_test,predictedy)
print(error)

import matplotlib.pyplot as plt

import seaborn as sns 
sns.heatmap(error,annot=True,fmt="d")

plt.show()
print(classification_report(y_test,predictedy))