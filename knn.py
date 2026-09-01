import pandas as pd

csv = pd.read_csv("iris.csv")
csv.info()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
csv["species"] = le.fit_transform(csv["species"])

X = csv[["sepal_length","petal_length"]]
y = csv["species"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1,)

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=10)

knc.fit(X_train,y_train)

predictedy = knc.predict(X_test)
print(predictedy)

from sklearn.metrics import confusion_matrix
error = confusion_matrix(y_test,predictedy)
print(error)

