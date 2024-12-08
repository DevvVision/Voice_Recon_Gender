import numpy as np
import matplotlib.pyplot as pyt
import pandas as pd

ds = pd.read_csv("voice.csv")
X = ds.iloc[:,[0,1,3,5,8,9,11,12]].values
y = ds.iloc[:,-1].values

# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(missing_values=0,strategy='mean')
# imp.fit(X[:,17:19])
# X[:,17:19] = imp.transform(X[:,17:19])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)

# Spliting the Values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

# Standard Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

# from sklearn.neighbors




