import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.externals 
import joblib

K_vector = [[1,2,3,4,5,6],[2,5,3,6,1,4],[4,6,8,3,9,1]]
y = [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]]
test = [[1,2,3,4,5,6],[2,5,3,6,1,4]]

X_train, X_test, y_train, y_test = train_test_split(K_vector, y, test_size = 0.25, random_state = 21)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test,X_train)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test, y_pred)