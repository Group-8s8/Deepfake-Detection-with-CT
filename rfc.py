import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.externals 
import joblib

K_vector = np.array([1.0, 0.9999994039535522, 1.0000005960464478, -46629515264.0, 0.9999946355819702, -46629543936.0, -46629625856.0, -46629625856.0, 1.0000053644180298, 1.0, -2918260224.0, 61.2149658203125, -46692163584.0, -2918260224.0, 61.21503448486328, -46692114432.0, -2918257152.0, -714564829184.0, 1.0000001192092896, 0.0, 1836.2933349609375, 1.000032663345337, -250518992.0, 1836.3533935546875, -5771768832.0, 0.0, -2208126468096.0])
y = np.array([1, 1, 1, 1, -62482743296, 1, -62482743296, 1, -62482857984, 1, 0, 61, -188908208128, 0, -2891008966656, 1, -11806789632, 61, 1, 0, 1836, -5780512768, 0, -2211592011776, -5781151232, 0, 1836])
X_train, X_test, y_train, y_test = train_test_split(K_vector, y, test_size = 0.25, random_state = 21)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1,1))
X_test = scaler.transform(X_test.reshape(-1,1))
#print(X_test,X_train)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print( y_pred)
y_pred = y_pred.tolist()
trueCount = y_pred.count(1)
falseCount = y_pred.count(0)
print(f"Fake; Accuracy {(trueCount/len(y_pred))*100}%")
#print(len(y_pred[0]))
