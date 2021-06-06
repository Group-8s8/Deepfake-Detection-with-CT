from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
import pandas as  pd

K_vector = [[1,2,3,4,5,6],[2,5,3,6,1,4]]

clf = rfc(n_estimators = 100)

data = pd.DataFrame({'test':K_vector[0],'train':K_vector[1]})

data.head()

X_train, X_test, y_train, y_test = tts(data, data, test_size=0.3)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))