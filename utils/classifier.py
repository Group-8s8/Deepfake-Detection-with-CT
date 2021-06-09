import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class Classifier:
	def classify(K_vector, y):
		#this is the random forest classifier
		K_vector = np.array(K_vector)
		y = np.array(y)
		X_train, X_test, y_train, y_test = train_test_split(K_vector, y, test_size = 0.25, random_state = 21)
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train.reshape(-1,1))
		X_test = scaler.transform(X_test.reshape(-1,1))
		classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
		classifier.fit(X_train, y_train)

		y_pred = classifier.predict(X_test)
		y_pred = y_pred.tolist()
		trueCount = y_pred.count(1)
		falseCount = y_pred.count(0)
		if trueCount >= falseCount:
			print(f"\nFake; Prediction {(trueCount/len(y_pred))*100}%\n")
		else:
			print(f"\nReal; Prediction {(falseCount/len(y_pred))*100}%\n")