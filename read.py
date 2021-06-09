import cv2 as cv
import numpy as np

from numpy import asarray
from PIL import Image

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#from sklearn.externals 
import joblib


class Initialize():
	def all(img):
		N = 3 #Kernel Size
		SIGMAZERO = 0.5
		#initializing R,P,W Matrices with 0. same size as img
		R = np.zeros(img.shape, np.float32)
		P = np.zeros(img.shape, np.float32)
		W = np.zeros(img.shape, np.float32)
		#Setting K with 1/(N*N).
		K = np.random.randint(2,size = (N,N)).astype("float32")

		#setting p0 and coordinates
		indices = np.where(img != [0])
		COORDINATES = zip(indices[0], indices[1])
		PZERO = 1/256
		ALPHA = 1
		return SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA

class EM():
	def algorithm(img):
		SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA = Initialize.all(img)
		sigma = SIGMAZERO #Initially 0.5, changes after each Iteration in m-step
		print("[INFO] Initial Random Kernel Matrix")
		print(K)
		iterate = 1
		for n in range(0, iterate): 
			K[0][0] = 0
			print(f"[INFO] In Iteration {n+1}")
			#e-step
			a,b,c,d,e,f,COORDINATES,g = Initialize.all(img)
			#a,bc,d,e,f,g are all unwanted returns.
			for x,y in COORDINATES:
				sumo = 0
				# This is equation 4 in the paper.
				for p in range(-ALPHA,ALPHA+1):
					for q in range(-ALPHA,ALPHA+1):
						try:
							sumo += K[p][q] * img[x + p][y + q]
						except IndexError:
							pass
				temp = img[x][y] - sumo
				# The above step gives "x-myu" for Guassian Distribution at each coordinate
				# The below step is to make sure there is no -Ve pixel values.
				if temp < 0:
					R[x][y] = temp * -1
				else:
					R[x][y] = temp
				# This finds the Guassian Distribution of the point
				P[x][y] = GD.guassian_distribution(R[x][y]**2, sigma)
				try:
					#bayes rule is applied at each point.
					W[x][y] = P[x][y]/(P[x][y] + PZERO)
				except IndexError:
					pass
			#m-step
			a,b,c,d,e,f,COORDINATES,g = Initialize.all(img)

			tsum = 0
			bsum = np.zeros(shape = (3,3),dtype = np.float32)
			csum = np.zeros(shape = (3,3),dtype = np.float32)
			BETA = np.zeros(shape = (3,3),dtype = np.float32)
			#program is really slow in the below loop.
			# In this loop, Kernel Matrix K is remade with newer values (W) that we got from e-step and older Kernel Matrix 
			for x,y in COORDINATES:
				for p in range(-ALPHA,ALPHA+1):
					for q in range(-ALPHA,ALPHA+1):
						try:
							BETA[p][q] = W[x][y] * img[x + p][y + q] * img[x][y]
						except IndexError:
							pass
						try:
							bsum[p][q] = W[x][y] * (img[x - p][y - q])**2
						except IndexError:
							pass
						for s in range(-ALPHA,ALPHA+1):
							for t in range(-ALPHA,ALPHA+1):
								try:
									tsum += W[x][y] * img[x - p][y - q] * img[x + s][y + t]
								except IndexError:
									pass
						for u in range(-ALPHA,ALPHA+1):
							for v in range(-ALPHA,ALPHA+1):
								csum[u][v] = K[u][v] * tsum
			for p in range(-ALPHA,ALPHA+1):
				for q in range(-ALPHA,ALPHA+1):
					if bsum[p][q] == 0:
						bsum[p][q] = 0.000001
					K[p][q] = (BETA[p][q] - csum[p][q])/bsum[p][q]

			# Here new Sigma value is generated from W and R.
			sigsum = 0
			wsum = 0
			a,b,c,d,e,f,COORDINATES,g = Initialize.all(img)
			for x,y in COORDINATES:
				sigsum += W[x][y] * R[x][y]**2
				wsum += W[x][y]

			#Checks and corrects to avoid divide by Zero
			if wsum < 0:
				wsum = wsum * -1
			sigmasqr = sigsum/wsum
			# new sigma
			sigma = np.sqrt(sigmasqr)
			if sigma > 10:
				sigma = int(str(sigma)[0])
		# return the final K vector after iterations
		return K

class GD():
	def guassian_distribution(x, sigma):
		_z = x / (sigma)**2
		# Below check avoids not using higher precision floating points because float32 cannot have exponent(709+).
		if _z < 709:
			_exp = np.exp(_z)
		else:
			_exp = 1
		_b = 2 * np.pi * sigma**2
		sq = np.sqrt(_b * _exp)
		if np.isinf(sq):
			sq= 1
		p = 1 / sq
		return p

class Normal():
	def normalize(img):
		# Image is normalized to values between 0 and 1 so that calculations will be easier.
		pixels = asarray(img)
		pixels = pixels.astype('float32')
		# 255 is max value. so (any number < 255)/255 => 0 < number < 1
		pixels /= 255.0
		
		return pixels

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
			print(f"\nFake; Prediction {(trueCount/len(y_pred))*100}%")
		else:
			print(f"\nReal; Prediction {(falseCount/len(y_pred))*100}%")

class Reader:
	def read(img):
		image = Image.open(img)
		#splits image to 3 channels. 
		r,g,b = image.split()
		bnormal = Normal.normalize(b)
		gnormal = Normal.normalize(g)
		rnormal = Normal.normalize(r)
		#calling em algorithm per channel.
		print("[INFO] Applying EMA for Blue Channel")
		bK = EM.algorithm(bnormal)
		print("[INFO] Applying EMA for Green Channel")
		gK = EM.algorithm(gnormal)
		print("[INFO] Applying EMA for Red Channel")
		rK = EM.algorithm(rnormal)

		print("[INFO] Concatnating 3 K vectors.")

		K_vector = np.concatenate((bK, gK, rK), axis=None)
		return K_vector.tolist()

if __name__ == '__main__':
	# loads image from given path
	fake = 'data/Fake/fake.png'
	real = 'data/Fake/real.png'
	print("[INFO] Performing for fake image.")
	K_vector = Reader.read(fake)
	#print(K_vector)
	print("[INFO] Performing for real image.")
	gg = Reader.read(real)
	y = [int(item) for item in gg]
	print("[INFO] Preparing for Random Forrest Classification.")
	Classifier.classify(K_vector,y)
