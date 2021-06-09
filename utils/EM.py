import numpy as np

from utils.GD import GD

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
		print("[INFO] Initial Random Kernel Matrix:")
		print(K)
		iterate = 1
		for n in range(0, iterate): 
			K[0][0] = 0
			print(f"[INFO] Iteration {n+1}")
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