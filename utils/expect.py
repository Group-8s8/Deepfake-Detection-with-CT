#another try with algorithm

import numpy as np

class Initialize():
	def all(img):
		N = 3 #Kernel Size
		SIGMAZERO = 0 #DONT KNOW WHAT GOES HERE. 
		#initializing R,P,W Matrices with 0. same size as img
		R = np.zeros(img.shape, np.int8)
		P = np.zeros(img.shape, np.int8)
		W = np.zeros(img.shape, np.int8)
		#Setting K with 1/(N*N).
		KMAT = np.ones((N,N), np.int32)
		K = (1/N*N)*KMAT

		#setting p0 and coordinates
		indices = np.where(img != [0])
		COORDINATES = zip(indices[0], indices[1])
		PZERO = 1/len(COORDINATES)
		ALPHA = N #DONT KNOW WHAT IS ALPHA
		return SIGMAZERO,R,P,W,K,PZERO,COORDINATES

class EM():
	def algorithm(img):
		SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA = Initialize.all(img)

		for n in range(1,100):
			#e-step
			for x,y in COORDINATES:
				isum = 0
				for p in range(-ALPHA,ALPHA):
					try:
						img[x + p][y + p]
					except IndexError:
						# this error is raised when kernel reaches an edge of the image, stopping the loop.
						break
					# this is the Equation No. 1 from the paper.
					isum += K[p][p] * img[x + p][y + p]
				temp = img[x][y] - isum
				if temp < 0:
					R[x][y] = temp * -1 
				else:
					R[x][y] = temp

				#gd eq
				sigma = SIGMAZERO # DONT KNOW WHAT HAPPENS HERE. THIS VALUE SHOULD CHANGE ON EVERY ITERATION, BUT WITH WHAT?
				
				P[x][y] = GD.guassian_distribution(R[x][y], sigma)

				W[x][y] = P[x][y]/(P[x][y] + PZERO)

			#m-step
				#to be completed.



class GD():
	def guassian_distribution(x, sigma):
		# equation 3 from the paper.
		_z = (x / sigma)**2
		_exp = np.exp(_z)
		_b = 2 * np.pi * sigma**2
		p = 1 / np.sqrt(_b * _exp)
		return p