import numpy as np
import cv2 as cv

class Initialize():
	def all(img):
		N = 3 #Kernel Size
		SIGMAZERO = 1 # Initializing with random (1)
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
		PZERO = 1/len(indices)
		ALPHA = 2 #DONT KNOW WHAT IS ALPHA. Passing value <= kernel size N
		return SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA

class EM():
	def algorithm(img):
		SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA = Initialize.all(img)

		sigma = SIGMAZERO 

		for n in range(1,100):
			print("in loop")
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
				for i in range(0,len(temp)):
					if temp[i] < 0:
						temp[i] = temp[i] * -1 
				R[x][y] = temp
				#gd eq
				P[x][y] = GD.guassian_distribution(R[x][y], sigma)
				W[x][y] = P[x][y]/(P[x][y] + PZERO)
			print(W)
			print(n)
			sigma = sigma + 1
		 # Why is everything 0
		'''
			#m-step
			
			sigmasqr = 1 #"Some Value is calcuted here to give the varience (sigma square)" # DONT KNOW HOW THIS IS CALCULATED

			sigma = np.sqrt(sigmasqr) #this sigma value is passed to next iteration
			
			# checking if it satisfies eq 1 and concluding whether its from Model 1 or 2.
			for i in range(0,len(img[x][y])):
				if img[x][y][i] == EM.convolve(img)[i]:
					print(1)
					return img
					pass # belongs to M1
				else:
					return img
					pass # belongs to M2

		
			# do eq 7

			# END OF EM Algorithm
		'''
	def convolve(img):
		SIGMAZERO,R,P,W,K,PZERO,NOT_USED_COORDS,ALPHA = Initialize.all(img)

		indices = np.where(img != [0])
		COORDINATES = zip(indices[0], indices[1])

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

			img[x][y] = isum
		
		# returns convulted image.
		return img


class GD():
	def guassian_distribution(x, sigma):
		# equation 3 from the paper.
		_z = []
		_exp = []
		p = []
		for i in range(0,len(x)):
			_z.append((x[i] / sigma)**2)
			if (_z[i]) > 709:
				 _z[i] = 0 # done to avoid fp precision issues.
			_exp.append(np.exp(_z[i]))
			_b = 2 * np.pi * sigma**2
			p.append(1 / np.sqrt(_b * _exp[i]))

		return p

if __name__ == '__main__':
	img = cv.imread('../data/Fake/ex.png')
	em = EM.algorithm(img)
	cv.imshow("merged", em)
	cv.waitKey(0)