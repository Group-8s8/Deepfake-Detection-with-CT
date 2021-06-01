import cv2 as cv
import numpy as np

from numpy import asarray
from PIL import Image

class Initialize():
	def all(img):
		N = 3 #Kernel Size
		SIGMAZERO = 0.5
		#initializing R,P,W Matrices with 0. same size as img
		R = np.zeros(img.shape, np.float32)
		P = np.zeros(img.shape, np.float32)
		W = np.zeros(img.shape, np.float32)
		#Setting K with 1/(N*N).
		K = np.random.randint(3,size = (N,N)).astype("float32")

		#setting p0 and coordinates
		indices = np.where(img != [0])
		COORDINATES = zip(indices[0], indices[1])
		PZERO = 1/256
		ALPHA = 1
		return SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA

class EM():
	def algorithm(img):
		SIGMAZERO,R,P,W,K,PZERO,COORDINATES,ALPHA = Initialize.all(img)
		BETA = 0.2
		sigma = SIGMAZERO
		print(K)
		for n in range(0, 2):
			print(f"In loop {n+1}")
			#e-step
			a,b,c,d,e,f,COORDINATES,g = Initialize.all(img)
			for x,y in COORDINATES:
				sumo = 0
				for p in range(-ALPHA,ALPHA+1):
					#p = pn# + 1
					for q in range(-ALPHA,ALPHA+1):
						#q = qn# + 1
						try:
							sumo += K[p][q] * img[x + p][y + q]
						except IndexError:
							pass
				temp = img[x][y] - sumo
				if temp < 0:
					R[x][y] = temp * -1
				else:
					R[x][y] = temp
				P[x][y] = GD.guassian_distribution(R[x][y]**2, sigma)
				try:
					W[x][y] = P[x][y]/(P[x][y] + PZERO)
				except IndexError:
					print("Error at PZ.")
					pass

			print("e-step done")
			#m-step
			a,b,c,d,e,f,COORDINATES,g = Initialize.all(img)

			tsum = 0
			bsum = np.zeros(shape = (3,3),dtype = np.float32)
			csum = np.zeros(shape = (3,3),dtype = np.float32)
			BETA = np.zeros(shape = (3,3),dtype = np.float32)

			#program is really slow in the below loop.
			
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
					K[p][q] = (BETA[p][q] - csum[p][q])/bsum[p][q]
			print(K)

			print("m-step progress")

			sigsum = 0
			wsum = 0
			a,b,c,d,e,f,COORDINATES,g = Initialize.all(img)
			for x,y in COORDINATES:
				sigsum += W[x][y] * R[x][y]**2
				wsum += W[x][y]

			if wsum < 0:
				wsum = wsum * -1
			sigmasqr = sigsum/wsum

			sigma = np.sqrt(sigmasqr)
			print(sigma)
			print("m-step done")
		print(K)

	def make_beta(img, W, p,q, x, y):
		isum = 0
						
		return isum

class GD():
	def guassian_distribution(x, sigma):
		_z = x / (sigma)**2
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
		pixels = asarray(img)
		pixels = pixels.astype('float32')
		pixels /= 255.0
		
		return pixels


if __name__ == '__main__':
	img = 'data/Fake/ex.png'
	image = Image.open(img)
	r,g,b = image.split()
	bnormal = Normal.normalize(b)
	EM.algorithm(bnormal)
	#Normal.normalize(g)
	#Normal.normalize(r)
	