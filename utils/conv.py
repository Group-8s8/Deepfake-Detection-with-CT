import cv2 as cv
import numpy as np

class Convolute():
	def convolve(img, s):
		if s%2 != 0:
			# making a kernel matrix of size (s,s) filled with '1'
			kmat = np.ones((s,s), np.int32)
		else:
			return False

		# k is the final kernel matrix with (1/s*s). This is done to minimize the effect of the kernel.
		k = (1/s*s)*kmat

		# a numpy step to get all the coordinated of the input image to avoid lag with a for loop.
		indices = np.where(img != [0])
		coordinates = zip(indices[0], indices[1])
		
		for x,y in coordinates:
			isum = 0
			for p in range(s):
				q=p
				try:
					img[x + p][y + q]
				except IndexError:
					# this error is raised when kernel reaches an edge of the image, stopping the loop.
					break
				# this is the Equation No. 1 from the paper.
				isum += k[p][q] * img[x + p][y + q]

			img[x][y] = isum
		
		# returns convulted image.
		return img