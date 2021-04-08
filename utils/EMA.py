import cv2 as cv
import numpy as np

class EM():
	def guassian_distribution(x, mu, sigma):
		# equation 3 from the paper.
		_z = ((x - mu) / sigma)**2
		_exp = np.exp(_z)
		_b = 2 * np.pi * sigma**2
		p = 1 / np.sqrt(_b * _exp)
		return p

	def bayes(pa,pb,pxa,pxb):
		b = (pxb * pb)/((pxb * pb) + (pxa * pa))
		a = 1 - b
		

if __name__ == '__main__':
	EM.guassian_distribution(1.1,1,2)
