import numpy as np

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