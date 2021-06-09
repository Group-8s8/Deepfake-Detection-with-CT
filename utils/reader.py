import numpy as np
from PIL import Image

from utils.normal import Normal
from utils.EM import EM

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