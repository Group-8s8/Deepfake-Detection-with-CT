from numpy import asarray

class Normal():
	def normalize(img):
		# Image is normalized to values between 0 and 1 so that calculations will be easier.
		pixels = asarray(img)
		pixels = pixels.astype('float32')
		# 255 is max value. so (any number < 255)/255 => 0 < number < 1
		pixels /= 255.0
		
		return pixels