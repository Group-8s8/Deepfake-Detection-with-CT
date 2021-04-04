import cv2 as cv

class Colors:

	def colorsplit(img, depth):
		b,g,r = cv.split(img)
		if depth == 1:
			return b,g,r
		elif depth == 2:
			blank = np.zeros(img.shape, dtype='uint8')
			blue = colors.colormerge(b,blank,blank)
			green = colors.colormerge(blank,g.blank)
			red = colors.colormerge(blank,blank,r)
			return blue, green, red