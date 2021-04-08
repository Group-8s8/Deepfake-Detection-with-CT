import cv2 as cv
from utils.csplit import Colors
from utils.conv import Convolute
from copy import copy

# im lazy to comment on this part. All i do here is test stuff.. read individual utils pgms and read the comments for more info.

class Read():
	def start_read():
		img = cv.imread('data/Fake/ex.png')
		#og = cv.imread('data/Fake/og.png')
		blue_1, green_1, red_1 = Colors.colorsplit(img, 1)
		#ogb, green_1, red_1 = Colors.colorsplit(og, 1)
		
		#merged = cv.merge([blue_1,green_1,red_1])
		bb = copy(blue_1)
		#ogg = copy(ogb)
		#co = Convolute.convolve(ogg,3)
		convimg = Convolute.convolve(bb,3)
		cv.imshow("conv", convimg)
		#cv.imshow("convog", co)
		cv.imshow("blue_1", blue_1)
		#cv.imshow("blueg", ogb)#
		#cv.imshow("green_1", green_1)
		#cv.imshow("red_1", red_1)
		#cv.imshow("merged", merged)
		cv.waitKey(0)

if __name__ == '__main__':
	Read.start_read()