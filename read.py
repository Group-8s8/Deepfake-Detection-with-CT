import cv2 as cv
from utils.csplit import Colors

class Read():
	def start_read():
		img = cv.imread('data/Photos/cats.jpg')
		blue_1, green_1, red_1 = Colors.colorsplit(img, 1)
		
		merged = cv.merge([blue_1,green_1,red_1])
		
		cv.imshow("blue_1", blue_1)
		cv.imshow("green_1", green_1)
		cv.imshow("red_1", red_1)
		cv.imshow("merged", merged)
		cv.waitKey(0)

if __name__ == '__main__':
	Read.start_read()