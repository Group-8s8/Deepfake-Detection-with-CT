import argparse

from utils.reader import Reader
from utils.classifier import Classifier

if __name__ == '__main__':
	# loads image from given path
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--fake', default=None)
	parser.add_argument('-r', '--real', default='data/Real/real.png')
	args = parser.parse_args()
	if args.fake is None:
		print("[ERROR] Use `-f <path_to_fakeimage>` argument.")
		exit()
	fake = args.fake
	real = args.real
	print("[INFO] Performing for fake image.")
	K_vector = Reader.read(fake)
	print("[INFO] Performing for real image.")
	gg = Reader.read(real)
	y = [int(item) for item in gg]
	print("[INFO] Performing Random Forrest Classification.")
	Classifier.classify(K_vector,y)