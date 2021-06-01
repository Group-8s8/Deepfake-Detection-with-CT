#THis is just a notepad.

			ksum = 0
			tsum = 0
			bsum = 0
			csum = 0
			
			for p in range(-ALPHA,ALPHA+1):
				for q in range(-ALPHA,ALPHA+1):
					for x,y in COORDINATES:
						try:
							BETA += W[x][y] * img[x + p][y + q] * img[x][y]
						except IndexError:
							pass

						try:
							bsum += W[x][y] * (img[x - p][y - q])**2
						except IndexError:
							pass

						for pe in range(-ALPHA,ALPHA+1):
							for qe in range(-ALPHA,ALPHA+1):
								try:
									tsum += W[x][y] * img[x - p][y - q] * img[x + pe][y + qe]
								except IndexError:
									pass
					for pe in range(-ALPHA,ALPHA+1):
						for qe in range(-ALPHA,ALPHA+1):
							csum += K[pe][qe] * tsum
					K[p][q] = (BETA - csum)/bsum

			for p in range(-ALPHA,ALPHA+1):
				for q in range(-ALPHA,ALPHA+1):


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