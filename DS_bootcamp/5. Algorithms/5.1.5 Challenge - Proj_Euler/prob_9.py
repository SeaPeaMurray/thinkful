import numpy as np

l1 = np.arange(1, 400)
l2 = np.arange(1, 400)
l3 = []

for i in l1:
	for j in l2:
		hyp = np.sqrt(i**2 + j**2)
		if hyp == int(hyp):
			l3.append([i, j, hyp])
		
print([np.prod(l) for l in l3 if sum(l) == 1000])