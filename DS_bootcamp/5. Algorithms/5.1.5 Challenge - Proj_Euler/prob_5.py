import numpy as np

answer = None
starter = 2520
while answer == None:
	for val in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
		if starter % val == 0:
			if val == 20:
				answer = starter
				print('Answer found')
				break
			continue
		else:
			print('{} is indivible by {}.'.format(starter, val))
			starter += 2520
			break
print(answer)