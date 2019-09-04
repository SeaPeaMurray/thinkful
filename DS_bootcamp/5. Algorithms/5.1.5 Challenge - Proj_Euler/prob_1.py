import numpy as np

sum_list = []
for val in range(1000):
	if val % 3 == 0 or val % 5 == 0:
		sum_list.append(val)

print(sum(sum_list))