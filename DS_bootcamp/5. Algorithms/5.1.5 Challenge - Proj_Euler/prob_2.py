import numpy as np

fib_list = [0, 1, 1, 2]

while max(fib_list) < 4000000:
	c = fib_list[-1] + fib_list[-2]
	if c > 4000000:
		break
	fib_list.append(c)

fib2_list = [val for val in fib_list if val % 2 == 0]
print(sum(fib2_list)) 