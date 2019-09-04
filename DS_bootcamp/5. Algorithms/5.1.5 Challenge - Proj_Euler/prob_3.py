import numpy as np

targ = 600851475143

targ_half = int(targ / 2)

# fact_list = []
# for val in range(1, targ_half):
# 	if targ % val == 0:
# 		fact_list.append(val)

fact_list = [1, 71, 839, 1471, 6857, 59569, 104441, 486847, 1234169, 5753023, 10086647, 87625999, 408464633, 716151937]

not_prime = []
for val in fact_list:
	for i in range(2, int(val / 2)):
		if val % i != 0:
			pass
		else:
			print("{} is divisible by {}.".format(val, i))
			not_prime.append(val)
			break

print(max([val for val in fact_list if val not in not_prime]))	