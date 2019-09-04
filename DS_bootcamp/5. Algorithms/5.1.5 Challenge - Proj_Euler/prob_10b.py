import numpy as np

low_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def isprime(val):
	for i in np.arange(3, int(np.sqrt(val)) + 2, 2):
		if val % i != 0:
			continue
		elif val % i == 0:
			return None
	return val

home = [isprime(val) for val in range(49, 2000000, 2) if isprime(val) != None]

sum_primes = sum(low_primes) + sum(home)
print(sum_primes)