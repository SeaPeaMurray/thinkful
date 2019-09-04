def isprime(val):
	for n in range(2, (val // 2) + 1):
		if val % n != 0:
			if n == val // 2:
				return True
			continue
		else:
			return False
			
def nextprime(myval):
	myval += 2
	while isprime(myval) == False:
		myval += 2
	return myval

def len_list(list_, length):
	mylist = list_
	while len(mylist) < length:
		val_ = mylist[-1]
		mylist.append(nextprime(val_))
	return mylist

prime_list = [1, 3, 5, 7, 11]
verdict = len_list(prime_list, 10001)
print(verdict[-1])