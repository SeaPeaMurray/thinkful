asdf = (sum(range(1, 101)))
asdf ** 2

asdf2 = 0
for val in range(1, 101):
	asdf2 += val ** 2

print(asdf ** 2 - asdf2)