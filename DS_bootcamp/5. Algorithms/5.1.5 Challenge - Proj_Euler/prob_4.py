import numpy as np

pal_list = []
for val in range(1000, 100, -1):
	for val_ in range(1000, 100, -1):
		prod = str(val * val_)
		prod_ = [val for val in reversed(prod)]
		if [val for val in prod] == prod_:
			pal_list.append(prod)

pal_list = [int(pal) for pal in pal_list]
print(max(pal_list))