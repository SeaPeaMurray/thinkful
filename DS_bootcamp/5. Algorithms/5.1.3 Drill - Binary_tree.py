import numpy as np
vals = np.random.randint(1, 100, 15)

def replace_with_child(list_):
	changes = 0
	for i, j in enumerate(list_):
		l = i * 2 + 1
		r = i * 2 + 2
		try:
			if list_[l] > j:
				list_[i], list_[l] = list_[l], list_[i]
				changes += 1
		except:
			pass
		try:
			if list_[r] > j:
				list_[i], list_[r] = list_[r], list_[i]
				changes += 1
		except:
			pass

	if changes == 0:
		return tuple(list_)
	else:
		return list_

new_vals = replace_with_child(vals)
while type(new_vals) != tuple:
	new_vals = replace_with_child(new_vals)
print(vals)
print(new_vals)