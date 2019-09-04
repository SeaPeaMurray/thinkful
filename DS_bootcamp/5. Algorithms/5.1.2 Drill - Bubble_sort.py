import numpy as np

class bubble:

	def __init__(self, array):
		self.a = array
		# Count provides an incremental to track swaps
		self.count = 0
		# Changes provides a change count before resetting
		self.changes = None

	def scan(self):
		for val in range(0, len(self.a) - 1):
			if self.a[val] > self.a[val + 1]:
				# Swap values
				self.a[val + 1], self.a[val] = self.a[val], self.a[val + 1]
				# Increment count
				self.count += 1
			elif val == len(self.a) - 2:
				# Preserve count before resetting for while loop
				self.changes = self.count
				# Check if any swaps were made
				if self.count == 0:
					return self
					break
				self.count = 0
		return self

	def sorting(self):
		while self.changes != 0:
			self.scan()
		return self

# Give the bubble sort a few test runs
for val in range(5):
	bb = bubble(np.random.randint(1, 100000, 500))
	print(bb.sorting().a)