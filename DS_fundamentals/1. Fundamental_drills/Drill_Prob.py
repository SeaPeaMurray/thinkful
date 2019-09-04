import numpy as np
chances = ['HTTH', 'HHHH', 'TTHH']
# Each chance is one possibility out of four flips yielding 2 ** 4 possibilities
print('The chance of any single outcome after four flips is {}'.format(1 / (2 ** 4)))

people_list = [24, 21]
print('The chance of not choosing a man: {}'.format(1 - (21/45)))

crash = 0.00005
flying = 0.1
print('The probability of a crash in the next year for Bernice is {}'.format(crash * flying))

# Conclusions from a segment that potentially only represents five percent of the population could represent the users outside two standard deviations, since within two SD of the mean we have the 95 percent confidence interval. Furthermore, the survey data is voluntary and makes for a bad sample.