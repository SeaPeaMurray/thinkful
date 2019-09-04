from math import log2

# Put your calculations below

'''Of the 20 students, 12 are boys with and 8 are girls. 
Of the 12 boys, 4 are tall, 6 are medium and 2 are short. 
Of the 8 girls, 1 is tall, 2 are medium, and 5 are short.'''

h_tall = 4/5 * log2(5/4) + 1/5 * log2(5)
h_not_tall = 8/15 * log2(15/8) + 7/15 * log2(15/7)
print(1/4 * h_tall + 3/4 * h_not_tall)

h_med = 3/4 * log2(4/3) + 1/4 * log2(4)
h_not_med = 1/2 * log2(2) + 1/2 * log2(2)
print(2/5 * h_med + 3/5 * h_not_med)

# Example solution is below here. Don't peek.


