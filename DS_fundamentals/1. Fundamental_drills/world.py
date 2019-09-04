trial = [["Grae Drake", 98110], ["Bethany Kok"], ["Alex Nussbacher", 94101], ["Darrell Silver", 11201]]

def user_contacts(data):
	my_dict = {}
	for entry in data:
		my_dict[entry[0]] = 1
	return my_dict

print(user_contacts(trial))