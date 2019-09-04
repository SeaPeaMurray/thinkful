def modes(data):
    entry_dict = {}
    for val in data:
        if val not in entry_dict.keys():
            entry_dict[val] = 1
        else:
            entry_dict[val] += 1
    maxmax = max(entry_dict.values())
    results = [key for key in entry_dict.keys() if entry_dict[key] == maxmax]
    if len(results) == len(data):
        return []
        return data
    elif results == None:
        return results
        return data
    else:
        return sorted(results)
        return data