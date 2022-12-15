import re
 
# helper function to perform sort for bipolar electrodes:
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]