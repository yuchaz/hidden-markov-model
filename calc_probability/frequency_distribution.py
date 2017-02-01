from collections import defaultdict

def calc_freq_dist(token_list):
    dictionary = defaultdict(int)
    for token in token_list:
        dictionary[token] += 1
    return dictionary
