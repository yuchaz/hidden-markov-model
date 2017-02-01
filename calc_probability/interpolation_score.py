import math
LOG_PROB_WHEN_NOT_FOUND = -1000

def calc_interpolation_score(tokens, LanguageModels, weights, k=0):
    score = sum(2**LanguageModels[i].get_score(tokens,k) * weights[i] for i in range(len(weights)))
    if score == 0:
        return LOG_PROB_WHEN_NOT_FOUND
    return math.log(score,2)
