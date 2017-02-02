import numpy, math
from preprocess.generate_ngram import generate_ngram, preprocess_tokens
from calc_probability.frequency_distribution import calc_freq_dist
COUNT_WHEN_NOT_FOUND = 0
LOG_PROB_WHEN_NOT_FOUND = -1000

class LanguageModel(object):
    def __init__(self, corpus, n, k=0):
        self.joint_count, self.self_count = calc_ngram_probability_distribution(corpus, n)
        self.vocabulary_list, self.vocabulary_size = get_vocabulary_size(corpus)
        self.ngram = n
        self.add_k = k

    def get_counts(self, token):
        joint_count = self.joint_count.get(token, COUNT_WHEN_NOT_FOUND)
        self_count = self.self_count if self.ngram==1 else self.self_count.get(tuple(token[:self.ngram-1]), COUNT_WHEN_NOT_FOUND)
        return joint_count+self.add_k, self_count+self.vocabulary_size*self.add_k

    def get_MLE_probabilty(self, token):
        numerator_count, denominator_count = self.get_counts(token)
        if numerator_count==0 or denominator_count==0:
            return LOG_PROB_WHEN_NOT_FOUND
        return math.log(float(numerator_count)/denominator_count,2)

    def get_score(self,raw_tokens):
        tokens = preprocess_tokens(raw_tokens,self.ngram)
        ngram_tokens = generate_ngram(tokens, self.ngram)
        return sum(self.get_MLE_probabilty(token) for token in ngram_tokens)


# Functions below should not be used in other place...

def calc_ngram_probability_distribution(corpus, n):
    n_gram_list = []
    n_1_gram_list = []
    for raw_tokens in corpus:
        tokens = preprocess_tokens(raw_tokens, n)
        n_gram_model = generate_ngram(tokens,n)
        n_gram_list.extend(n_gram_model)
        if n != 1:
            n_1_gram_model = generate_ngram(tokens,n-1)
            n_1_gram_list.extend(n_1_gram_model)

    n_gram_freq_dist = calc_freq_dist(n_gram_list)
    vocabulary_size = get_vocabulary_size(corpus)
    if n != 1:
        n_1_gram_freq_dist = calc_freq_dist(n_1_gram_list)
        return n_gram_freq_dist, n_1_gram_freq_dist
    else:
        return n_gram_freq_dist, len(n_gram_list)

def get_vocabulary_size(corpus):
    unigram_list = [token for sentence in corpus for token in preprocess_tokens(sentence,1)]
    unigram_freq_dist = calc_freq_dist(unigram_list)
    return unigram_freq_dist.keys(), len(unigram_freq_dist)
