import math
COUNT_WHEN_NOT_FOUND = 0
LOG_PROB_WHEN_NOT_FOUND = -1000

class EmissionModel(object):
    def __init__(self,corpus):
        self.emission_counts, self.vocabulary_size, self.hiddenstate_counts = calc_emission_state_distribution(corpus)

    def get_counts(self,token_tag_pair,k=0):
        emission_count = self.emission_counts.get(token_tag_pair.get_tuples(), COUNT_WHEN_NOT_FOUND)
        hiddenstate_count = self.hiddenstate_counts.get(token_tag_pair.token, COUNT_WHEN_NOT_FOUND)
        return emission_count+k, hiddenstate_count+self.vocabulary_size*k

    def get_MLE_probabilty(self, token_tag_pair, k=0):
        numerator_count, denominator_count = self.get_counts(token_tag_pair,k)
        if numerator_count==0 or denominator_count==0:
            return LOG_PROB_WHEN_NOT_FOUND
        return math.log(float(numerator_count)/denominator_count,2)

    def get_score(self,document,k=0):
        return sum(self.get_MLE_probabilty(tokn_pos_pair,k) for tokn_pos_pair in document)

def calc_emission_state_distribution(corpus):
    document_list = []
    tokens_list = []
    hiddenstate_count = []
    for document in corpus:
        document_list += document.token_pos_tag_pairs
        tokens_list += document.token_pos_tag_pairs.token
        hiddenstate_count += document.token_pos_tag_pairs.pos_tag
    document_freq_dist = calc_freq_dist(document_list)
    return document_freq_dist, len(set(tokens_list)), len(set(hiddenstate_count))
