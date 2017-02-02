from calc_probability.frequency_distribution import calc_freq_dist
import math
COUNT_WHEN_NOT_FOUND = 0
LOG_PROB_WHEN_NOT_FOUND = -1000

class EmissionModel(object):
    def __init__(self,corpus,k=0):
        self.emission_counts, self.vocabulary_size, self.hiddenstate_counts = calc_emission_state_distribution(corpus)
        self.add_k = k

    def get_counts(self,token_tag_pair):
        emission_count = self.emission_counts.get(token_tag_pair.get_tuples(), COUNT_WHEN_NOT_FOUND)
        hiddenstate_count = self.hiddenstate_counts.get(token_tag_pair.pos_tag, COUNT_WHEN_NOT_FOUND)
        return emission_count+self.add_k, hiddenstate_count+self.vocabulary_size*self.add_k

    def get_MLE_probabilty(self, token_tag_pair):
        numerator_count, denominator_count = self.get_counts(token_tag_pair)
        if numerator_count==0 or denominator_count==0:
            return LOG_PROB_WHEN_NOT_FOUND
        return math.log(float(numerator_count)/denominator_count,2)

    def get_score(self,document):
        return sum(self.get_MLE_probabilty(tokn_pos_pair) for tokn_pos_pair in document)

def calc_emission_state_distribution(corpus):
    document_list = []
    tokens_list = []
    hiddenstate_count = []
    for document in corpus:
        document_list += document.get_token_pos_tag_pair()
        tokens_list += document.get_token_corpus()
        hiddenstate_count += document.get_pos_tag_corpus()
    document_freq_dist = calc_freq_dist(document_list)
    hiddenstate_freq_dist = calc_freq_dist(hiddenstate_count)
    return document_freq_dist, len(set(tokens_list)), hiddenstate_freq_dist
