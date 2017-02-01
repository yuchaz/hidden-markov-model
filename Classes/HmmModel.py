from Classes.LanguageModel import LanguageModel
from Classes.EmissionModel import EmissionModel

class HmmModel(object):
    def __init__(self, corpus, n):
        POS_tag_corpus = [doc.get_pos_tag_corpus() for doc in corpus]
        self.language_model = LanguageModel(POS_tag_corpus, n)
        self.emission_model = EmissionModel(corpus)
        self.ngram = n
    def e_score(self,token_tag_pair,k=0):
        return self.emission_model.get_MLE_probabilty(token_tag_pair,k)
    def q_score(self,token,k=0):
        return self.language_model.get_MLE_probabilty(token,k)
