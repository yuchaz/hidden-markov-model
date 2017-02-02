from Classes.LanguageModel import LanguageModel
from Classes.EmissionModel import EmissionModel

class HmmModel(object):
    def __init__(self, corpus, n, k_lan_model=0, k_emiss_model=0):
        POS_tag_corpus = [doc.get_pos_tag_corpus() for doc in corpus]
        self.language_model = LanguageModel(POS_tag_corpus, n, k_lan_model)
        self.emission_model = EmissionModel(corpus, k_emiss_model)
        self.ngram = n
    def e_score(self,token_tag_pair):
        return self.emission_model.get_MLE_probabilty(token_tag_pair)
    def q_score(self,token):
        return self.language_model.get_MLE_probabilty(token)
