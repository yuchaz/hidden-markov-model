from Classes.LanguageModel import LanguageModel
from Classes.EmissionModel import EmissionModel

class HmmModel(object):
    def __init__(self, corpus, n):
        POS_tag_corpus = [doc.get_pos_tag_corpus() for doc in corpus]
        self.language_model = LanguageModel(POS_tag_corpus, n)
        self.emmision_model = EmissionModel(corpus)
