from Classes.LanguageModel import LanguageModel
from Classes.EmissionModel import EmissionModel
from Classes.Document import TokenPosTagPair

class HmmModel(object):
    def __init__(self, corpus, n):
        POS_tag_corpus = [doc.get_pos_tag_corpus() for doc in corpus]
        self.language_model = LanguageModel(POS_tag_corpus, n)
        self.emission_model = EmissionModel(corpus)
        self.ngram = n

    def calc_viterbi_score(self, document):
        argmax_pos_tag, max_score = calc_max_score(document)
        document.update_predicted_pos_tags(argmax_pos_tag)
        if document.end_index == 1:
            document.reset_end_index()
            return max_score
        else:
            document.pop_back()
            return self.calc_viterbi_score(document)

    def calc_max_score(self, document, k=0):
        token = document.get_last_term().get_tuples().token
        last_predicted_POS = document.get_last_predicted_pos()
        e_score = self.emission_model.get_MLE_probabilty
        q_score = self.language_model.get_MLE_probabilty
        pos_choice = self.language_model.vocabulary_list

        return max(((pos_tag, e_score(TokenPosTagPair(token,pos_tag),k) *  q_score((pos_tag,last_predicted_POS),k) )
                   for pos_tag in pos_choice),
                   key=lambda p:p[1])
