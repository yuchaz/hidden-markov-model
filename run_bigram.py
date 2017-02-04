from Classes.Corpus import Corpus
from Classes.HmmModel import HmmModel
from Classes.GridsearchWrapper import get_accuracy
import packages.confusion_matrix as cm

def main():
    train_corpus = Corpus.trainCorpus()
    test_corpus = Corpus.testCorpus(ratio=1,shuffle=True)
    train_corpus.replace_oov_with_UNK(unk_threshold=1,
                                      unk_oov_ratio=0.005,
                                      trans_prob=0.7,)

    test_corpus.replace_oov_with_UNK(unk_threshold=1, unk_oov_ratio=0.005,
                                     trans_prob=0.7, known_unk_dict=train_corpus.to_unk_dict)

    hmm = HmmModel(train_corpus, 2, 0.01, 0.001)
    accuracy = get_accuracy(hmm,test_corpus)
    confusion_matrix = cm.calc_confusion_matrix(hmm, test_corpus)
    cm.plot_confusion_matrix(confusion_matrix, hmm.possible_pos_choices )
    print accuracy

if __name__ == '__main__':
    main()
