from Classes.Corpus import Corpus
from Classes.HmmModel import HmmModel
from Classes.GridsearchWrapper import get_accuracy

def main():
    train_corpus = Corpus.trainCorpus()
    test_corpus = Corpus.testCorpus(shuffle=True)
    train_corpus.replace_oov_with_UNK(unk_threshold=1,
                                      unk_oov_ratio=0.005,
                                      trans_prob=0.7,)

    test_corpus.replace_oov_with_UNK(unk_threshold=1, unk_oov_ratio=0.005,
                                     trans_prob=0.7, known_unk_dict=train_corpus.to_unk_dict)

    hmm = HmmModel(corpus=train_corpus, n=3, k_lan_model=0.01, k_emiss_model=0.001)
    accuracy = get_accuracy(hmm,test_corpus)
    print accuracy
if __name__ == '__main__':
    main()
