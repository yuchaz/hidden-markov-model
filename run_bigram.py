from Classes.Corpus import Corpus
from Classes.HmmModel import HmmModel
from Classes.GridsearchWrapper import get_accuracy

def main():
    train_corpus = Corpus.trainCorpus()
    train_corpus_star = Corpus.trainCorpus(ratio=0.1,shuffle=True)
    # test_corpus = Corpus.testCorpus(shuffle=True)
    train_corpus.replace_oov_with_UNK(unk_threshold=1,
                                      unk_oov_ratio=0.005,
                                      trans_prob=0.7,)

    train_corpus_star.replace_oov_with_UNK(unk_threshold=1, unk_oov_ratio=0.005,
                                     trans_prob=1, known_unk_dict=train_corpus.to_unk_dict)

    # test_corpus.replace_oov_with_UNK(unk_threshold=1, unk_oov_ratio=0.005,
    #                                  trans_prob=1, known_unk_dict=train_corpus.to_unk_dict)

    hmm = HmmModel(train_corpus, 2, 0.01, 0.001)
    accuracy = get_accuracy(hmm,train_corpus_star)
    # accuracy = get_accuracy(hmm,test_corpus)
    print accuracy
if __name__ == '__main__':
    main()
