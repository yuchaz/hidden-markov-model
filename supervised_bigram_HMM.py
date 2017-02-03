from Classes.HmmModel import HmmModel
from Classes.Corpus import Corpus
import parsers.corpus_parser as cp
import time

def get_accuracy(hmm, corpus):
    total_accu_count = 0
    total_count = 0
    for doc in corpus:
        doc.run_viterbi(hmm)
        accu_count, ttl_count = doc.evaluate()
        total_accu_count += accu_count
        total_count += ttl_count
    return float(total_accu_count)/total_count

def main():
    train_corpus = Corpus.trainCorpus()
    dev_corpus = Corpus.devCorpus(ratio=0.1,shuffle=True)
    train_corpus.replace_oov_with_UNK()
    dev_corpus.replace_oov_with_UNK()

    hmm_model = HmmModel(train_corpus,n=2,k_lan_model=0,k_emiss_model=0)
    # train_accuracy = get_accuracy(hmm_model, train_corpus)
    dev_accuracy = get_accuracy(hmm_model, dev_corpus)
    # print train_accuracy
    print dev_accuracy


if __name__ == '__main__':
    try:
        start = time.time()
        main()
        print time.time()-start
    except:
        import sys, pdb, traceback
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
