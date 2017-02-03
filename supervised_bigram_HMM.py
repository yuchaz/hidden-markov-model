from Classes.Corpus import Corpus
from Classes.GridsearchWrapper import GridsearchWrapper
import time

train_list = ['train']
dev_list = ['dev']
train_corpus_param = {
    'ratio': 1.,
    'shuffle': False
}
dev_corpus_param = {
    'ratio': 0.1,
    'shuffle': True
}

default_params = {
    "unk_threshold": 1,
    "unk_oov_ratio": 0.005,
    "trans_prob": 0.7,
    "k_lan_model": 2,
    "k_emiss_model": 2
}
updated_params = {
    "unk_threshold": [1,3],
    "unk_oov_ratio": [1e-3,5e-3,1e-2],
    "trans_prob": [0.7],
    "k_lan_model": [1e-3,1e-2,1e-1,1,10],
    "k_emiss_model": [1e-3,1e-2,1e-1,1,10],
}

updated_params_1 = {
    "unk_threshold": [1,3],
    "unk_oov_ratio": [1e-3,5e-3],
    "trans_prob": [0.7,0.8],
    "k_lan_model": [1e-3,1e-2],
    "k_emiss_model": [1e-3,1e-2],
}

def main():
    # train_corpus = Corpus.trainCorpus()
    # dev_corpus = Corpus.devCorpus(shuffle=True)
    best_model = GridsearchWrapper(train_list, dev_list,train_corpus_param,
                                   dev_corpus_param, 2, default_params)
    print best_model.best_params_, best_model.best_score_

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
