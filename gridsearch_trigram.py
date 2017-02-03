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
    'ratio': 1,
    'shuffle': True
}

default_params = {
    "unk_threshold": 1,
    "unk_oov_ratio": 0.005,
    "trans_prob": 0.8,
    "k_lan_model": 0.01,
    "k_emiss_model": 0.01
}
updated_params = {
    "k_lan_model": [1e-3,1e-2,1e-1,1,10],
    "k_emiss_model": [1e-3,1e-2,1e-1,1,10],
}

def main():
    best_model = GridsearchWrapper(train_list, dev_list,train_corpus_param,
                                   dev_corpus_param, 3, default_params,updated_params)
    print best_model.best_model


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
