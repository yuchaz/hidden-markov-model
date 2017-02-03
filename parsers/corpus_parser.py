import os, json, ConfigParser
import parsers.datapath_parser as dp
from Classes.Document import Document
from Classes.Corpus import Corpus

config_path = './config.ini'
section = 'path'
data_dir = 'data_dir'

path_dict = {
    'train': dp.get_train_set_path(),
    'dev': dp.get_dev_set_path(),
    'test': dp.get_test_set_path(),
    'bonus': dp.get_bonus_set_path()
}

def corpus_parser(*filedirs):
    corpus = []
    for filedir in list(filedirs):
        with open(filedir, 'r') as fn:
            corpus += [Document(json.loads(token_tag_pair)) for token_tag_pair in fn]
        fn.close()
    return corpus

def get_train_corpus():
    return corpus_parser(path_dict['train'])

def get_dev_corpus():
    return corpus_parser(path_dict['dev'])

def get_test_corpus():
    return corpus_parser(path_dict['test'])

def get_bonus_corpus():
    return corpus_parser(path_dict['bonus'])

def get_corpus_by_tag(*tags):
    path_list = [path_dict.get(tag) for tag in list(tags)]
    if len(path_list) == 0: raise ValueError('You should only input train, dev, test or bonus')
    return corpus_parser(ratio,shuffle,*path_list)

if __name__ == '__main__':
    import pdb; pdb.set_trace()
