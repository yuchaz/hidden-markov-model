import os
import json

data_dir = './data'
train_corpus = 'train'
dev_corpus = 'dev'
test_corpus = 'test'

class Document(object):
    def __init__(self, token_tag_pairs):
        self.tokens = [word[0] for word in token_tag_pairs]
        self.tags = [word[1] for word in token_tag_pairs]

def read_file(filedir):
    with open(filedir, 'r') as fn:
        corpus = [Document(json.loads(token_tag_pair)) for token_tag_pair in fn]
    fn.close()
    return corpus
