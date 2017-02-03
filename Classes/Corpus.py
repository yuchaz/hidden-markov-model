import re
from numpy import random
import parsers.corpus_parser as cp
from calc_probability.frequency_distribution import calc_freq_dist

UNK = '<UNK>'
NAME = '<@NAME>'
HASHTAG = '<#HASHTAG>'
URL = '<URL>'

class Corpus(object):
    def __init__(self, documents, ratio=1.0, shuffle=False):
        self.documents = documents
        self.to_unk_dict = {}
        self.actual_size = len(self.documents)
        self.used_size = int(ratio * self.actual_size)
        self.index_map = random.permutation(self.actual_size)[:self.used_size] \
                         if shuffle==True \
                         else range(self.actual_size)[:self.used_size]

    @classmethod
    def trainCorpus(cls,ratio=1.0,shuffle=False):
        documents = cp.get_train_corpus()
        return cls(documents,ratio,shuffle)
    @classmethod
    def devCorpus(cls, ratio=1.0,shuffle=False):
        documents = cp.get_dev_corpus()
        return cls(documents,ratio,shuffle)
    @classmethod
    def testCorpus(cls, ratio=1.0,shuffle=False):
        documents = cp.get_test_corpus()
        return cls(documents,ratio,shuffle)
    @classmethod
    def bonusCorpus(cls, ratio=1.0,shuffle=False):
        documents = cp.get_bonus_corpus()
        return cls(documents,ratio,shuffle)
    @classmethod
    def combinedCorpus(cls, ratio, shuffle, *tags):
        documents = cp.get_corpus_by_tag(*tags)
        return cls(documents,ratio,shuffle)

    def __iter__(self):
        for idx in self.index_map:
            yield self.documents[idx]

    def __len__(self):
        return self.used_size

    def replace_oov_with_UNK(self, unk_threshold=1, unk_oov_ratio=0.005,
                             trans_prob=0.7, known_unk_dict={}):
        to_unk_dict = self.generate_to_unk_dict(unk_threshold, unk_oov_ratio,
                                                trans_prob, known_unk_dict)

        if len(to_unk_dict.keys()) == 0:
            return
        else:
            for document in self:
                document.replaced_with(to_unk_dict)
            return

    def generate_to_unk_dict(self, unk_threshold, unk_oov_ratio, trans_prob, known_unk_dict={}):
        token_freq_dist = self.find_freq_dist()
        vocabulary_size = len(token_freq_dist)
        self.to_unk_dict = self.to_unk_dict_factory(token_freq_dist, vocabulary_size,
                                                unk_threshold, unk_oov_ratio, trans_prob)
        self.to_unk_dict.update(known_unk_dict)

        return self.to_unk_dict

    def find_freq_dist(self):
        tokens_list = []
        for document in self:
            tokens_list += document.get_token_corpus()
        return calc_freq_dist(tokens_list)

    def to_unk_dict_factory(self,token_freq_dist, vocabulary_size, unk_threshold, unk_oov_ratio, trans_prob):
        unk_count = 0
        to_unk_dict = {}
        regex_not_to_unk = re.compile(r'^(?!https://|http://|#|@).*$')
        for k,v in random.permutation(token_freq_dist.items()):
            if regex_not_to_unk.match(k) and int(v) <= unk_threshold and \
                    unk_count <= unk_oov_ratio * vocabulary_size and \
                    random.uniform(0,1) < trans_prob:
                to_unk_dict.update({k:UNK})
                unk_count += 1
            elif k[0]=='@':
                to_unk_dict.update({k:NAME})
            elif k[0]=='#':
                to_unk_dict.update({k:HASHTAG})
            elif re.match(r'^(http://|https://).*$', k):
                to_unk_dict.update({k:URL})

        return to_unk_dict
