from numpy import random

class Corpus(object):
    def __init__(self, documents, ratio=1.0, shuffle=False):
        self.documents = documents
        self.actual_size = len(documents)
        self.used_size = int(ratio * self.actual_size)
        self.index_map = random.permutation(self.actual_size)[:self.used_size] \
                         if shuffle==True \
                         else range(self.actual_size)[:self.used_size]

    def __iter__(self):
        for idx in self.index_map:
            yield self.documents[idx]

    def __len__(self):
        return self.used_size
