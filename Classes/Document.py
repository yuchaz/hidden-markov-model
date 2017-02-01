class Document(object):
    def __init__(self, sentence):
        self.token_pos_tag_pairs = [TokenPosTagPair(word[0], word[1]) for word in sentence]

    def __iter__(self):
        for token_pos_tag_pair in self.token_pos_tag_pairs:
            yield tokens_tag_pair

    def get_pos_tag_corpus():
        return [tokn_pos_pair.pos_tag for tokn_pos_pair in self.token_pos_tag_pairs]

class TokenPosTagPair(object):
    def __init__(self, token, pos_tag):
        self.token = token
        self.pos_tag = pos_tag
    def get_tuples():
        return self.token, self.pos_tag
