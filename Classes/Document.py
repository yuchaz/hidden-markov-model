STOP_SYMBOL = '<STOP>'

class Document(object):
    def __init__(self, sentence):
        self.token_pos_tag_pairs = [TokenPosTagPair(word[0], word[1]) for word in sentence]
        self.end_index = len(sentence)
        self.predicted_pos_tags = []*len(sentence)
        self.doc_length = len(sentence)

    def __iter__(self):
        for token_pos_tag_pair in (self.token_pos_tag_pair[idx]
                                   for idx in range(self.end_index)
                                   if idx < self.end_index):
            yield tokens_tag_pair

    def get_pos_tag_corpus():
        return [tokn_pos_pair.pos_tag for tokn_pos_pair in self.token_pos_tag_pairs]

    def pop_back():
        self.end_index -= 1

    def reset_end_index():
        self.end_index = self.doc_length

    def get_last_term():
        return self.token_pos_tag_pairs[self.end_index-1]

    def get_last_predicted_pos():
        return STOP_SYMBOL if self.end_index == self.doc_length else self.predicted_pos_tags[end_index]

    def update_predicted_pos_tags(newtag):
        self.predicted_pos_tags[self.end_index-1] = newtag

    def evaluate():
        return sum(1 for idx in range(len(self.token_pos_tag_pairs))
                   if self.token_pos_tag_pairs[i].pos_tag == self.predicted_pos_tags[i])

class TokenPosTagPair(object):
    def __init__(self, token, pos_tag):
        self.token = token
        self.pos_tag = pos_tag
    def get_tuples():
        return self.token, self.pos_tag
