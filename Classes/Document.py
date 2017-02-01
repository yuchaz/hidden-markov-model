class Document(object):
    def __init__(self, token_tag_pairs):
        self.tokens = [word[0] for word in token_tag_pairs]
        self.tags = [word[1] for word in token_tag_pairs]
