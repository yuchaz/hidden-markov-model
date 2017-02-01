import re

START_SYMBOL = '<START>'
STOP_SYMBOL = '<STOP>'
UNK_SYMBOL = '<UNK>'

def generate_ngram(tokens,n):
    return zip(*[tokens[i:] for i in range(n)])

def preprocess_tokens(tokens, n):
    return [START_SYMBOL for i in range(n-1)]+tokens+[STOP_SYMBOL]

def replace_with_UNK(corpus, words_to_replace):
    if len(words_to_replace) == 0:
        return corpus
    else:
        replaced_regex = re.compile('|'.join(map(re.escape, words_to_replace)))
        return [replaced_regex.sub(UNK_SYMBOL,sentence) for sentence in corpus]
