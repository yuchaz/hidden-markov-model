from collections import deque
STOP_SYMBOL = '<STOP>'
START_SYMBOL = '<START>'

class Document(object):
    def __init__(self, sentence):
        self.token_pos_tag_pairs = [TokenPosTagPair(word[0], word[1]) for word in sentence]
        self.end_index = 0
        self.predicted_pos_tags = deque([])
        self.doc_length = len(sentence)
        self.hidden_cells = []*self.doc_length
        self.hidden_cells[0] = {START_SYMBOL:Cell(1,'')}

    def __iter__(self):
        for token_pos_tag_pair in (self.token_pos_tag_pair[idx]
                                   for idx in range(self.end_index)
                                   if idx < self.end_index):
            yield tokens_tag_pair

    def get_pos_tag_corpus():
        return [tokn_pos_pair.pos_tag for tokn_pos_pair in self.token_pos_tag_pairs]

    def push_back():
        self.end_index += 1

    def pop_back():
        self.end_index -= 1

    def get_last_term():
        return self.token_pos_tag_pairs[self.end_index-1]

    def store_cells(self,idx,pos_tag,score,argmax_pos_tag):
        self.hidden_cells[idx] = {pos_tag: Cell(score, argmax_pos_tag)}

    def calc_pi_score(self, hmm, current_pos):
        current_token = self.get_last_term().token
        current_idx = self.end_index+1
        if self.end_index == 0:
            max_pi_score = hmm.e_score(TokenPosTagPair(current_token,current_pos), k) * \
                           hmm.q_score((START_SYMBOL,current_pos),k)
            predicted_prev_pos = START_SYMBOL
        else:
            pos_choices = hmm.language_model.vocabulary_list
            predicted_prev_pos, max_pi_score = \
                max(((pos_tag, hmm.e_score(TokenPosTagPair(current_token,current_pos),k) * \
                               hmm.q_score((pos_tag,current_pos),k) * \
                               self.hidden_cells[current_idx-1][pos_tag].score)
                               for pos_tag in pos_choice),
                               key=lambda p:p[1])
        self.store_cells(current_idx, current_pos, max_pi_score, predicted_prev_pos)

    def run_viterbi(self, hmm):
        if document.end_index <= self.doc_length:
            pos_choices = hmm.language_model.vocabulary_list \
                if document.end_index != self.doc_length else [STOP_SYMBOL]

            for target_pos_tag in pos_choices:
                document.calc_pi_score(hmm,target_pos_tag)
            document.push_back()
            self.run_viterbi(hmm)
        else:
            self.back_propagation()

    def back_propagation():
        if document.end_index > self.doc_length:
            document.pop_back()
            back_propagation

        elif document.end_index > 0:
            current_pos = STOP_SYMBOL \
                            if  document.end_index == document.doc_length \
                            else self.predicted_pos_tags[0]

            score, back_pos = self.hidden_cells[document.end_index+1][current_pos].get_tuples()
            self.predicted_pos_tags.appendleft(back_pos)
            document.pop_back()
            self.back_propagation()
        else: return

    def get_score_and_predicted_pos_list():
        return self.hidden_cells[-1][STOP_SYMBOL].score, elf.predicted_pos_tags

    def evaluate():
        if len(self.predicted_pos_tags) != self.doc_length:
            raise ValueError(
                "You should not called this before running Viterbi algorithms"
            )

        accurate_count = sum(1 for idx in range(len(self.token_pos_tag_pairs))
                   if self.token_pos_tag_pairs[i].pos_tag == self.predicted_pos_tags[i])
        totol_count = len(self.token_pos_tag_pairs)
        return accurate_count,totol_count

class TokenPosTagPair(object):
    def __init__(self, token, pos_tag):
        self.token = token
        self.pos_tag = pos_tag
    def get_tuples():
        return self.token, self.pos_tag

class Cell(object):
    def __init__(self, score, back_pos):
        self.score = .0
        self.back_pos = ''
    def get_tuples():
        return self.score, self.back_pos
