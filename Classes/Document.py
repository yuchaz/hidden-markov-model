from collections import deque
STOP_SYMBOL = '<STOP>'
START_SYMBOL = '<START>'

class Document(object):
    def __init__(self, sentence):
        self.token_pos_tag_pairs = [TokenPosTagPair(word[0], word[1]) for word in sentence]
        self.end_index = 0
        self.predicted_pos_tags = deque([])
        self.doc_length = len(sentence)
        self.hidden_cells = [{} for i in range(self.doc_length+2)]
        self.hidden_cells[0] = {START_SYMBOL:Cell(1,'')}

    def __iter__(self):
        for token_pos_tag_pair in (self.token_pos_tag_pair[idx]
                                   for idx in range(self.end_index)
                                   if idx < self.end_index):
            yield tokens_tag_pair

    def get_pos_tag_corpus(self):
        return [tokn_pos_pair.pos_tag for tokn_pos_pair in self.token_pos_tag_pairs]

    def get_token_corpus(self):
        return [tokn_pos_pair.token for tokn_pos_pair in self.token_pos_tag_pairs]

    def get_token_pos_tag_pair(self):
        return [tokn_pos_pair.get_tuples() for tokn_pos_pair in self.token_pos_tag_pairs]

    def push_back(self):
        self.end_index += 1

    def pop_back(self):
        self.end_index -= 1

    def get_last_term(self):
        return self.token_pos_tag_pairs[self.end_index-1]

    def store_cells(self,idx,pos_tag,score,argmax_pos_tag):
        self.hidden_cells[idx].update({pos_tag: Cell(score, argmax_pos_tag)})

    def calc_pi_score(self, hmm, current_pos):
        current_token = self.get_last_term().token
        current_idx = self.end_index+1
        if self.end_index == 0:
            max_pi_score = hmm.e_score(TokenPosTagPair(current_token,current_pos)) + \
                           hmm.q_score((START_SYMBOL,current_pos))
            predicted_prev_pos = START_SYMBOL
        elif self.end_index == self.doc_length+1:
            predicted_prev_pos, max_pi_score = \
                max(((pos_tag, hmm.q_score((pos_tag,current_pos)) + \
                               self.hidden_cells[current_idx-1][pos_tag].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        else:
            print current_token
            predicted_prev_pos, max_pi_score = \
                max(((pos_tag, hmm.e_score(TokenPosTagPair(current_token,current_pos)) + \
                               hmm.q_score((pos_tag,current_pos)) + \
                               self.hidden_cells[current_idx-1][pos_tag].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        self.store_cells(current_idx, current_pos, max_pi_score, predicted_prev_pos)

    def run_viterbi(self, hmm):
        if self.end_index <= self.doc_length+1:
            pos_choices = hmm.possible_pos_choices \
                if self.end_index != self.doc_length+1 else [STOP_SYMBOL]

            for target_pos_tag in pos_choices:
                self.calc_pi_score(hmm,target_pos_tag)
            self.push_back()
            self.run_viterbi(hmm)
        else:
            self.back_propagation()

    def back_propagation(self):
        if self.end_index > self.doc_length:
            self.pop_back()
            self.back_propagation()

        elif self.end_index > 0:
            current_pos = STOP_SYMBOL \
                            if  self.end_index == self.doc_length \
                            else self.predicted_pos_tags[0]
            score, back_pos = self.hidden_cells[self.end_index+1][current_pos].get_tuples()
            import pdb; pdb.set_trace()
            self.predicted_pos_tags.appendleft(back_pos)
            self.pop_back()
            self.back_propagation()
        else:
            import pdb; pdb.set_trace()
            return

    def get_score_and_predicted_pos_list(self):
        return self.hidden_cells[-1][STOP_SYMBOL].score, elf.predicted_pos_tags

    def evaluate(self):
        if len(self.predicted_pos_tags) != self.doc_length:
            raise ValueError(
                "You should not called this before running Viterbi algorithms"
            )

        accurate_count = sum(1 for idx in range(len(self.token_pos_tag_pairs))
                   if self.token_pos_tag_pairs[idx].pos_tag == self.predicted_pos_tags[idx])
        totol_count = len(self.token_pos_tag_pairs)
        return accurate_count,totol_count

    def print_ith_layer_hidden_cell(self,i):
        for k,v in self.hidden_cells[i].items(): print k,v

class TokenPosTagPair(object):
    def __init__(self, token, pos_tag):
        self.token = token
        self.pos_tag = pos_tag
    def get_tuples(self):
        return self.token, self.pos_tag

class Cell(object):
    def __init__(self, score, back_pos):
        self.score = score
        self.back_pos = back_pos
    def __str__(self):
        return str(self.get_tuples())
    def get_tuples(self):
        return self.score, self.back_pos
