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
        for idx in range(self.doc_length):
            yield self.token_pos_tag_pairs[idx]

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
        return self.token_pos_tag_pairs[self.end_index]

    def run_viterbi(self,hmm):
        if hmm.ngram == 2:
            self.run_bigram_viterbi(hmm)
        elif hmm.ngram == 3:
            self.hidden_cells.append({})
            self.run_trigram_viterbi(hmm)
    def run_bigram_viterbi(self, hmm):
        if self.end_index <= self.doc_length:
            pos_choices = hmm.possible_pos_choices \
                if self.end_index != self.doc_length else [STOP_SYMBOL]

            for target_pos_tag in pos_choices:
                self.calc_pi_score(hmm,target_pos_tag)
            self.push_back()
            self.run_bigram_viterbi(hmm)
        else:
            self.back_propagation()

    def calc_pi_score(self, hmm, current_pos):
        current_idx = self.end_index+1
        current_token = None if self.end_index == self.doc_length else self.get_last_term().token
        if self.end_index == 0:
            max_pi_score = hmm.e_score(TokenPosTagPair(current_token,current_pos)) + \
                           hmm.q_score((START_SYMBOL,current_pos))
            predicted_prev_pos = START_SYMBOL
        elif self.end_index == self.doc_length:
            predicted_prev_pos, max_pi_score = \
                max(((pos_tag, hmm.q_score((pos_tag,current_pos)) + \
                               self.hidden_cells[current_idx-1][pos_tag].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        else:
            predicted_prev_pos, max_pi_score = \
                max(((pos_tag, hmm.e_score(TokenPosTagPair(current_token,current_pos)) + \
                               hmm.q_score((pos_tag,current_pos)) + \
                               self.hidden_cells[current_idx-1][pos_tag].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        self.store_cells(current_idx, current_pos, max_pi_score, predicted_prev_pos)

    def run_trigram_viterbi(self, hmm):
        if self.end_index <= self.doc_length+1:
            current_pos_choices = hmm.possible_pos_choices \
                if self.end_index < self.doc_length else [STOP_SYMBOL]
            prev_pos_choices = [START_SYMBOL] if self.end_index == 0 \
                               else [STOP_SYMBOL] if self.end_index == self.doc_length+1 \
                               else hmm.possible_pos_choices

            for current_pos in current_pos_choices:
                for prev_pos in prev_pos_choices:
                    self.calc_trigram_pi_score(hmm,(current_pos,prev_pos))
            self.push_back()
            self.run_trigram_viterbi(hmm)
        else:
            self.trigram_back_propagation()

    def calc_trigram_pi_score(self,hmm,current_pos_tuples):
        current_idx = self.end_index+1
        current_pos, last_pos = current_pos_tuples
        current_token = None if self.end_index >= self.doc_length else self.get_last_term().token

        if self.end_index <= 1:
            max_pi_score = hmm.e_score(TokenPosTagPair(current_token,current_pos)) + \
                           hmm.q_score((START_SYMBOL,last_pos,current_pos))
            predicted_prev_2_pos = START_SYMBOL

        elif self.end_index == self.doc_length:
            predicted_prev_2_pos, max_pi_score = \
                max(((pos_tag, hmm.q_score((pos_tag,last_pos,current_pos)) + \
                               self.hidden_cells[current_idx-1][(last_pos,pos_tag)].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        elif self.end_index == self.doc_length+1:
            predicted_prev_2_pos, max_pi_score = \
                max(((pos_tag, self.hidden_cells[current_idx-1][(last_pos,pos_tag)].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        else:
            predicted_prev_2_pos, max_pi_score = \
                max(((pos_tag, hmm.e_score(TokenPosTagPair(current_token,current_pos)) + \
                               hmm.q_score((pos_tag,last_pos,current_pos)) + \
                               self.hidden_cells[current_idx-1][(last_pos,pos_tag)].score)
                               for pos_tag in hmm.possible_pos_choices),
                               key=lambda p:p[1])
        predicted_prev_pos = (last_pos,predicted_prev_2_pos)
        current_pos = current_pos_tuples if self.end_index != self.doc_length+1 else STOP_SYMBOL
        self.store_cells(current_idx, current_pos, max_pi_score, predicted_prev_pos)


    def store_cells(self,idx,pos_tag,score,argmax_pos_tag):
        self.hidden_cells[idx].update({pos_tag: Cell(score, argmax_pos_tag)})

    def back_propagation(self):
        if self.end_index > self.doc_length:
            self.pop_back()
            self.back_propagation()

        elif self.end_index > 0:
            current_pos = STOP_SYMBOL \
                            if  self.end_index == self.doc_length \
                            else self.predicted_pos_tags[0]
            score, back_pos = self.hidden_cells[self.end_index+1][current_pos].get_tuples()
            self.predicted_pos_tags.appendleft(back_pos)
            self.pop_back()
            self.back_propagation()
        else:
            return

    def trigram_back_propagation(self):
        if self.end_index > self.doc_length+1:
            self.pop_back()
            self.trigram_back_propagation()

        elif self.end_index > 1:
            current_pos = STOP_SYMBOL \
                            if  self.end_index == self.doc_length+1 \
                            else (self.predicted_pos_tags[1],self.predicted_pos_tags[0])
            # import pdb; pdb.set_trace()
            score, back_pos_tuples = self.hidden_cells[self.end_index+1][current_pos].get_tuples()
            back_pos, prev_pos = back_pos_tuples
            if current_pos == STOP_SYMBOL:
                self.predicted_pos_tags.appendleft(back_pos)
                self.predicted_pos_tags.appendleft(prev_pos)
            else:
                self.predicted_pos_tags.appendleft(prev_pos)
            self.pop_back()
            self.trigram_back_propagation()
        else:
            self.predicted_pos_tags.pop()
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

    def replaced_with(self,to_unk_dict):
        for tokn_pos_pair in self:
            tokn_pos_pair.replaced_with(to_unk_dict)


class TokenPosTagPair(object):
    def __init__(self, token, pos_tag):
        self.token = token
        self.pos_tag = pos_tag
    def get_tuples(self):
        return self.token, self.pos_tag
    def replaced_with(self,to_unk_dict):
        unk_type = to_unk_dict.get(self.token)
        if unk_type != None:
            self.token = unk_type

class Cell(object):
    def __init__(self, score, back_pos):
        self.score = score
        self.back_pos = back_pos
    def __str__(self):
        return str(self.get_tuples())
    def get_tuples(self):
        return self.score, self.back_pos
