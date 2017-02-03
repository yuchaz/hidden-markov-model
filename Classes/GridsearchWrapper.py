from Classes.HmmModel import HmmModel
from Classes.Corpus import Corpus

class GridsearchWrapper(object):
    def __init__(self,train_list,eval_list,train_corpus_param,eval_corpus_param,ngram,default_params,params={}):
        self.best_params_ = {}
        self.best_score_ = .0
        self.params_scores_list = []
        self.grid_search(train_list,eval_list,train_corpus_param,eval_corpus_param,ngram,default_params,params)

    def grid_search(self,train_list,eval_list,train_list_param,eval_corpus_param,ngram,default_params,params={}):
        default_unk_threshold = default_params.get('unk_threshold')
        default_unk_oov_ratio = default_params.get('unk_oov_ratio')
        default_trans_prob = default_params.get('trans_prob')
        default_k_lan_model = default_params.get('k_lan_model')
        default_k_emiss_model = default_params.get('k_emiss_model')
        if len(params) == 0:
            accu = run_experiment(train_list,eval_list,train_list_param,
                                  eval_corpus_param,ngram,default_unk_threshold,
                                  default_unk_oov_ratio,default_trans_prob,
                                  default_k_lan_model,default_k_emiss_model)
            self.best_params_ = params
            self.best_score_ = accu
        else:
            for update_unk_threshold in params.get('unk_threshold',[]):
                accu = run_experiment(train_list,eval_list,train_list_param,
                                      eval_corpus_param,ngram,update_unk_threshold,
                                      default_unk_oov_ratio,default_trans_prob,
                                      default_k_lan_model,default_k_emiss_model)
                self.params_scores_list.append((accu,{'unk_threshold':update_unk_threshold}))
            for update_unk_oov_ratio in params.get('unk_oov_ratio',[]):
                accu = run_experiment(train_list,eval_list,train_list_param,
                                      eval_corpus_param,ngram,default_unk_threshold,
                                      update_unk_oov_ratio,default_trans_prob,
                                      default_k_lan_model,default_k_emiss_model)
                self.params_scores_list.append((accu,{'unk_oov_ratio':update_unk_oov_ratio}))

            for update_trans_prob in params.get('trans_prob',[]):
                accu = run_experiment(train_list,eval_list,train_list_param,
                                      eval_corpus_param,ngram,default_unk_threshold,
                                      default_unk_oov_ratio,update_trans_prob,
                                      default_k_lan_model,default_k_emiss_model)
                self.params_scores_list.append((accu,{'trans_prob':update_trans_prob}))

            for update_k_lan_model in params.get('k_lan_model',[]):
                accu = run_experiment(train_list,eval_list,train_list_param,
                                      eval_corpus_param,ngram,default_unk_threshold,
                                      default_unk_oov_ratio,default_trans_prob,
                                      update_k_lan_model,default_k_emiss_model)
                self.params_scores_list.append((accu,{'k_lan_model':update_k_lan_model}))

            for update_k_emiss_model in params.get('k_emiss_model',[]):
                accu = run_experiment(train_list,eval_list,train_list_param,
                                      eval_corpus_param,ngram,default_unk_threshold,
                                      default_unk_oov_ratio,default_trans_prob,
                                      default_k_lan_model,update_k_emiss_model)
                self.params_scores_list.append((accu,{'k_emiss_model':update_k_emiss_model}))

            self.best_score_, self.best_params_ = max((parscore for parscore in self.params_scores_list), lambda ps:ps[0])

def run_experiment(train_list,eval_list,train_corpus_param,eval_corpus_param,ngram,unk_threshold,unk_oov_ratio,trans_prob,k_lan_model,k_emiss_model):
    train_corpus = Corpus.combinedCorpus(train_corpus_param['ratio'], train_corpus_param['shuffle'], *train_list)
    train_corpus.replace_oov_with_UNK(unk_threshold,unk_oov_ratio,trans_prob)
    eval_corpus = Corpus.combinedCorpus(eval_corpus_param['ratio'], eval_corpus_param['shuffle'], *eval_list)
    eval_corpus.replace_oov_with_UNK(unk_threshold,unk_oov_ratio,trans_prob,train_corpus.to_unk_dict)
    hmm = HmmModel(train_corpus,ngram,k_lan_model,k_emiss_model)
    return get_accuracy(hmm, eval_corpus)

def get_accuracy(hmm, corpus):
    total_accu_count = 0
    total_count = 0
    for doc in corpus:
        doc.run_viterbi(hmm)
        accu_count, ttl_count = doc.evaluate()
        total_accu_count += accu_count
        total_count += ttl_count
    return float(total_accu_count)/total_count
