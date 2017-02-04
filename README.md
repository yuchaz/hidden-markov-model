Before you started. Make sure that you have data with you. be sure to cp `config.ini.template` to `config.ini` and fill in the parent directory the data in. For example if the `twi.train.json`,  `twi.dev.json`, `twi.test.json` and `twi.bonus.json` is in `/home/data`, then you should put the path here.

Use `python run_bigram.py` to run bigram HMM on test set.

If you want to get the visualization of confusion matrix of bigram model, be sure to install `matplotlib` and uncomment the 18, 19 line in `run_bigram.py`.

Use `python run_trigram.py` to run trigram HMM on test set.

If you want to change the input corpus, you can call these functions:
`Corpus.trainCorpus()` to get train corpus; `Corpus.devCorpus()` to get dev corpus; `Corpus.testCorpus()` to get test corpus; `Corpus.bonusCorpus` to get bonus corpus. There are two parameters can be used here, one is `ratio`, means the ratio of corpus you want to use; second is `shuffle`, means if you want to shuffle the corpus.

If you want to get mixed corpus, you can use `Corpus.combinedCorpus()` by providing `ratio` of corpus, `shuffle`, and `*tags`. tags should be either 'train', 'dev', 'test', or 'bonus'.

You also can determine whether to do OOV handling by calling `corpus.replace_oov_with_UNK()` method. There are four parameters you can input: `unk_threshold`, `unk_oov_ratio`, `trans_prob`, and  `known_unk_dict`. The first three parameters are descirbed in the `HW2.pdf` file. And the last parameter means that if you have a unk_dict in hand, you can input it inside. It is often used when we want to extend the `unk_dict` of evaluation set with the `unk_dict` of training set.

As for the `HmmModel` class, `n` means ngram HMM you want to use, it will whether be 2 or 3. `k_lan_model` and `k_emiss_model` parameters are also descirbed in `HW2.pdf`

Use `python gridsearch_bigram` to run gridsearch on bigram HMM model. you can update the parameters in `updated_params`. Avaiable keys will be `unk_threshold`, `unk_oov_ratio`, `trans_prob`, `k_lan_model` and `k_emiss_model`. And the value of that dictionary should be an list of values you wanna test.

Use `python gridsearch_trigram` to un gridsearch on trigram HMM model. But it will take forever to run...
