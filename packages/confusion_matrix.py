import numpy as np
import matplotlib.pyplot as plt

def calc_confusion_matrix(hmm, corpus):
    vocabulary_list = hmm.possible_pos_choices
    vocab_size = len(vocabulary_list)
    vocab_inverse_dict = {vocabulary_list[idx]: idx for idx in range(vocab_size)}
    all_confusion_matrix = np.zeros((vocab_size,vocab_size))

    for doc in corpus:
        confusion_matrix = doc.store_confusion_matrx(vocab_inverse_dict)
        all_confusion_matrix = np.add(all_confusion_matrix, confusion_matrix)
    return all_confusion_matrix

def plot_confusion_matrix(conf_arr, classes):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if a == 0:
                tmp_arr.append(0)
            else:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues,
                    interpolation='nearest')

    width, height = conf_arr.shape

    cb = fig.colorbar(res)
    plt.xticks(range(width), classes)
    plt.yticks(range(height), classes)
    plt.savefig('confusion_matrix.png', format='png')

def main():
    a = np.zeros((3,3))
    a[0][1] = 3
    plot_confusion_matrix(a, list('ABC'))

if __name__ == '__main__':
    main()
