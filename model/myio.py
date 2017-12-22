import gzip
import json
import random

import numpy as np
from pyrouge import Rouge155

from nn.basic import EmbeddingLayer
from util import load_embedding_iterator


def read_rationales(path):
    data = []
    fopen = gzip.open if path.endswith(".gz") else open

    with fopen(path) as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)

    return data


def read_docs(filename):
    data_x, data_y = [], []

    with open(filename, 'r') as data_file:
        data = json.load(data_file)

    data_x = data['x']
    data_y = data['y']

    data_file.close()

    return data_x, data_y


def create_embedding_layer(path):
    embedding_layer = EmbeddingLayer(
        n_d=200,
        vocab=["<unk>", "<padding>"],
        embs=load_embedding_iterator(path),
        oov="<unk>",
        # fix_init_embs = True
        fix_init_embs=False
    )
    return embedding_layer


def create_batches(x, y, batch_size, padding_id, sort=True):
    batches_x, batches_y, batches_v = [], [], []
    N = len(x)
    M = (N - 1) / batch_size + 1
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]

    for i in xrange(M):
        bx, by, bv = create_one_batch(
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size],
            padding_id,
            batch_size
        )
        batches_x.append(bx)
        batches_y.append(by)
        batches_v.append(bv)

    if sort:
        random.seed(5817)
        perm2 = range(M)
        random.shuffle(perm2)
        batches_x = [batches_x[i] for i in perm2]
        batches_y = [batches_y[i] for i in perm2]
        batches_v = [batches_v[i] for i in perm2]

    return batches_x, batches_y, batches_v


def create_one_batch(lstx, lsty, padding_id, b_len):
    max_len = max(len(x) for x in lstx)
    max_len_y = max(len(y) for y in lsty)

    assert min(len(x) for x in lstx) > 0

    lstbv = bigram_vectorize(lstx, lsty, padding_id)

    bx = np.column_stack([np.pad(x, (max_len - len(x), 0), "constant",
                                 constant_values=padding_id) for x in lstx])
    by = np.column_stack([np.pad(y, (max_len_y - len(y), 0), "constant",
                                 constant_values=padding_id) for y in lsty])

    bv = np.column_stack([b for b in lstbv])

    return bx, by, bv


def write_train_results(bz, bx, by, emb_layer, ofp, padding_id):
    ofp.write("BULLET POINTS :\n")

    for i in xrange(4):
        ofp.write("BP # " + str(i) + "\n")
        for j in xrange(15):

            idx = i*15 + j
            if by[idx][0] != padding_id:
                ofp.write(emb_layer.lst_words[by[idx][0]] + " ")

        ofp.write("\n")

    ofp.write("SUMMARY :\n\n")

    for i in xrange(10):
        did_write = False

        for j in xrange(30):
            idx = i * 30 + j

            if bz[idx][0] > 0 and bx[idx][0] != padding_id:
                did_write = True
                ofp.write(emb_layer.lst_words[bx[idx][0]] + " ")

        if did_write:
            ofp.write("\n")

    ofp.write("\n")


def write_summ_for_rouge(args, bz, bx, by, emb_layer):
    s_num = 1
    for z in xrange(len(bx)):
        for i in xrange(len(bx[z][0])):
            ofp = open(args.system_summ_path + str(args.sparsity) + 's.' + str(s_num) + '.txt', 'w+')

            for j in xrange(len(bx[z])):
                word = emb_layer.lst_words[bx[z][j][i]]

                if word == '<padding>' or word == '<unk>' or bz[z][j][i] == 0:
                    continue

                ofp.write(word + ' ')

            ofp.close()
            s_num += 1

    s_num = 1
    for batch in by:
        for i in xrange(len(batch[0])):
            ofp = open(args.model_summ_path + str(args.sparsity) + 's.' + str(s_num) + '.txt', 'w+')

            for j in xrange(len(batch)):
                word = emb_layer.lst_words[batch[j][i]]

                if word == '<padding>' or word == '<unk>':
                    continue

                ofp.write(word + ' ')

            ofp.close()
            s_num += 1


def get_rouge(args):
    r = Rouge155()
    r.system_dir = args.system_summ_path
    r.model_dir = args.model_summ_path
    r.system_filename_pattern = str(args.sparsity) + 's.(\d+).txt'
    r.model_filename_pattern = str(args.sparsity) + 's.#ID#.txt'

    return r.convert_and_evaluate()


def get_ngram(l, n=2):
    return set(zip(*[l[i:] for i in range(n)]))


def bigram_vectorize(lstx, lsty, padding_id):
    bin_vectors = []

    for i in xrange(len(lsty)):
        target_ngrams = get_ngram(lsty[i])

        bin_vec = np.zeros_like(lstx[i])

        for j in xrange(bin_vec.shape[0] - 1):
            w_1 = lstx[i][j]
            w_2 = lstx[i][j + 1]
            if w_1 == padding_id:
                continue

            bigram = (w_1, w_2)

            if bigram in target_ngrams:
                bin_vec[j] = 1

        bin_vectors.append(bin_vec)

    return bin_vectors

def total_words(z):
    return np.sum(z, axis=None)


def write_metrics(num_sum, total_w, ofp, epoch, args):
    ofp.write('Epoch : ' + str(epoch) + '\n')
    ofp.write('Average words in summary : ' + str(total_w/float(num_sum)) + '\n')
    ofp.write('Rouge :\n' + str(get_rouge(args)) + '\n\n')