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


def read_docs(args):
    data_x, data_y = [], []

    with open(args.train, 'r') as data_file:
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
    batches_x, batches_y, batches_ym = [], [], []
    N = len(x)
    M = (N - 1) / batch_size + 1
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]

    for i in xrange(M):
        bx, by, bym = create_one_batch(
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size],
            padding_id,
            batch_size
        )
        batches_x.append(bx)
        batches_y.append(by)
        batches_ym.append(bym)
    if sort:
        random.seed(5817)
        perm2 = range(M)
        random.shuffle(perm2)
        batches_x = [batches_x[i] for i in perm2]
        batches_y = [batches_y[i] for i in perm2]
        batches_ym = [batches_ym[i] for i in perm2]

    return batches_x, batches_y, batches_ym


def create_one_batch(lstx, lsty, padding_id, b_len):
    max_len = max(len(x) for x in lstx)
    max_len_y = max(len(y) for y in lsty)

    assert min(len(x) for x in lstx) > 0

    bx = np.column_stack([np.pad(x, (max_len - len(x), 0), "constant",
                                 constant_values=padding_id) for x in lstx])
    by = np.column_stack([np.pad(y, (max_len_y - len(y), 0), "constant",
                                 constant_values=padding_id) for y in lsty])

    bym = np.column_stack([np.asarray([0 if lsty[k][i] == padding_id else 1 for i in xrange(max_len_y)],dtype='int32') for k in xrange(b_len)])

    return bx, by, bym


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
            ofp = open(args.system_summ_path + 's.' + str(s_num) + '.txt', 'w+')

            for j in xrange(len(bx[z])):
                word = emb_layer.lst_words[bx[z][j][i]]

                if word == '<padding>' or word == '<unk>' or bz[z][j][i] == 0:
                    continue

                ofp.write(word + ' ')

            ofp.close()
            s_num += 1

    s_num = 0
    for batch in by:
        for i in xrange(len(batch[0])):
            ofp = open(args.model_summ_path + 's.' + str(s_num) + '.txt', 'w+')

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
    r.system_filename_pattern = 's.(\d+).txt'
    r.model_filename_pattern = 's.#ID#.txt'

    return r.convert_and_evaluate()


def total_words(z):
    return np.sum(z, axis=None)


def write_metrics(num_sum, total_w, ofp, epoch, args, overall=False):
    if overall:
        ofp.write('OVERALL STATS :\n')
        ofp.write('Average words in summary : ' + str(total_w/float(num_sum)))
        ofp.write('Rouge :\n' + str(get_rouge(args)) + '\n\n')
    else:
        ofp.write('Epoch : ' + str(epoch) + '\n')
        ofp.write('\tAvg Words : ' + str(total_w/float(num_sum)) + '\n\n')

