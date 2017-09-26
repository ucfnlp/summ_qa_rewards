
import gzip
import random
import json

import theano
import numpy as np

from nn.basic import EmbeddingLayer
from util import say, load_embedding_iterator


def read_rationales(path):
    data = [ ]
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
            n_d = 200,
            vocab = [ "<unk>", "<padding>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            #fix_init_embs = True
            fix_init_embs = False
        )
    return embedding_layer


def create_batches(x, y, batch_size, padding_id, sort=True):
    batches_x, batches_y = [ ], [ ]
    N = len(x)
    M = (N-1)/batch_size + 1
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [ x[i] for i in perm ]
        y = [ y[i] for i in perm ]
    for i in xrange(M):
        bx, by = create_one_batch(
                    x[i*batch_size:(i+1)*batch_size],
                    y[i*batch_size:(i+1)*batch_size],
                    padding_id
                )
        batches_x.append(bx)
        batches_y.append(by)
    if sort:
        random.seed(5817)
        perm2 = range(M)
        random.shuffle(perm2)
        batches_x = [ batches_x[i] for i in perm2 ]
        batches_y = [ batches_y[i] for i in perm2 ]
    return batches_x, batches_y


def create_one_batch(lstx, lsty, padding_id):
    max_len = max(len(x) for x in lstx)
    assert min(len(x) for x in lstx) > 0
    bx = np.column_stack([ np.pad(x, (max_len-len(x),0), "constant",
                        constant_values=padding_id) for x in lstx ])
    by = np.vstack(lsty).astype(theano.config.floatX)
    return bx, by
