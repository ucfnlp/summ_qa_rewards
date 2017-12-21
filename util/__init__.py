
import sys
import gzip

import numpy as np


def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals


def get_ngram(l, n=2):
    return set(zip(*[l[i:] for i in range(n)]))


def gen_set(max_sentences, sentence_length, batch, x, z=None, n=2):
    l = []

    for j in range(batch):
        tmp = []
        for i in xrange(max_sentences * sentence_length):
            if z is None:
                tmp.append(x[j, i])
            elif z[j,i] == 0:
                tmp.append(x[j,i,0])

        l.append(set(zip(*[tmp[i:] for i in range(n)])))

    return l
