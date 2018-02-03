import json
from nltk.tokenize import RegexpTokenizer
import numpy as np
import random

import summarization_args


def read_docs(args, type):
    filename = type + '_model.json' if args.full_test else "small_" + type + '_model.json'
    filename = '../data/'+ args.source + '_' + str(args.vocab_size) + '_' + filename

    with open(filename, 'rb') as data_file:
        data = json.load(data_file)

    if type == 'test':
        return data['x']
    elif type == 'dev':
        return data['x'], data['y'], data['e'], data['clean_y'], data['raw_x'], data['sha']
    else:
        return data['x'], data['y'], data['e'], data['clean_y'], data['sha']


def create_vocab(args):
    vocab_map = dict()
    vocab_ls = []

    ifp = open('../data/' + str(args.source) + '_vocab_' + str(args.vocab_size) + '.txt', 'r')

    for line in ifp:
        w = line.rstrip()
        vocab_map[w] = len(vocab_map)
        vocab_ls.append(w)

    ifp.close()

    return vocab_map, vocab_ls


def create_stopwords(args, vocab_map, lst_words):
    ifp = open(args.stopwords, 'r')
    stopwords = set()

    # Add stopwords
    for line in ifp:
        w = line.rstrip()

        if w in vocab_map:
            stopwords.add(vocab_map[w])

    ifp.close()

    tokenizer = RegexpTokenizer(r'\w+')

    # add punctuation
    for i in xrange(len(lst_words)):
        w = lst_words[i]

        if len(tokenizer.tokenize(w)) == 0:
            stopwords.add(i)

    record_stopwords(stopwords, lst_words)

    return stopwords


def create_batches(args, n_classes, x, y, e, cy, sha, rx,  batch_size, padding_id, stopwords, sort=True, model_type=''):
    batches_x, batches_y, batches_e, batches_bm, batches_sha, batches_rx  = [], [], [], [], [], []
    N = len(x)
    M = (N - 1) / batch_size + 1
    num_batches = 0
    num_files = 0

    if args.sanity_check:
        sort= False
        M = 1

    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
        e = [e[i] for i in perm]
        cy = [cy[i] for i in perm]
        sha = [sha[i] for i in perm]

    for i in xrange(M):
        bx, by, be, bm = create_one_batch(
            args,
            n_classes,
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size],
            e[i * batch_size:(i + 1) * batch_size],
            cy[i * batch_size:(i + 1) * batch_size],
            padding_id,
            batch_size,
            stopwords
        )
        bsh = sha[i * batch_size:(i + 1) * batch_size]
        if rx is not None:
            brx = rx[i * batch_size:(i + 1) * batch_size]
            batches_rx.append(brx)

        batches_x.append(bx)
        batches_y.append(by)
        batches_e.append(be)
        batches_bm.append(bm)
        batches_sha.append(bsh)

        num_batches += 1

        if num_batches >= args.online_batch_size or i == M - 1:

            fname = args.batch_dir + args.source + model_type
            print 'Creating file #', str(num_files + 1)

            if model_type == 'train':
                data = [
                    batches_x,
                    batches_y,
                    batches_e,
                    batches_bm,
                    batches_sha
                ]
            else:
                data = [
                    batches_x,
                    batches_y,
                    batches_e,
                    batches_bm,
                    batches_sha,
                    batches_rx
                ]
            with open(fname + str(num_files), 'w+') as ofp:
                np.save(ofp, data)

            batches_x, batches_y, batches_e, batches_bm, batches_sha, batches_rx = [], [], [], [], [], []
            num_batches = 0
            num_files += 1

    print "Num Files :", num_files


def create_one_batch(args, n_classes, lstx, lsty, lste, lstcy, padding_id, b_len, stopwords):
    """
    Parameters
    ----------
    lstx : List of 1-a documents.

    lsty : List of list of sentences
        usually a preselected limit of input Highlights

    lstve : List of all valid entity indexes to use in loss

    lste : For each hl in lsty, the correct class

    """
    max_len = args.inp_len

    assert min(len(x) for x in lstx) > 0

    # padded y
    by, unigrams, be = process_hl(args, lsty, lste, padding_id, n_classes, lstcy)

    bx = np.column_stack([np.pad(x[:max_len], (0, max_len - len(x) if len(x) <= max_len else 0), "constant",
                                 constant_values=padding_id).astype('int32') for x in lstx])

    bm = create_unigram_masks(lstx, unigrams, max_len, stopwords)

    bm = np.column_stack([m for m in bm])
    by = np.column_stack([y for y in by])

    return bx, by, be, bm


def process_hl(args, lsty, lste, padding_id, n_classes, lstcy):
    max_len_y = args.hl_len
    y_processed = [[] for i in xrange(args.n)]
    e_processed = [[] for i in xrange(args.n)]
    unigrams = []

    for i in xrange(len(lsty)):
        sample_u = set()

        for j in xrange(len(lsty[i])):
            y = lsty[i][j][:max_len_y]
            single_hl = np.pad(y, (max_len_y - len(y), 0), "constant", constant_values=padding_id).astype('int32')

            single_e_1h = np.zeros((n_classes,), dtype='int32')
            single_e_1h[lste[i][j]] = 1

            y_processed[j].append(single_hl)
            e_processed[j].append(single_e_1h)

        for clean_hl in lstcy[i]:
            trimmed_cy = clean_hl[:max_len_y]

            for token in trimmed_cy:
                sample_u.add(token)

        unigrams.append(sample_u)

    by = []
    be = []
    for i in xrange(len(y_processed)):
        by.extend(y_processed[i])
        be.extend(e_processed[i])

    return by, unigrams, be


def create_unigram_masks(lstx, unigrams, max_len, stopwords):
    masks = []

    for i in xrange(len(lstx)):
        len_x = len(lstx[i])
        m = np.zeros((max_len,), dtype='int32')

        for j in xrange(len_x - 1):
            if j >= max_len:
                break
            w1 = lstx[i][j]
            w2 = lstx[i][j+1]

            if w1 in unigrams[i] and w2 in unigrams[i]:
                if contains_single_valid_word(w1, w2, stopwords):
                    m[j] = 1

        masks.append(m)

    return masks


def contains_single_valid_word(w1, w2, stopwords):
    if w1 in stopwords and w2 in stopwords:
        return False
    return True


def process_ent(n_classes, lste):
    ret_e = []

    for e in lste:
        e_mask = np.zeros((n_classes,),dtype='int32')

        for e_idx in e:
            e_mask[e_idx] = 1

        ret_e.append(e_mask)

    return ret_e


def record_stopwords(stopwords, lst_words):
    ofp = open('../data/stopword_map.json', 'w+')
    data = dict()

    for w_idx in stopwords:
        data[w_idx] = lst_words[w_idx]

    json_d = dict()
    json_d['tokens'] = data
    json.dump(json_d, ofp)
    ofp.close()


def main(args):
    vocab_map, lst_words = create_vocab(args)
    stopwords = create_stopwords(args, vocab_map, lst_words)
    pad_id = vocab_map["<padding>"]

    del vocab_map
    del lst_words

    if args.train:
        print 'TRAIN data'
        print '  Read JSON..'

        train_x, train_y, train_e, train_clean_y, train_sha = read_docs(args, 'train')

        print '  Create batches..'

        create_batches(args, args.nclasses, train_x, train_y, train_e, train_clean_y, train_sha, None, args.batch,
                       pad_id, stopwords, sort=True, model_type='train')

        print '  Purge references..'

        del train_x
        del train_y
        del train_e
        del train_clean_y
        del train_sha

        print '  Finished Train Proc.'

    if args.dev:
        print 'DEV data'
        print '  Read JSON..'

        dev_x, dev_y, dev_e, dev_clean_y, dev_rx, dev_sha = read_docs(args, 'dev')

        print '  Create batches..'

        create_batches(args, args.nclasses, dev_x, dev_y, dev_e, dev_clean_y, dev_sha, dev_rx, args.batch, pad_id,
                       stopwords, sort=False, model_type='dev')

        print '  Purge references..'

        del dev_x
        del dev_y
        del dev_e
        del dev_clean_y
        del dev_rx
        del dev_sha

        print '  Finished Dev Proc.'


if __name__ == "__main__":
    args = summarization_args.get_args()
    main(args)