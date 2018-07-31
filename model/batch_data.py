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
        return data['x'], data['y'], data['e'], data['sha'], data['clean_y'], data['raw_x'], data['mask'], data['chunk'], data['scut']
    elif type == 'dev':
        return data['x'], data['y'], data['e'], data['clean_y'], data['raw_x'], data['sha'], data['mask'], data['chunk'], data['scut']
    else:
        return data['x'], data['y'], data['e'], data['clean_y'], data['sha'], data['mask'], data['chunk'], data['scut']


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
    punctuation = set()

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
            punctuation.add(i)

    record_stopwords(stopwords, punctuation, lst_words)

    return stopwords, punctuation


def create_batches_test(args, x, y, cy, pt, sha, rx, batch_size, padding_id, padding_id_pt, stopwords):
    batches_x, batches_bm, batches_sha, batches_rx, batches_pt = [], [], [], [], []
    N = len(x)
    M = (N - 1) / batch_size + 1
    num_batches = 0
    num_files = 0

    for i in xrange(M):
        bx, bm, bpt = create_one_batch_test(
            args,
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size],
            cy[i * batch_size:(i + 1) * batch_size],
            pt[i * batch_size:(i + 1) * batch_size],
            padding_id,
            padding_id_pt,
            batch_size,
            stopwords
        )
        bsh = sha[i * batch_size:(i + 1) * batch_size]

        brx_ = rx[i * batch_size:(i + 1) * batch_size]
        brx = []

        for j in xrange(len(brx_)):
            brx.append([w for sent in brx_[j] for w in sent])

        batches_rx.append(brx)
        batches_x.append(bx)
        batches_bm.append(bm)
        batches_pt.append(bpt)
        batches_sha.append(bsh)

        num_batches += 1

        if num_batches >= args.online_batch_size or i == M - 1:

            fname = args.batch_dir + args.source + 'test'
            print 'Creating file #', str(num_files + 1)

            data = [
                batches_x,
                batches_bm,
                batches_pt,
                batches_sha,
                batches_rx
            ]
            with open(fname + str(num_files), 'w+') as ofp:
                np.save(ofp, data)

            batches_x, batches_bm, batches_sha, batches_rx, batches_pt = [], [], [], [], []
            num_batches = 0
            num_files += 1

    print "Num Files :", num_files


def create_batches(args, n_classes, x, y, e, sc, cy, m, sha, ch, rx, batch_size, padding_id, stopwords, sort=True, model_type=''):
    batches_x, batches_y, batches_e, batches_bm, batches_sha, batches_rx, batches_lm, batches_ch_f, batches_ch_sz, batches_sc = [], [], [], [], [], [], [], [], [], []

    N = len(x)
    M = (N - 1) / batch_size + 1
    num_batches = 0
    num_files = 0

    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
        e = [e[i] for i in perm]
        cy = [cy[i] for i in perm]
        ch = [ch[i] for i in perm]
        m = [m[i] for i in perm]
        sha = [sha[i] for i in perm]
        sc = [sc[i] for i in perm]

    for i in xrange(M):
        # batched: X, Y, entities, pre-training masking, loss masking (if < n queries)
        # chunks, chunk sizes
        bx, by, be, bm, blm, bch, bsz, bsc = create_one_batch(
            args,
            n_classes,
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size] if e is not None else None,
            e[i * batch_size:(i + 1) * batch_size] if e is not None else None,
            sc[i * batch_size:(i + 1) * batch_size] if sc is not None else None,
            cy[i * batch_size:(i + 1) * batch_size] if e is not None else None,
            m[i * batch_size:(i + 1) * batch_size],
            ch[i * batch_size:(i + 1) * batch_size],
            padding_id,
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
        batches_lm.append(blm)
        batches_sha.append(bsh)
        batches_ch_f.append(bch)
        batches_ch_sz.append(bsz)
        batches_sc.append(bsc)

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
                    batches_lm,
                    batches_sha,
                    batches_ch_f,
                    batches_ch_sz,
                    batches_sc
                ]
            elif model_type == 'dev':

                data = [
                    batches_x,
                    batches_y,
                    batches_e,
                    batches_bm,
                    batches_lm,
                    batches_sha,
                    batches_rx,
                    batches_ch_f,
                    batches_ch_sz,
                    batches_sc

                ]
            else:
                data = [
                    batches_x,
                    batches_y,
                    batches_e,
                    batches_bm,
                    batches_sha,
                    batches_rx,
                    batches_ch_f,
                    batches_ch_sz,
                    batches_sc
                ]
            with open(fname + str(num_files), 'w+') as ofp:
                np.save(ofp, data)

                batches_x, batches_y, batches_e, batches_bm, batches_sha, batches_rx, batches_lm, batches_ch_f, batches_ch_sz, batches_sc = [], [], [], [], [], [], [], [], [], []
            num_batches = 0
            num_files += 1

    print "Num Files :", num_files


def create_one_batch(args, n_classes, lstx, lsty, lste, lstsc, lstcy, lstbm, lstch, padding_id, stopwords, create_mask=False):
    max_len = args.inp_len

    assert min(len(x) for x in lstx) > 0

    # padded y
    if lste is not None:
        by, unigrams, be, blm = process_hl(args, lsty, lste, padding_id, n_classes, lstcy)
        by = np.column_stack([y for y in by])
    else:
        unigrams = None

    bch, bsz = create_chunk_mask(lstch, max_len, args.word_level_c)

    bx = np.column_stack([np.pad(x[:max_len], (0, max_len - len(x) if len(x) <= max_len else 0), "constant",
                                 constant_values=padding_id).astype('int32') for x in lstx])

    if create_mask and unigrams is not None:
        bm = create_unigram_masks(lstx, unigrams, max_len, stopwords, args)
        bm = np.column_stack([m for m in bm])
    else:
        bm = np.column_stack([np.pad(x[:max_len], (0, max_len - len(x) if len(x) <= max_len else 0), "constant",
                                 constant_values=0).astype('int32') for x in lstbm])

    bsc = sentence_indexing(lstsc, max_len)

    if lste is not None:
        return bx, by, be, bm, blm, bch, bsz, bsc
    else:
        return bx, [], [], bm, None, [], bch, bsz, bsc


def create_one_batch_test(args, lstx_, lsty, lstcy, lstpt, padding_id, padding_id_pt, b_len, stopwords):
    max_len = args.inp_len
    lstx = []
    for i in xrange(len(lstx_)):
        lstx.append([w for sent in lstx_[i] for w in sent])

    assert min(len(x) for x in lstx) > 0

    unigrams = process_hl_test(args, lsty, lstcy)

    bx = np.column_stack([np.pad(x[:max_len], (0, max_len - len(x) if len(x) <= max_len else 0), "constant",
                                 constant_values=padding_id).astype('int32') for x in lstx])

    bm = create_unigram_masks(lstx, unigrams, max_len, stopwords, args)

    bm = np.column_stack([m for m in bm])
    bpt = stack_pt(args, lstpt, padding_id_pt)

    return bx, bm, bpt


def stack_pt(args, lspt, padding_id_pt):
    num_samples = len(lspt)
    bmpt = []
    for i in xrange(num_samples):
        for j in xrange(args.inp_len):
            if j < len(lspt[i]):
                x = len(lspt[i][j])

                bmpt.append(np.pad(lspt[i][j][-args.pt_len:], (args.pt_len - x if x <= args.pt_len else 0, 0), "constant",
                                   constant_values=padding_id_pt).astype('int32'))
            else:
                x = 0
                bmpt.append(np.pad([], (args.pt_len - x if x <= args.pt_len else 0, 0), "constant",
                                   constant_values=padding_id_pt).astype('int32'))

            assert len(bmpt[-1]) == args.pt_len

    return np.column_stack([m for m in bmpt])


def process_hl(args, lsty, lste, padding_id, n_classes, lstcy):
    max_len_y = args.hl_len

    y_processed = [[] for _ in xrange(args.n)]
    e_processed = [[] for _ in xrange(args.n)]

    loss_mask = np.ones((len(lsty), args.n), dtype='int32')
    unigrams = []

    for i in xrange(len(lsty)):
        sample_u = set()

        for j in xrange(len(lsty[i])):
            y = lsty[i][j][:max_len_y]
            single_hl = np.pad(y, (max_len_y - len(y), 0), "constant", constant_values=padding_id).astype('int32')

            if n_classes > 0:
                single_e_1h = np.zeros((n_classes,), dtype='int32')
                single_e_1h[lste[i][j]] = 1
            else:
                single_e_1h = lste[i][j]

            y_processed[j].append(single_hl)
            e_processed[j].append(single_e_1h)

        # For the case of not having padded y
        if not args.pad_repeat and len(lsty[i]) < args.n:
            for j in range(len(lsty[i]), args.n):
                y_processed[j].append(np.full((max_len_y,), fill_value=padding_id).astype('int32'))

                if n_classes > 0:
                    single_e_1h = np.zeros((n_classes,), dtype='int32')
                    single_e_1h[0] = 1
                else:
                    single_e_1h = -1

                e_processed[j].append(single_e_1h)
                loss_mask[i,j] = 0

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

    return by, unigrams, be, loss_mask


def pad_sentences(args, padding_id, lstx):
    s_len = args.inp_len_sent
    lstx_batch_flat = []

    for sample in lstx:
        flattened_sample = []

        for sentence in sample:
            cur_s_len = len(sentence)

            if cur_s_len >= s_len:
                flattened_sample.extend(sentence[:s_len])
            else:
                padding = [padding_id] * (s_len - cur_s_len)
                flattened_sample.extend(sentence)
                flattened_sample.extend(padding)
                assert len(padding) + cur_s_len == s_len

        lstx_batch_flat.append(flattened_sample)

    return lstx_batch_flat


def stack_p_vec(max_x, sent_bounds, x, padding_id):
    position_idx_batch = []

    for a_idx in xrange(len(x)):
        position_idx_x = []
        total_x = 0
        cur_encoding = 0

        for sent_len in sent_bounds[a_idx]:
            cur_s = [cur_encoding] * sent_len

            position_idx_x.extend(cur_s)

            total_x += sent_len
            cur_encoding += 1

            if total_x > max_x:
                position_idx_x = position_idx_x[:max_x]
                break

        if len(position_idx_x) < max_x:
            position_idx_x.extend([padding_id] * (max_x - len(position_idx_x)))

        position_idx_batch.append(position_idx_x)

    return np.column_stack([j for j in position_idx_batch]).astype('int32')


def create_chunk_mask(lstch, max_len, word_level=False):
    fw_mask_ls = []
    mask_chunk_sizes = []

    for article in lstch:
        num_w = 0
        fw_mask = []
        mask_csz = []

        for c in article:
            if c == 0:
                continue

            if num_w + c <= max_len:
                fw_mask.extend([0]*(c-1))
                fw_mask.append(1)
                mask_csz.append(c)

                num_w += c
            else:
                if num_w == max_len:
                    break
                elif num_w < max_len:
                    fw_mask.extend([0] * (max_len - (num_w + 1)))
                    fw_mask.append(1)

                    mask_csz.append(max_len - num_w)
                    break

        if len(fw_mask) < max_len:
            left = max_len - len(fw_mask)
            fw_mask.extend([0] * (max_len - (len(fw_mask) + 1)))
            fw_mask.append(1)

            mask_csz[-1] = mask_csz[-1] + left

            print 'trunc inp', sum(article)

        if word_level:
            mask_csz = [1] * len(mask_csz)

        mask_csz.extend([0]*(max_len - len(mask_csz)))
        fw_mask_ls.append(fw_mask)
        mask_chunk_sizes.append(mask_csz)

    fw_mask_ls = np.column_stack([x for x in fw_mask_ls]).astype('int32')
    mask_chunk_sizes = np.column_stack([x for x in mask_chunk_sizes]).astype('int32')

    return fw_mask_ls, mask_chunk_sizes


def sentence_indexing(lstsc, max_len):
    indexed_x = []

    for x in lstsc:
        single_doc = []
        total_w = 0

        for i in xrange(len(x)):
            single_doc.extend([i+1]*x[i])
            total_w +=x[i]

            if total_w >= max_len:
                break

        indexed_x.append(single_doc[:max_len])

    return np.column_stack([np.pad(x[:max_len], (0, max_len - len(x) if len(x) <= max_len else 0), "constant",
                                   constant_values=0).astype('int32') for x in indexed_x])


def process_hl_test(args, lsty, lstcy):
    max_len_y = args.hl_len
    unigrams = []

    for i in xrange(len(lsty)):
        sample_u = set()

        for clean_hl in lstcy[i]:
            trimmed_cy = clean_hl[:max_len_y]

            for token in trimmed_cy:
                sample_u.add(token)

        unigrams.append(sample_u)

    return unigrams


def create_unigram_masks(lstx, unigrams, max_len, stopwords, args):
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
    sw = stopwords[0]
    punct = stopwords[1]

    if w1 in punct or w2 in punct:
        return False

    if w1 in sw and w2 in sw:
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


def record_stopwords(stopwords, punctuation, lst_words):
    ofp = open('../data/stopword_map.json', 'w+')
    data = dict()

    for w_idx in stopwords:
        data[w_idx] = lst_words[w_idx]

    json_d = dict()
    json_d['stopwords'] = data
    data = dict()

    for w_idx in punctuation:
        data[w_idx] = lst_words[w_idx]

    json_d['punctuation'] = data
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
        # data['x'], data['y'], data['e'], data['clean_y'], data['sha'], data['mask'], data['chunk']
        train_x, train_y, train_e, train_clean_y, train_sha, train_bm, train_ch, train_sc = read_docs(args, 'train')

        print '  Create batches..'

        create_batches(args, args.nclasses, train_x, train_y, train_e, train_sc,
                       train_clean_y, train_bm, train_sha, train_ch, None, args.batch,
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

        dev_x, dev_y, dev_e, dev_clean_y, dev_rx, dev_sha, dev_bm, dev_ch, dev_sc = read_docs(args, 'dev')

        print '  Create batches..'

        create_batches(args, args.nclasses, dev_x, dev_y, dev_e, dev_sc, dev_clean_y,
                       dev_bm, dev_sha, dev_ch, dev_rx, args.batch,pad_id,
                       stopwords, sort=False, model_type='dev')

        print '  Purge references..'

        del dev_x
        del dev_y
        del dev_e
        del dev_clean_y
        del dev_rx
        del dev_sha

        print '  Finished Dev Proc.'

    if args.test:
        print 'TEST data'
        print '  Read JSON..'

        test_x, test_y, test_e, test_sha, test_clean_y, test_rx, test_bm, test_ch, test_sc = read_docs(args, 'test')

        print '  Create batches..'

        create_batches(args, -1, test_x, test_y, test_e, test_sc, test_clean_y,
                       test_bm, test_sha, test_ch, test_rx,args.batch,
                       pad_id, stopwords, sort=False, model_type='test')

        print '  Purge references..'

        del test_x
        del test_y
        del test_e
        del test_clean_y
        del test_rx
        del test_sha

        print '  Finished Dev Proc.'


if __name__ == "__main__":
    args = summarization_args.get_args()
    main(args)