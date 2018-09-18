import json
from nltk.tokenize import RegexpTokenizer
import numpy as np
import random

import summarization_args


def read_docs(args, type_):
    filename = type_ + '_model.json' if args.full_test else "small_" + type_ + '_model.json'
    filename = '../data/'+ args.source + '_' + str(args.vocab_size) + '_' + filename

    with open(filename, 'rb') as data_file:
        data = json.load(data_file)

    ret_data = [data['x'], data['y'], data['e'], data['clean_y'], data['sha'], data['mask'], data['chunk'], data['scut']]

    if type_ != 'train':
        ret_data.append(data['raw_x'])
    else:
        ret_data.append(None)

    return ret_data


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


def create_batches(args, x, y, entities, sentence_sizes, clean_indexed_y, sha, chunk_sizes, raw_x, padding_id,
                   stopwords, sort=True, model_type=''):

    batch_size = args.batch
    batches_x, batches_y, batches_entities, batches_overlap_mask, batches_sha = [], [], [], [], []
    batches_raw_x, batches_fw_m, batches_chunk_sizes, batches_sentence_idx = [], [], [], []

    N = len(x)
    M = (N - 1) / batch_size + 1
    num_batches = 0
    num_files = 0

    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
        entities = [entities[i] for i in perm]
        sentence_sizes = [sentence_sizes[i] for i in perm]
        clean_indexed_y = [clean_indexed_y[i] for i in perm]
        chunk_sizes = [chunk_sizes[i] for i in perm]
        sha = [sha[i] for i in perm]

    for i in xrange(M):
        single_batch_x, single_batch_y, single_batch_overlap_mask, single_batch_fw_m, single_batch_chunk_sizes, single_batch_sentence_idx = create_one_batch(
            args,
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size],
            sentence_sizes[i * batch_size:(i + 1) * batch_size],
            clean_indexed_y[i * batch_size:(i + 1) * batch_size],
            chunk_sizes[i * batch_size:(i + 1) * batch_size],
            padding_id,
            stopwords
        )
        single_batch_sha = sha[i * batch_size:(i + 1) * batch_size]
        single_batch_entities = entities[i * batch_size:(i + 1) * batch_size]

        batches_x.append(single_batch_x)
        batches_y.append(single_batch_y)
        batches_entities.append(single_batch_entities)
        batches_overlap_mask.append(single_batch_overlap_mask)
        batches_sha.append(single_batch_sha)
        batches_fw_m.append(single_batch_fw_m)
        batches_chunk_sizes.append(single_batch_chunk_sizes)
        batches_sentence_idx.append(single_batch_sentence_idx)

        if raw_x is not None:
            single_batch_raw_x = raw_x[i * batch_size:(i + 1) * batch_size]
            batches_raw_x.append(single_batch_raw_x)

        num_batches += 1

        if num_batches >= args.online_batch_size or i == M - 1:
            fname = args.batch_dir + args.source + model_type
            print 'Creating file #', str(num_files + 1)

            data = [batches_x, batches_y, batches_entities, batches_overlap_mask, batches_sha]

            if raw_x is not None:
                data.append(batches_raw_x)

            data.extend([batches_fw_m, batches_chunk_sizes, batches_sentence_idx])

            with open(fname + str(num_files), 'w+') as ofp:
                np.save(ofp, data)

            batches_x, batches_y, batches_entities, batches_overlap_mask, batches_sha = [], [], [], [], []
            batches_raw_x, batches_fw_m, batches_chunk_sizes, batches_sentence_idx = [], [], [], []

            num_batches = 0
            num_files += 1

    print "Num Files :", num_files


def create_one_batch(args, lst_x, lst_y, lst_sentence_sizes, lst_clean_y, lst_chunk_sizes, padding_id, stopwords):
    max_len = args.inp_len

    assert min(len(x) for x in lst_x) > 0

    single_batch_y, unigrams = process_hl(args, lst_y, padding_id, lst_clean_y)
    single_batch_y = np.column_stack([y for y in single_batch_y])

    single_batch_fw_m, single_batch_chunk_sizes = create_chunk_mask(lst_chunk_sizes, max_len)

    single_batch_x = np.column_stack([np.pad(x[:max_len], (0, max_len - len(x) if len(x) <= max_len else 0), "constant",
                                             constant_values=padding_id).astype('int32') for x in lst_x])
    assert unigrams is not None

    single_batch_overlap_mask = create_unigram_masks(lst_x, unigrams, max_len, stopwords, args)

    if not args.word_level_c:
        single_batch_overlap_mask = create_chunk_masks(single_batch_overlap_mask, single_batch_chunk_sizes, max_len)

    single_batch_overlap_mask = np.column_stack([m for m in single_batch_overlap_mask])

    single_batch_sentence_idx = sentence_indexing(lst_sentence_sizes, max_len)

    return single_batch_x, single_batch_y, single_batch_overlap_mask, single_batch_fw_m, single_batch_chunk_sizes, single_batch_sentence_idx


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


def process_hl(args, lsty, padding_id, lstcy):
    max_len_y = args.hl_len

    y_processed = [[] for _ in xrange(args.n)]

    unigrams = []

    for i in xrange(len(lsty)):
        sample_u = set()

        for j in xrange(len(lsty[i])):
            if j == args.n:
                break
            y = lsty[i][j][:max_len_y]
            single_hl = np.pad(y, (0, max_len_y - len(y)), "constant", constant_values=padding_id).astype('int32')

            y_processed[j].append(single_hl)

        for j in range(len(lsty[i]), args.n):
            y_processed[j].append(np.full((max_len_y,), fill_value=padding_id).astype('int32'))

        for clean_hl in lstcy[i]:
            trimmed_cy = clean_hl[:max_len_y]

            for token in trimmed_cy:
                sample_u.add(token)

        unigrams.append(sample_u)

    by = []
    for i in xrange(len(y_processed)):
        by.extend(y_processed[i])

    return by, unigrams


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


def create_chunk_mask(lstch, max_len):
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

            mask_csz.append(left)

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

        indexed_x.append(single_doc)

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
                    try:
                        m[j] = 1
                        m[j+1] = 1
                    except IndexError:
                        continue

        masks.append(m)

    return masks


def create_chunk_masks(word_level_bm, bsz, max_len):
    masks = []

    for i in xrange(len(word_level_bm)):
        m = []
        chunks = bsz[:, i]
        gs_words = word_level_bm[i]
        end = 0

        for c in chunks:
            begin = end
            end = begin + c

            use_cur_chunk = False

            for j in range(begin, end):
                if gs_words[j] > 0:
                    use_cur_chunk = True
                    break

            if use_cur_chunk:
                cur_chunk = [1] * c
            else:
                cur_chunk = [0] * c
            m.extend(cur_chunk)

        if len(m) < max_len:
            m.extend([0]*(max_len - len(m)))

        masks.append(m)

    return np.asarray(masks, dtype='int32')


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

    type_ls = ['train', 'dev', 'test']

    for type_ in type_ls:
        print type_, ':'
        print '  Read JSON..'
        cur_data = read_docs(args, type_)

        create_batches(args=args,
                       x=cur_data[0],
                       y=cur_data[1],
                       entities=cur_data[2],
                       sentence_sizes=cur_data[7],
                       clean_indexed_y=cur_data[3],
                       sha=cur_data[4],
                       chunk_sizes=cur_data[6],
                       raw_x=cur_data[8],
                       padding_id=pad_id,
                       stopwords=stopwords,
                       sort=(type_ == 'train'),
                       model_type=type_)

        print '  Purge references..'
        del cur_data
        print '  Finished', type_


if __name__ == "__main__":
    args = summarization_args.get_args()
    main(args)
