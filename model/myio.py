import gzip
import json
import random

import numpy as np
# from pyrouge import Rouge155

from nn.basic import EmbeddingLayer
from util import load_embedding_iterator


def read_docs(args, type):
    filename = type + '_model.json' if args.full_test else "small_" + type + '_model.json'
    filename = '../data/' + str(args.vocab_size) + '_' + filename

    with open(filename, 'rb') as data_file:
        data = json.load(data_file)

    data_x = data['x']
    data_y = data['y']
    data_e = data['e']

    data_file.close()

    if type == 'test':
        return data_x

    return data_x, data_y, data_e


def load_e(args):
    filename = args.entities if args.full_test else "small_" + args.entities
    filename = '../data/' + str(args.vocab_size) + '_' + filename

    with open(filename, 'rb') as data_file:
        data = json.load(data_file)

    entites = data['entities']
    data_file.close()

    return entites


def get_vocab(args):
    ifp = open('../data/vocab_' + str(args.vocab_size) + '.txt', 'r')
    vocab = []

    for line in ifp:
        vocab.append(line.rstrip())

    ifp.close()

    return vocab


def create_embedding_layer(args, path, vocab):

    embedding_layer = EmbeddingLayer(
        n_d=args.embedding_dim,
        vocab=vocab,
        embs=load_embedding_iterator(path),
        oov="<unk>",
        fix_init_embs = True
        # fix_init_embs=False
    )
    return embedding_layer


def create_test(args, x, padding_id):
    batches_x = []
    batch_size = args.batch
    max_len = args.inp_len

    N = len(x)
    M = (N - 1) / batch_size + 1

    for i in xrange(M):
        lstx = x[i * batch_size:(i + 1) * batch_size]
        bx = np.column_stack([np.pad(x[:max_len], (max_len - len(x) if len(x) <= max_len else 0, 0), "constant",
                                     constant_values=padding_id).astype('int32') for x in lstx])

        batches_x.append(bx)

    return batches_x


def create_batches(args, n_classes, x, y, e, batch_size, padding_id, sort=True):
    batches_x, batches_y, batches_e, batches_bm  = [], [], [], []
    N = len(x)
    M = (N - 1) / batch_size + 1

    if args.sanity_check:
        sort= False
        M = 1

    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
        e = [e[i] for i in perm]

    for i in xrange(M):
        bx, by, be, bm = create_one_batch(
            args,
            n_classes,
            x[i * batch_size:(i + 1) * batch_size],
            y[i * batch_size:(i + 1) * batch_size],
            e[i * batch_size:(i + 1) * batch_size],
            padding_id,
            batch_size
        )

        batches_x.append(bx)
        batches_y.append(by)
        batches_e.append(be)
        batches_bm.append(bm)

    if sort:
        random.seed(5817)
        perm2 = range(M)
        random.shuffle(perm2)
        batches_x = [batches_x[i] for i in perm2]
        batches_y = [batches_y[i] for i in perm2]
        batches_e = [batches_e[i] for i in perm2]
        batches_bm = [batches_bm[i] for i in perm2]

    return batches_x, batches_y, batches_e, batches_bm


def create_one_batch(args, n_classes, lstx, lsty, lste, padding_id, b_len):
    """
    Parameters
    ----------
    lstx : List of 1D documents.

    lsty : List of list of sentences
        usually a preselected limit of input Highlights

    lstve : List of all valid entity indexes to use in loss

    lste : For each hl in lsty, the correct class

    """
    max_len = args.inp_len

    assert min(len(x) for x in lstx) > 0

    # padded y
    by, unigrams, be = process_hl(args, lsty, lste,padding_id, n_classes)

    bx = np.column_stack([np.pad(x[:max_len], (max_len - len(x) if len(x) <= max_len else 0, 0), "constant",
                                 constant_values=padding_id).astype('int32') for x in lstx])

    bm = create_unigram_masks(lstx, unigrams, max_len)
    # be = np.column_stack([e for e in be])
    bm = np.column_stack([m for m in bm])
    by = np.column_stack([y for y in by])

    return bx, by, be, bm


def round_batch(lstx, lsty, lste, b_len):
    lstx_rounded, lsty_rounded, lste_rounded = [],[],[]
    missing = b_len - len(lstx)

    while missing > 0:
        missing = b_len - len(lstx_rounded)
        lstx_rounded.extend(lstx[:])
        lsty_rounded.extend(lsty[:])
        lste_rounded.extend(lste[:])

    return lstx_rounded[:b_len], lsty_rounded[:b_len], lste_rounded[:b_len]


def process_hl(args, lsty, lste, padding_id, n_classes):
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

            for token in y:
                sample_u.add(token)

        unigrams.append(sample_u)

    by = []
    be = []
    for i in xrange(len(y_processed)):
        by.extend(y_processed[i])
        be.extend(e_processed[i])

    return by, unigrams, be


def create_unigram_masks(lstx, unigrams, max_len):
    masks = []

    for i in xrange(len(lstx)):
        len_x = len(lstx[i])
        m = np.zeros((max_len,),dtype='int32')

        for j in xrange(len_x - 1):
            if lstx[i][j] in unigrams[i] and lstx[i][j+1] in unigrams[i]:
                m[i] = 1

        masks.append(m)

    return masks


def process_ent(n_classes, lste):
    ret_e = []

    for e in lste:
        e_mask = np.zeros((n_classes,),dtype='int32')

        for e_idx in e:
            e_mask[e_idx] = 1

        ret_e.append(e_mask)

    return ret_e


def create_fname_identifier(args):
    return 'source_' + str(args.source) + \
           '_train_data_embdim_' + str(args.embedding_dim) + \
           '_vocab_size_' + str(args.vocab_size) + \
           '_batch_' + str(args.batch) + \
           '_epochs_' + str(args.max_epochs) + \
           '_coeff_summ_len_' + str(args.coeff_summ_len) + \
           '_coeff_adequacy_' + str(args.coeff_adequacy) + \
           '_coeff_fluency_' + str(args.coeff_fluency)


def create_json_filename(args):
    path = '../data/results/'
    filename = create_fname_identifier(args) + '.json'

    return path + filename


def get_readable_file(args, epoch):
    path = args.train_output_readable
    filename = create_fname_identifier(args) +'_e_' + str(epoch + 1) + '.out'

    return path + filename


def record_observations(ofp_json, epoch, loss, obj, zsum, bigram_loss, loss_vec_all, z_diff):
    epoch_data = dict()

    epoch_data['loss'] = [l.tolist() for l in loss]
    epoch_data['obj'] = [l.tolist() for l in obj]
    epoch_data['zsum'] = [l.tolist() for l in zsum]
    epoch_data['bigram_loss'] = [l.tolist() for l in bigram_loss]
    epoch_data['loss_vec'] = [l.tolist() for l in loss_vec_all]
    epoch_data['zdiff'] = [l.tolist() for l in z_diff]

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations(ofp_json, epoch, loss,obj,zsum,loss_vec,z_diff,cost_logpz,logpz,probs,z_pred,cost_vec):
    epoch_data = dict()

    epoch_data['loss'] = [l.tolist() for l in loss]
    epoch_data['obj'] = [l.tolist() for l in obj]
    epoch_data['zsum'] = [l.tolist() for l in zsum]
    epoch_data['loss_vec'] = [l.tolist() for l in loss_vec]
    epoch_data['zdiff'] = [l.tolist() for l in z_diff]

    epoch_data['cost_logpz'] = [l.tolist() for l in cost_logpz]
    epoch_data['logpz'] = [l.tolist() for l in logpz]
    epoch_data['probs'] = [l.tolist() for l in probs]
    epoch_data['z_pred'] = [l.tolist() for l in z_pred]
    epoch_data['cost_vec'] = [l.tolist() for l in cost_vec]

    ofp_json['e' + str(epoch)] = epoch_data


def save_dev_results(args, epoch, dev_z, dev_batches_x, emb_layer):
    s_num = 0

    filename = get_readable_file(args, epoch)
    ofp_samples = open(filename, 'w+')
    ofp_samples_system = []

    for i in xrange(len(dev_z)):

        for j in xrange(len(dev_z[i][0])):

            ofp_for_rouge = open(args.system_summ_path + 'e_' + str(epoch + 1) + '.' + str(s_num) + '.txt', 'w+')
            ofp_system_output = []

            for k in xrange(len(dev_z[i])):
                word = emb_layer.lst_words[dev_batches_x[i][k][j]]

                if word == '<padding>' or word == '<unk>' or dev_z[i][k][j] == 0:
                    continue

                ofp_for_rouge.write(word + ' ')
                ofp_system_output.append(word)

            ofp_samples_system.append(' '.join(ofp_system_output))
            ofp_for_rouge.close()
            s_num += 1

    for i in xrange(len(ofp_samples_system)):

        ofp_samples.write('System Summary :')

        if len(ofp_samples_system[i]) == 0:
            ofp_samples.write('**No Summary**')
        else:
            ofp_samples.write(ofp_samples_system[i])

        ofp_samples.write('\n\n')

    ofp_samples.close()


def save_test_results_rouge(args, z, test_batches_x, emb_layer):
    s_num = 0

    for i in xrange(len(z)):

        for j in xrange(len(z[i][0])):

            ofp_for_rouge = open(args.system_summ_path + 'test.' + str(s_num) + '.txt', 'w+')

            for k in xrange(len(z[i])):
                word = emb_layer.lst_words[test_batches_x[i][k][j]]

                if word == '<padding>' or word == '<unk>' or z[i][k][j] == 0:
                    continue

                ofp_for_rouge.write(word + ' ')

            ofp_for_rouge.close()
            s_num += 1


def get_rouge(args, epoch):
    r = Rouge155()
    r.system_dir = args.system_summ_path
    r.model_dir = args.model_summ_path
    r.system_filename_pattern = 'e_' + str(epoch + 1) + '.(\d+).txt'
    r.model_filename_pattern = 'e_' + str(epoch + 1) + '.#ID#.txt'

    return r.convert_and_evaluate()


def get_ngram(l, n=2):

    return set(zip(*[l[i:] for i in range(n)]))


def convert_bv_to_z(bv):
    dz = []

    for batch in bv:
        dz_batch = []
        for b in xrange(len(batch[0])):
            dz_b = [0] * len(batch)
            for i in xrange(len(batch) - 1):
                if batch[i][b] == 1:
                    dz_b[i] = dz_b[i + 1] = 1

            dz_batch.append(dz_b)

        dz.append(np.column_stack([z for z in dz_batch]))

    return dz


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