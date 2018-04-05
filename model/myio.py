import os
import json

import numpy as np
from pyrouge import Rouge155
from nn.basic import EmbeddingLayer
from util import load_embedding_iterator
import shutil


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


def load_e(args):
    filename = args.entities if args.full_test else "small_" + args.entities
    filename = '../data/' + args.source + '_' + str(args.vocab_size) + '_' + filename

    with open(filename, 'rb') as data_file:
        data = json.load(data_file)

    entites = data['entities']
    data_file.close()

    return entites


def get_vocab(args):
    ifp = open('../data/'+ str(args.source) + '_vocab_' + str(args.vocab_size) + '.txt', 'r')
    ifp_pt = open('../data/' + str(args.source) + '_vocab_pt_.txt', 'r')

    vocab = []
    vocab_pt = []
    pt_dict = dict()

    for line in ifp:
        vocab.append(line.rstrip())
    ifp.close()

    for line in ifp_pt:
        items = line.rstrip().split()
        pt_dict[int(items[1])] = items[0]

    for key in sorted(pt_dict.keys()):
        vocab_pt.append(pt_dict[key])

    ifp_pt.close()

    return vocab, vocab_pt


def create_embedding_layer(args, path, vocab, oov=None):

    embedding_layer = EmbeddingLayer(
        n_d=args.embedding_dim,
        vocab=vocab,
        embs=load_embedding_iterator(path) if path is not None else None,
        oov=oov,
        fix_init_embs = False
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


def load_batches(name, iteration):
    ifp = open(name + str(iteration), 'rb')
    data = np.load(ifp)
    ifp.close()
    if len(data) == 8:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
    elif len(data) == 7:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6]
    else:
        return data[0], data[1], data[2], data[3], data[4]


def round_batch(lstx, lsty, lste, b_len):
    lstx_rounded, lsty_rounded, lste_rounded = [],[],[]
    missing = b_len - len(lstx)

    while missing > 0:
        missing = b_len - len(lstx_rounded)
        lstx_rounded.extend(lstx[:])
        lsty_rounded.extend(lsty[:])
        lste_rounded.extend(lste[:])

    return lstx_rounded[:b_len], lsty_rounded[:b_len], lste_rounded[:b_len]


def create_fname_identifier(args):
    return 'source_' + str(args.source) + \
           '_pretrain_' + str(args.pretrain) + \
           '_load_model_pre_' + str(args.load_model_pretrain) + \
           '_train_data_edim_' + str(args.embedding_dim) + \
           '_vocab_' + str(args.vocab_size) + \
           '_batch_' + str(args.batch) + \
           '_inplen_' + str(args.inp_len) + \
           '_epochs_' + str(args.max_epochs) + \
           '_layer_' + str(args.layer) + \
           '_bilin_' + str(args.bilinear) + \
           '_ncl_' + str(args.nclasses) + \
           '_q' + str(args.n) + \
           '_root_' + str(args.is_root) + \
           '_cf_z_' + str(args.coeff_z) + \
           '_cf_adq_' + str(args.coeff_adequacy) + \
           '_cf_cst_scl_' + str(args.coeff_cost_scale)


def create_json_filename(args):
    path = '../data/results/'
    filename = ('pretrain_' if args.pretrain else '') + create_fname_identifier(args) + '.json'

    return path + filename


def get_readable_file(args, epoch, test=False):
    path = args.train_output_readable
    filename = create_fname_identifier(args) + ('_e_' + str(epoch + 1)  if epoch is not None else '') + ('_TEST' if test else '') + '.fson'

    return path + filename


def get_mask_file(args, epoch, test=False):
    path = args.train_output_mask
    filename = create_fname_identifier(args) + ('_e_' + str(epoch + 1)  if epoch is not None else '') + ('_TEST' if test else '') + '.out'

    return path + filename


def record_observations(ofp_json, epoch, loss, obj, zsum, loss_vec, z_diff):
    epoch_data = dict()

    epoch_data['loss'] = float(np.mean(loss))
    epoch_data['obj'] = float(np.mean(obj))
    epoch_data['zsum'] = float(np.mean(zsum))
    epoch_data['loss_vec'] = float(np.mean(loss_vec))
    epoch_data['zdiff'] = float(np.mean(z_diff))

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations_pretrain(ofp_json, epoch , obj, zsum, z_diff, z_pred):
    epoch_data = dict()

    epoch_data['obj'] = float(np.mean(obj))
    epoch_data['zsum'] = float(np.mean(np.ravel(zsum)))
    epoch_data['zdiff'] = float(np.mean(np.ravel(z_diff)))
    epoch_data['z_pred'] = float(np.mean(z_pred))

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations_verbose(ofp_json, epoch, loss, obj, zsum, loss_vec, z_diff, cost_logpz, logpz, z_pred, cost_vec, bigram_loss, dev_acc, train_acc):
    epoch_data = dict()

    epoch_data['loss'] = float(np.mean(loss))
    epoch_data['obj'] = float(np.mean(obj))
    epoch_data['dev_acc'] = dev_acc
    epoch_data['train_acc'] = train_acc
    epoch_data['zsum'] = float(np.mean(zsum))
    epoch_data['loss_vec'] = float(np.mean(loss_vec))
    epoch_data['zdiff'] = float(np.mean(z_diff))
    epoch_data['bigram_loss'] = float(np.mean(bigram_loss))

    epoch_data['cost_logpz'] = float(np.mean(cost_logpz))
    epoch_data['logpz'] = float(np.mean(logpz))
    epoch_data['z_pred'] = float(np.mean(z_pred))
    epoch_data['cost_vec'] = float(np.mean(cost_vec))

    ofp_json['e' + str(epoch)] = epoch_data


def save_dev_results(args, epoch, dev_z, dev_batches_x, dev_sha):
    s_num = 0

    filename_ = get_readable_file(args, epoch)
    filename_m = get_mask_file(args, epoch)

    ofp_samples = open(filename_, 'w+')
    ofp_samples_m = open(filename_m, 'w+')

    ofp_samples_system = []
    ofp_m = dict()
    ofp_samples_sha = []

    model_specific_dir = create_fname_identifier(args).replace('.', '_') + '/'
    rouge_fname = args.system_summ_path + model_specific_dir

    if not os.path.exists(rouge_fname):
        os.mkdir(rouge_fname)

    for i in xrange(len(dev_z)):

        for j in xrange(len(dev_z[i][0])):
            filename = rouge_fname + 'sum.' + str(s_num).zfill(6) + '.txt'

            ofp_for_rouge = open(filename, 'w+')
            ofp_system_output = []

            for k in xrange(len(dev_z[i])):
                if k >= len(dev_batches_x[i][j]):
                    break

                word = dev_batches_x[i][j][k].encode('utf-8')

                if dev_z[i][k][j] < 0.5:
                    continue

                ofp_for_rouge.write(word + ' ')
                ofp_system_output.append(word)

            raw_and_mask = dict()
            raw_and_mask['m'] = list(dev_z[i][:,j])
            raw_and_mask['r'] = dev_batches_x[i][j][:]

            ofp_m[dev_sha[i][j]] = raw_and_mask

            ofp_samples_system.append(' '.join(ofp_system_output))
            ofp_samples_sha.append(dev_sha[i][j])
            ofp_for_rouge.close()
            s_num += 1

    for i in xrange(len(ofp_samples_system)):
        ofp_samples.write(str(ofp_samples_sha[i]))
        ofp_samples.write('\nSystem Summary : ')

        if len(ofp_samples_system[i]) == 0:
            ofp_samples.write('**No Summary**')
        else:
            ofp_samples.write(ofp_samples_system[i])

        ofp_samples.write('\n\n')

    json.dump(ofp_m, ofp_samples_m)

    ofp_samples_m.close()
    ofp_samples.close()


def eval_baseline(args, bm, rx, type_):
    s_num = 0

    filename_ = args.train_output_readable + 'baseline_'+ type_ + '.out'
    ofp_samples = open(filename_, 'w+')
    ofp_samples_system = []

    rouge_fname = args.system_summ_path + 'baseline_' + type_ + '_' + args.source + '/'

    for i in xrange(len(bm)):

        for j in xrange(len(bm[i][0])):

            filename = rouge_fname + 'sum.' + str(s_num).zfill(6) + '.txt'
            ofp_for_rouge = open(filename, 'w+')
            ofp_system_output = []

            binary_mask = bm[i][:,j]
            binary_shifted = np.pad(binary_mask[:-1], (1,0), "constant", constant_values=0)
            complete_bm = binary_mask | binary_shifted

            for k in xrange(len(complete_bm)):
                if k >= len(rx[i][j]):
                    break

                word = rx[i][j][k].encode('utf-8')

                if complete_bm[k] == 0:
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

    if args.source == 'dm':
        r = Rouge155(rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
    else:
        r = Rouge155()

    r.system_dir = rouge_fname
    r.model_dir = args.model_summ_path
    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = type_ + '_' + args.source + '_#ID#.txt'

    fname = args.rouge_dir + 'baseline_rouge_' + args.source + '_' + type_ + '.out'
    ofp = open(fname, 'w+')

    ofp.write(r.convert_and_evaluate())
    ofp.close()

    tmp_dir = r._config_dir
    print 'Cleaning up..', tmp_dir

    shutil.rmtree(tmp_dir)


def save_test_results_rouge(args, z, x, sha):
    s_num = 0
    epoch = None

    filename_ = get_readable_file(args, epoch, test=True)
    filename_m = get_mask_file(args, epoch, test=True)

    ofp_samples = open(filename_, 'w+')
    ofp_samples_m = open(filename_m, 'w+')

    ofp_samples_system = []
    ofp_samples_sha = []

    ofp_m = dict()

    model_specific_dir = create_fname_identifier(args).replace('.', '_') + '_TEST/'
    rouge_fname = args.system_summ_path + model_specific_dir

    if not os.path.exists(rouge_fname):
        os.mkdir(rouge_fname)

    for i in xrange(len(z)):

        for j in xrange(len(z[i][0])):
            filename = rouge_fname + 'sum.' + str(s_num).zfill(6) + '.txt'
            ofp_for_rouge = open(filename, 'w+')
            ofp_system_output = []

            for k in xrange(len(z[i])):
                if k >= len(x[i][j]):
                    break

                word = x[i][j][k].encode('utf-8')

                if z[i][k][j] < 0.5:
                    continue

                ofp_for_rouge.write(word + ' ')
                ofp_system_output.append(word)

            raw_and_mask = dict()
            raw_and_mask['m'] = list(z[i][:, j])
            raw_and_mask['r'] = x[i][j][:]

            ofp_m[sha[i][j]] = raw_and_mask

            ofp_samples_system.append(' '.join(ofp_system_output))
            ofp_samples_sha.append(sha[i][j])
            ofp_for_rouge.close()
            s_num += 1

    for i in xrange(len(ofp_samples_system)):
        ofp_samples.write(str(ofp_samples_sha[i]))
        ofp_samples.write('\nSystem Summary : ')

        if len(ofp_samples_system[i]) == 0:
            ofp_samples.write('**No Summary**')
        else:
            ofp_samples.write(ofp_samples_system[i])

        ofp_samples.write('\n\n')

    json.dump(ofp_m, ofp_samples_m)

    ofp_samples_m.close()
    ofp_samples.close()

    if args.source == 'dm':
        r = Rouge155(rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
    else:
        r = Rouge155()

    r.system_dir = rouge_fname
    r.model_dir = args.model_summ_path + ('test/' if args.source == 'cnn' else 'dm_test/')
    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = 'test_' + args.source + '_#ID#.txt'

    fname = args.rouge_dir + create_fname_identifier(args) + '_test.out'
    ofp = open(fname, 'w+')

    ofp.write(r.convert_and_evaluate())
    ofp.close()

    tmp_dir = r._config_dir
    print 'Cleaning up..', tmp_dir

    shutil.rmtree(tmp_dir)


def save_dev_results_r(args, probs, x, embedding):
    s_num = 0
    if not os.path.exists(args.system_summ_path):
        os.makedirs(args.system_summ_path)

    for i in xrange(len(probs)):

        filename = args.system_summ_path + 'a.' + str(s_num).zfill(6) + '.txt'
        ofp_for_rouge = open(filename, 'w+')
        for j in xrange(len(probs[i][0])):

            for k in xrange(len(probs[i])):
                word = x[i][k][j]

                if probs[i][k][j] < 0.5:
                    continue

                ofp_for_rouge.write(str(word+ ' '))

        ofp_for_rouge.close()
        s_num += 1

    get_rouge(args, 'a')


def get_rouge(args):

    model_specific_dir = create_fname_identifier(args).replace('.', '_') + '/'
    rouge_fname = args.system_summ_path + model_specific_dir

    if args.source == 'dm':
        r = Rouge155(rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
    else:
        r = Rouge155()

    r.system_dir = rouge_fname
    r.model_dir = args.model_summ_path + 'dev/'
    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = 'dev_cnn_#ID#.txt'

    fname = args.rouge_dir + create_fname_identifier(args) + '_rouge.out'
    ofp = open(fname, 'w+')

    ofp.write(r.convert_and_evaluate())
    ofp.close()

    tmp_dir = r._config_dir
    print 'Cleaning up..', tmp_dir

    shutil.rmtree(tmp_dir)


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
