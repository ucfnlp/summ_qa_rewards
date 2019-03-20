import os
import json
import tempfile

import numpy as np
try:
    from pyrouge import Rouge155
except ImportError:
    print 'Not for eval'
from nn.basic import EmbeddingLayer, PositionEmbeddingLayer
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


def get_vocab(args):
    ifp = open('../data/'+ str(args.source) + '_vocab_' + str(args.vocab_size) + '.txt', 'r')

    vocab = []

    for line in ifp:
        vocab.append(line.rstrip())
    ifp.close()

    return vocab


def create_embedding_layer(args, path, vocab, embedding_dim, oov=None):

    embedding_layer = EmbeddingLayer(
        n_d=embedding_dim,
        vocab=vocab,
        embs=load_embedding_iterator(path) if path is not None else None,
        oov=oov,
        fix_init_embs=False
    )
    return embedding_layer


def create_posit_embedding_layer(vocab, embedding_dim):

    embedding_layer = PositionEmbeddingLayer(
        n_d=embedding_dim,
        vocab=vocab,
        fix_init_embs=False
    )
    return embedding_layer


def load_batches(name, iteration):
    ifp = open(name + str(iteration), 'rb')
    data = np.load(ifp)
    ifp.close()

    if len(data) == 4:
        return data[0], data[1], data[2], data[3]
    elif len(data) == 8:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
    elif len(data) == 9:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]
    else:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]


def create_fname_identifier(args):
    if args.word_level_c:
        chunk_typ = 'word'
    else:
        chunk_typ = 'chnk'
    if not hasattr(args, 'qa_performance'):
        args.qa_performance = ""
    return 'src_' + str(args.source) + \
           '_qap_' + args.qa_performance + \
           '_pretr_' + str(args.pretrain) + \
           '_load_pre_' + str(args.load_model_pretrain) + \
           '_edim_' + str(args.embedding_dim) + \
           '_batch_' + str(args.batch) + \
           '_inp_' + str(args.inp_len) + \
           '_epochs_' + str(args.max_epochs) + \
           '_layer_' + str(args.layer) + \
           '_gen_t_' + str(args.generator_encoding) + \
           '_gen_x_' + str(args.use_generator_h) + \
           '_bilin_' + str(args.bilinear) + \
           '_dp_' + str(args.dropout) + \
           '_ext_ck_' + str(args.extended_c_k) + \
           '_rl_nqa_' + str(args.rl_no_qa) + \
           '_ncl_' + str(args.nclasses) + \
           '_q' + str(args.n) + \
           '_ch_t_' + chunk_typ + \
           '_z_' + str(args.coeff_z) + \
           '_adq_' + str(args.coeff_adequacy) + \
           '_c_scl_' + str(args.coeff_cost_scale) + \
           '_zp_' + str(args.z_perc)


def create_json_filename(args):
    path = '../data/results/'

    if not os.path.exists(path):
        os.makedirs(path)

    filename = ('pretrain_' if args.pretrain else '') + create_fname_identifier(args) + '.json'

    return path + filename


def get_readable_file(args, epoch, test=False):
    path = args.train_output_readable

    if not os.path.exists(path):
        os.makedirs(path)

    filename = create_fname_identifier(args) + ('_e_' + str(epoch + 1)  if epoch is not None else '') + ('_TEST' if test else '') + '.out'

    return path + filename


def get_mask_file(args, epoch, test=False):
    path = "../data/results/masks/"

    if not os.path.exists(path):
        os.makedirs(path)

    filename = create_fname_identifier(args) + ('_e_' + str(epoch + 1)  if epoch is not None else '') + ('_TEST' if test else '') + '.out'

    return path + filename


def record_observations_pretrain(ofp_json, epoch , obj, zsum, z_diff, z_pred):
    epoch_data = dict()

    epoch_data['obj'] = float(np.mean(obj))
    epoch_data['zsum'] = float(np.mean(np.ravel(zsum)))
    epoch_data['zdiff'] = float(np.mean(np.ravel(z_diff)))
    epoch_data['z_pred'] = float(np.mean(z_pred))

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations_verbose(ofp_json, epoch, loss, obj, zsum, loss_vec, z_diff, cost_logpz,
                                logpz, z_pred, cost_vec, cost_g):
    epoch_data = dict()

    epoch_data['loss'] = float(np.mean(loss))
    epoch_data['obj'] = float(np.mean(obj))

    epoch_data['zsum'] = float(np.mean(zsum))
    epoch_data['loss_vec'] = float(np.mean(loss_vec))
    epoch_data['zdiff'] = float(np.mean(z_diff))

    epoch_data['cost_logpz'] = float(np.mean(cost_logpz))
    epoch_data['logpz'] = float(np.mean(logpz))
    epoch_data['z_pred'] = float(np.mean(z_pred))
    epoch_data['cost_vec'] = float(np.mean(cost_vec))
    epoch_data['cost_g'] = float(cost_g)

    ofp_json['e' + str(epoch)] = epoch_data


def save_dev_results(args, epoch, dev_z, dev_batches_x, dev_sha, dev_chunks=None):
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
            post_proc = set()

            continue_index = 0
            continue_count = 0
            skip_word = False

            for k in xrange(len(dev_z[i])):
                if k >= len(dev_batches_x[i][j]):
                    break

                word = dev_batches_x[i][j][k].encode('utf-8')

                if hasattr(args, 'post_proc'): # TODO: fix this asap
                    # print 'error'
                    if continue_count == 0:
                        continue_count = dev_chunks[i][continue_index][j]
                        next_chunk = retrieve_chunk(dev_batches_x[i][j], k, continue_count)
                        continue_index += 1

                        skip_word = next_chunk in post_proc

                        if dev_z[i][k][j] >= 0.5:
                            post_proc.add(next_chunk)

                continue_count -= 1
                if dev_z[i][k][j] < 0.5:
                    continue

                if skip_word:
                    continue

                ofp_for_rouge.write(word + ' ')
                ofp_system_output.append(word)

            raw_and_mask = dict()
            raw_and_mask['m'] = dev_z[i][:, j].tolist()
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


def retrieve_chunk(x, begin, end):
    return ' '.join(x[begin:begin + end]).encode('utf-8').lower()


def save_test_results_rouge(args, z, x, y, e, sha, embedding_layer, chunks=None):
    tempfile.tempdir = '/scratch/'
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

        start_hl_idx = 1

        for j in xrange(len(z[i][0])):
            filename = rouge_fname + 'sum.' + str(s_num).zfill(6) + '.txt'
            ofp_for_rouge = open(filename, 'w+')
            ofp_system_output = []

            word_idx = 0
            sent_idx = 0

            post_proc = set()
            raw_and_mask = dict()
            raw_and_mask_m = [0.0] * args.inp_len

            continue_index = 0
            continue_count = 0
            skip_word = False

            for k in xrange(len(z[i])):
                if word_idx >= len(x[i][j][sent_idx]):
                    sent_idx += 1
                    word_idx = 0

                if sent_idx >= len(x[i][j]):
                    break

                word = x[i][j][sent_idx][word_idx].encode('utf-8')
                word_idx += 1

                if not hasattr(args, 'post_proc'): # TODO: fix this asap
                    if continue_count == 0:
                        continue_count = chunks[i][continue_index][j]
                        next_chunk = retrieve_chunk(x[i][j][sent_idx], word_idx, continue_count)

                        continue_index += 1
                        skip_word = next_chunk in post_proc

                        if z[i][k][j] >= 0.5:
                            post_proc.add(next_chunk)

                continue_count -= 1
                if z[i][k][j] < 0.5:
                    continue

                if skip_word:
                    continue

                raw_and_mask_m[k] = 1.0
                ofp_for_rouge.write(word + ' ')
                ofp_system_output.append(word)

            raw_and_mask['m'] = raw_and_mask_m

            raw_and_mask['r'] = x[i][j][:]
            # raw_and_mask['y'] = []
            # raw_and_mask['y'] = create_eval_questions(args, y[i], e[i], index_to_e_map, embedding_layer, start_hl_idx,
            #                                           end_hl_idx, hl_step)

            ofp_m[sha[i][j]] = raw_and_mask

            ofp_samples_system.append(' '.join(ofp_system_output))
            ofp_samples_sha.append(sha[i][j])
            ofp_for_rouge.close()

            s_num += 1
            start_hl_idx += 1

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
        r = Rouge155(
            rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
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

    # create_readable_file_html(args=args, system_json=ofp_m)

    tmp_dir = r._config_dir
    print 'Cleaning up..', tmp_dir

    shutil.rmtree(tmp_dir)


def get_rouge(args):
    tempfile.tempdir = '/scratch/'

    model_specific_dir = create_fname_identifier(args).replace('.', '_') + '/'
    rouge_fname = args.system_summ_path + model_specific_dir

    if args.source == 'dm':
        r = Rouge155(rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
    else:
        r = Rouge155()

    r.system_dir = rouge_fname
    r.model_dir = args.model_summ_path + ('dev/' if args.source == 'cnn' else 'dm_dev/')
    # r.model_dir = args.model_summ_path + 'dev/'
    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = 'dev_'+args.source+'_#ID#.txt'

    fname = args.rouge_dir + create_fname_identifier(args) + '_rouge.out'
    ofp = open(fname, 'w+')

    ofp.write(r.convert_and_evaluate())
    ofp.close()

    tmp_dir = r._config_dir
    print 'Cleaning up..', tmp_dir

    shutil.rmtree(tmp_dir)


def create_1h(lste, n):
    loss_mask = [[] for _ in xrange(n)]
    e_processed = [[] for _ in xrange(n)]
    for i in xrange(len(lste)):
        for j in xrange(len(lste[i])):
            if j == n:
                break
            e_processed[j].append(lste[i][j])
            loss_mask[j].append(1)

        # For the case of not having padded y
        if len(lste[i]) < n:
            for j in range(len(lste[i]), n):
                e_processed[j].append(0)
                loss_mask[j].append(0)

    be = []
    lm = []

    for i in xrange(len(e_processed)):
        assert len(e_processed[i]) == len(loss_mask[i])
        be.extend(e_processed[i])
        lm.extend(loss_mask[i])

    return be, lm


def total_words(z):
    return np.sum(z, axis=None)
