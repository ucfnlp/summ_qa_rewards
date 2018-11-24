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

    if len(data) == 4:
        return data[0], data[1], data[2], data[3]
    elif len(data) == 8:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
    elif len(data) == 9:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]
    else:
        return data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]


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
    if args.word_level_c:
        chunk_typ = 'word'
    else:
        chunk_typ = 'chnk'

    return 'src_' + str(args.source) + \
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
           '_root_' + str(args.is_root) + \
           '_ch_t_' + chunk_typ + \
           '_z_' + str(args.coeff_z) + \
           '_adq_' + str(args.coeff_adequacy) + \
           '_c_scl_' + str(args.coeff_cost_scale) + \
           '_zp_' + str(args.z_perc)


def create_fname_identifier_qa(args):

    return 'src_' + str(args.source) + \
           '_edim_' + str(args.embedding_dim) + \
           '_batch_' + str(args.batch) + \
           '_inp_' + str(args.inp_len) + \
           '_epochs_' + str(args.max_epochs) + \
           '_layer_' + str(args.layer) + \
           '_bilin_' + str(args.bilinear) + \
           '_dp_' + str(args.dropout) + \
           '_ext_ck_' + str(args.extended_c_k) + \
           '_ncl_' + str(args.nclasses) + \
           '_q' + str(args.n) + \
           '_root_' + str(args.is_root) + \
           '_c_scl_' + str(args.coeff_cost_scale) + \
           '_hl_only_' + str(args.qa_hl_only) + \
           '_prune_ov_' + str(args.use_overlap) + \
           '_prune_perc_' + str(args.x_sample_percentage)


def create_json_filename(args):
    path = '../data/results/'
    filename = ('pretrain_' if args.pretrain else '') + create_fname_identifier(args) + '.json'

    return path + filename


def create_json_filename_qa(args):
    path = '../data/results/'
    filename = ('pretrain_' if args.pretrain else '') + create_fname_identifier_qa(args) + '.json'

    return path + filename


def get_readable_file(args, epoch, test=False):
    path = args.train_output_readable
    filename = create_fname_identifier(args) + ('_e_' + str(epoch + 1)  if epoch is not None else '') + ('_TEST' if test else '') + '.out'

    return path + filename


def get_mask_file(args, epoch, test=False):
    path = "../data/results/masks/"#args.train_output_mask
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


def record_observations_leaks(ofp_json, epoch, soft_mask_ls, z_pred_ls, mask_ls):
    epoch_data = dict()

    epoch_data['soft_mask'] = soft_mask_ls
    epoch_data['z_pred'] = z_pred_ls
    epoch_data['mask'] = mask_ls

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations_pretrain(ofp_json, epoch , obj, zsum, z_diff, z_pred):
    epoch_data = dict()

    epoch_data['obj'] = float(np.mean(obj))
    epoch_data['zsum'] = float(np.mean(np.ravel(zsum)))
    epoch_data['zdiff'] = float(np.mean(np.ravel(z_diff)))
    epoch_data['z_pred'] = float(np.mean(z_pred))

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations_verbose(ofp_json, epoch, loss, obj, zsum, loss_vec, z_diff, cost_logpz,
                                logpz, z_pred, cost_vec, bigram_loss, dev_acc, dev_f1, train_acc,
                                train_f1, l2_enc, l2_gen, cost_g):
    epoch_data = dict()

    epoch_data['loss'] = float(np.mean(loss))
    epoch_data['obj'] = float(np.mean(obj))
    epoch_data['dev_acc'] = dev_acc
    epoch_data['dev_f1'] = dev_f1
    epoch_data['train_acc'] = train_acc
    epoch_data['train_f1'] = train_f1
    epoch_data['zsum'] = float(np.mean(zsum))
    epoch_data['loss_vec'] = float(np.mean(loss_vec))
    epoch_data['zdiff'] = float(np.mean(z_diff))
    epoch_data['bigram_loss'] = float(np.mean(bigram_loss))

    epoch_data['cost_logpz'] = float(np.mean(cost_logpz))
    epoch_data['logpz'] = float(np.mean(logpz))
    epoch_data['z_pred'] = float(np.mean(z_pred))
    epoch_data['cost_vec'] = float(np.mean(cost_vec))
    epoch_data['cost_g'] = float(cost_g)

    epoch_data['l2_gen'] = float(l2_gen)
    epoch_data['l2_enc'] = float(l2_enc)

    ofp_json['e' + str(epoch)] = epoch_data


def record_observations_verbose_qa(ofp_json, epoch, loss, loss_vec, dev_acc,
                                dev_f1, train_acc, train_f1):
    epoch_data = dict()

    epoch_data['loss'] = float(np.mean(loss))
    epoch_data['dev_acc'] = dev_acc
    epoch_data['dev_f1'] = dev_f1
    epoch_data['train_acc'] = train_acc
    epoch_data['train_f1'] = train_f1
    epoch_data['loss_vec'] = float(np.mean(loss_vec))

    ofp_json['e' + str(epoch)] = epoch_data


def save_dev_results_s(args, epoch, dev_z, dev_batches_x, dev_sha):
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
            sentence_idx = 0
            sample_mask = []
            sample_flat = []

            for k in xrange(len(dev_z[i])):
                if sentence_idx >= len(dev_batches_x[i][j]):
                    break

                if k%45 == 0:
                    sentence = dev_batches_x[i][j][sentence_idx][:45]
                    sample_flat.extend(sentence)

                    sentence_idx += 1

                    if dev_z[i][k][j] < 0.5:
                        sample_mask.extend([0]*len(sentence))
                        continue

                    sample_mask.extend([1] * len(sentence))

                    ofp_for_rouge.write(' '.join(sentence).encode('utf-8'))
                    ofp_system_output.extend(sentence)

            raw_and_mask = dict()
            raw_and_mask['m'] = sample_mask
            raw_and_mask['r'] = ' '.join(sample_flat).encode('utf-8')

            ofp_m[dev_sha[i][j]] = raw_and_mask

            ofp_samples_system.append(' '.join(ofp_system_output).encode('utf-8'))
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
        r = Rouge155(
            rouge_args='-e /home/kristjan/data1/softwares/rouge/ROUGE/RELEASE-1.5.5/data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m -b 75')
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


def save_test_results_rouge(args, z, x, y, e, sha, embedding_layer):
    tempfile.tempdir = '/scratch/'
    s_num = 0
    epoch = None

    filename_ = get_readable_file(args, epoch, test=True)
    filename_m = get_mask_file(args, epoch, test=True)

    index_to_e_map = get_entities(args)

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
        end_hl_idx = len(y[i][0])
        hl_step = end_hl_idx / args.n

        for j in xrange(len(z[i][0])):
            filename = rouge_fname + 'sum.' + str(s_num).zfill(6) + '.txt'
            ofp_for_rouge = open(filename, 'w+')
            ofp_system_output = []

            word_idx = 0
            sent_idx = 0

            for k in xrange(len(z[i])):
                if word_idx >= len(x[i][j][sent_idx]):
                    sent_idx += 1
                    word_idx = 0

                if sent_idx >= len(x[i][j]):
                    break

                word = x[i][j][sent_idx][word_idx].encode('utf-8')
                word_idx += 1

                if z[i][k][j] < 0.5:
                    continue

                ofp_for_rouge.write(word + ' ')
                ofp_system_output.append(word)

            raw_and_mask = dict()
            raw_and_mask['m'] = z[i][:, j].tolist()

            raw_and_mask['r'] = x[i][j][:]
            # raw_and_mask['y'] = []
            raw_and_mask['y'] = create_eval_questions(args, y[i], e[i], index_to_e_map, embedding_layer, start_hl_idx,
                                                      end_hl_idx, hl_step)

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


def create_readable_file_html(args, system_json):
    batched_file = args.batch_dir + args.source + 'test0'

    ifp_model = open(batched_file, 'rb')
    data_model = np.load(ifp_model)
    ifp_model.close()

    html_file = '../data/results/html/' + create_fname_identifier(args=args) + '.html'
    ofp = open(html_file, 'w+')
    ofp.write('<html><body>')

    dev_sha_np = data_model[4][1]
    dev_bm_np = data_model[3][1].T
    dev_cz_np = data_model[7][1].T

    for i in xrange(len(dev_sha_np)):
        cur_sha = dev_sha_np[i]
        cur_bm = dev_bm_np[i]
        cur_chunks = np.repeat(dev_cz_np[i], dev_cz_np[i])

        cur_r = system_json[cur_sha]['r']
        cur_samples = system_json[cur_sha]['m']

        assert len(cur_bm) == len(cur_samples), ('oh' + str(len(cur_bm)) + ' ' + str(len(cur_samples)))
        ofp.write('<p>')
        ofp.write(cur_sha)
        ofp.write(' :</br>')

        for w in xrange(len(cur_bm)):
            style_st = ''
            if len(cur_r) <= w:
                break
            if cur_samples[w] > 0 and cur_bm[w] > 0:  # sampled AND BM : green
                style_st += 'text-decoration: underline; font-weight: bold;'
            elif cur_samples[w] > 0:  # sampled not part of abstract : Red
                style_st += 'font-weight: bold;'
            elif cur_bm[w] > 0:  # not sample, part of abstract
                style_st += 'text-decoration: underline;'
            ofp.write('<span style="' + style_st + '">')
            print cur_r[w]
            print cur_chunks[w]
            ofp.write(cur_r[w].encode('utf-8') + '(' + str(cur_chunks[w]) + ') ')
            ofp.write('</span>')

        ofp.write('</p>')

    ofp.write('<html><body>')
    ofp.close()


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


def create_eval_questions(args, y, e, index_to_e_map, embedding_layer, start_hl_idx, end_hl_idx, step):
    num_samples = len(y)
    questions = []

    for i in range(start_hl_idx, end_hl_idx, step):
        single_y = dict()
        hl_str = ''
        entity = np.argmax(e[i])

        if entity < 0:
            continue

        for j in xrange(len(y)):
            w_idx = y[j][i]
            raw_w = embedding_layer.lst_words[w_idx]

            if raw_w == '<unk>' or raw_w == '<padding>':
                continue

            hl_str += raw_w
            hl_str += ' '

        single_y['hl'] = hl_str.rstrip()
        single_y['e'] = index_to_e_map[entity]

        questions.append(single_y)

    return questions


def get_entities(args):
    i_to_e = dict()
    print args.source

    filename_entities = 'entities_model.json' if args.full_test else "small_entities_model.json"
    filename_entities = args.source + '_' + str(args.vocab_size) + '_' + filename_entities

    ifp_entities = open('../data/' + filename_entities, 'rb')

    data = json.load(ifp_entities)['entities']

    for e in data:
        text = e[0]
        index = e[1][0]
        ner = e[1][1]

        i_to_e[index] = text

        ifp_entities.close()

    return i_to_e


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
