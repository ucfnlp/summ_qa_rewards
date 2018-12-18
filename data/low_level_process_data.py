import json
import numpy as np

import data_args


def process_data(args):
    train, dev, test = prune_hl(args)
    write_model_ready(args, train, dev, test)


def prune_hl(args):
    train_x, train_y, train_e, train_ve, train_cly, train_sha, train_m, train_ch = load_json(args, args.train)
    dev_x, dev_y, dev_e, dev_ve, dev_cly, dev_rx, dev_sha, dev_m, dev_ch = load_json(args, args.dev)
    test_x, test_y, test_e, test_cy, test_rx, test_m, test_sha, test_ch = load_json(args, args.test)

    entity_map = get_entities(args)
    used_e = set()

    usable_e = determine_usable_entities(args, train_e, dev_e, test_e, train_y, dev_y, test_y, entity_map, args.ent_cutoff)
    chunk_freq = [0]*5

    updated_train_y, updated_train_e, updated_train_x, updated_train_ve, updated_train_cly, updated_train_sha, updated_train_ma, updated_train_ch, sent_cut_train = prune_type(
        args, train_x, train_y, train_e, train_ve, train_cly, None, train_m, train_sha, train_ch, chunk_freq, used_e, usable_e)
    updated_dev_y, updated_dev_e, updated_dev_x, updated_dev_ve, updated_dev_cly, updated_dev_rx, updated_dev_sha, updated_dev_ma, updated_dev_ch, sent_cut_dev = prune_type(
        args, dev_x, dev_y, dev_e, dev_ve, dev_cly, dev_rx, dev_m, dev_sha, dev_ch, chunk_freq, used_e, usable_e)
    updated_test_y, updated_test_e, updated_test_x, updated_test_cly, updated_test_rx, updated_test_sha, updated_test_ma, updated_test_ch, sent_cut_test = prune_type(
        args, test_x, test_y, test_e, None, test_cy, test_rx, test_m, test_sha, test_ch, chunk_freq, used_e, usable_e)

    print 'used/total entities = ', len(used_e) / float(len(entity_map))
    if not args.word_level_c:
        print_chunk_info(chunk_freq)

    print 'pruning entities'
    e_map_new = re_map(used_e)

    re_map_entities(updated_train_e, e_map_new)
    re_map_entities(updated_dev_e, e_map_new)
    re_map_entities(updated_test_e, e_map_new)
    # determine_usable_entities(args, updated_train_e, updated_dev_e, updated_test_e, updated_train_y, updated_dev_y,
    #                           updated_test_y, e_map_new, args.ent_cutoff)
    save_updated_e(args, e_map_new, entity_map)

    return (updated_train_x, updated_train_y, updated_train_e, updated_train_ve, updated_train_cly, updated_train_ma, updated_train_sha, updated_train_ch, sent_cut_train), \
           (updated_dev_x, updated_dev_y, updated_dev_e, updated_dev_ve, updated_dev_cly, updated_dev_rx, updated_dev_ma, updated_dev_sha, updated_dev_ch, sent_cut_dev), \
            (updated_test_x, updated_test_y, updated_test_e, updated_test_cly, updated_test_rx, updated_test_sha, updated_test_ma, updated_test_ch, sent_cut_test)


def write_model_ready(args, train, dev, test):
    filename_train = args.train_model if args.full_test else "small_" + args.train_model
    filename_train = args.source + '_' + str(args.vocab_size) + '_' + filename_train

    ofp_train = open(filename_train, 'w+')
    final_json_train = dict()

    final_json_train['x'] = train[0]
    final_json_train['y'] = train[1]
    final_json_train['e'] = train[2]
    final_json_train['valid_e'] = train[3]
    final_json_train['clean_y'] = train[4]
    final_json_train['mask'] = train[5]
    final_json_train['sha'] = train[6]
    final_json_train['chunk'] = train[7]
    final_json_train['scut'] = train[8]

    json.dump(final_json_train, ofp_train)
    ofp_train.close()

    filename_dev = args.dev_model if args.full_test else "small_" + args.dev_model
    filename_dev = args.source + '_' + str(args.vocab_size) + '_' + filename_dev

    ofp_dev = open(filename_dev, 'w+')
    final_json_dev = dict()

    final_json_dev['x'] = dev[0]
    final_json_dev['y'] = dev[1]
    final_json_dev['e'] = dev[2]
    final_json_dev['valid_e'] = dev[3]
    final_json_dev['clean_y'] = dev[4]
    final_json_dev['raw_x'] = dev[5]
    final_json_dev['mask'] = dev[6]
    final_json_dev['sha'] = dev[7]
    final_json_dev['chunk'] = dev[8]
    final_json_dev['scut'] = dev[9]

    json.dump(final_json_dev, ofp_dev)
    ofp_dev.close()

    filename_test = args.test_model if args.full_test else "small_" + args.test_model
    filename_test = args.source + '_' + str(args.vocab_size) + '_' + filename_test

    ofp_test = open(filename_test, 'w+')
    final_json_test = dict()

    final_json_test['x'] = test[0]
    final_json_test['y'] = test[1]
    final_json_test['e'] = test[2]
    final_json_test['clean_y'] = test[3]
    final_json_test['raw_x'] = test[4]
    final_json_test['sha'] = test[5]
    final_json_test['mask'] = test[6]
    final_json_test['chunk'] = test[7]
    final_json_test['scut'] = test[8]

    json.dump(final_json_test, ofp_test)
    ofp_test.close()


def prune_type(args, x, y, e, ve, cy, rx, ma, sha, ch, chunk_freq, used_e, usable_e):
    length = len(y)
    updated_y = []
    updated_cy = []
    updated_e = []
    updated_ve = []
    updated_x = []
    updated_raw_x = []
    updated_ma = []
    updated_sha = []
    updated_ch = []
    sentence_cutoffs = []

    invalid_articles = 0

    for i in xrange(length):

        y_idx = 0
        total_entries = 0

        updated_y_ls = []
        updated_e_ls = []

        no_usable_hl = True

        for highlight in e[i]:

            if total_entries >= args.n:
                break

            num_perms = get_perms(highlight)

            if args.use_root and highlight[0][0] in usable_e:

                updated_y_ls.append(y[i][y_idx])
                updated_e_ls.append(highlight[0][0])

                used_e.add(highlight[0][0])

                total_entries += 1
                no_usable_hl = False

            elif args.use_obj_subj:
                index_offset = 1

                flat_hl = [item for item in highlight[1]]

                for j in xrange(len(flat_hl)):

                    if flat_hl[j] not in usable_e:
                        continue

                    updated_y_ls.append(y[i][y_idx + j + index_offset])
                    updated_e_ls.append(flat_hl[j])

                    used_e.add(flat_hl[j])
                    no_usable_hl = False

                total_entries += 1
            elif args.use_ner:
                index_offset = 1 + len([item for item in highlight[1]])

                flat_hl = [item for item in highlight[2]]

                for j in xrange(len(flat_hl)):

                    if flat_hl[j] not in usable_e:
                        continue

                    updated_y_ls.append(y[i][y_idx + j + index_offset])
                    updated_e_ls.append(flat_hl[j])

                    used_e.add(flat_hl[j])
                    no_usable_hl = False

                total_entries += 1
            else: # USE all QA content
                index_offset = 0

                flat_hl = [item for group in highlight for item in group]

                for j in xrange(len(flat_hl)):

                    if flat_hl[j] not in usable_e:
                        continue

                    updated_y_ls.append(y[i][y_idx + j + index_offset])
                    updated_e_ls.append(flat_hl[j])

                    used_e.add(flat_hl[j])
                    no_usable_hl = False

                total_entries += 1

            y_idx += num_perms

        if no_usable_hl:
            invalid_articles += 1

        if total_entries == 0:
            invalid_articles += 1
            continue

        updated_e.append(updated_e_ls[:args.n])
        updated_y.append(updated_y_ls[:args.n])

        updated_cy.append(cy[i])
        updated_sha.append(sha[i])

        updated_x.append([w for sent in x[i] for w in sent])
        updated_ma.append([w for sent in ma[i] for w in sent])
        sentence_cutoffs.append([len(sent) for sent in x[i]])

        if args.word_level_c:
            updated_ch.append([1 for sent in ch[i] for _ in sent])
        else:
            updated_chunks = []

            for sent in ch[i]:
                for w in sent:
                    updated_chunks.append(w)
                    chunk_freq[w - 1] += 1

            updated_ch.append(updated_chunks)

        if ve is not None:
            updated_ve.append(ve[i])
        if rx is not None and ve is not None:
            updated_raw_x.append([w for sent in rx[i] for w in sent])
        elif rx is not None:
            updated_raw_x.append(rx[i])

    print invalid_articles, "Invalid Articles"
    print length, "of Articles"

    if ve is None:
        return updated_y, updated_e, updated_x, updated_cy, updated_raw_x, \
               updated_sha, updated_ma, updated_ch, sentence_cutoffs
    if rx is None:
        return updated_y, updated_e, updated_x, updated_ve, updated_cy, \
               updated_sha, updated_ma, updated_ch, sentence_cutoffs
    else:
        return updated_y, updated_e, updated_x, updated_ve, updated_cy, \
               updated_raw_x, updated_sha, updated_ma, updated_ch, sentence_cutoffs


def determine_usable_entities(args, train_e, dev_e, test_e, train_y, dev_y, test_y, entity_map, cutoff=5):
    e = train_e + dev_e + test_e
    y = train_y + dev_y + test_y

    length = len(y)
    used_e = dict()
    new_map = dict()

    empty_articles = 0

    for i in xrange(length):

        y_idx = 0
        total_entries = 0

        empty_article = True

        for highlight in e[i]:

            num_perms = get_perms(highlight)

            if args.use_root:
                if highlight[0][0] in used_e:
                    used_e[highlight[0][0]] = used_e[highlight[0][0]] + 1
                else:
                    used_e[highlight[0][0]] = 1

                total_entries += 1
            elif args.use_obj_subj:

                flat_hl = [item for item in highlight[1]]

                for j in xrange(len(flat_hl)):
                    if flat_hl[j] in used_e:
                        used_e[flat_hl[j]] = used_e[flat_hl[j]] + 1
                    else:
                        used_e[flat_hl[j]] = 1

                total_entries += 1
            elif args.use_ner:
                flat_hl = [item for item in highlight[2]]

                for j in xrange(len(flat_hl)):
                    if flat_hl[j] in used_e:
                        used_e[flat_hl[j]] = used_e[flat_hl[j]] + 1
                    else:
                        used_e[flat_hl[j]] = 1

                total_entries += 1
            else:
                flat_hl = [item for group in highlight for item in group]

                if len(flat_hl) > 0:
                    empty_article = False

                for j in xrange(len(flat_hl)):

                    if flat_hl[j] in used_e:
                        used_e[flat_hl[j]] = used_e[flat_hl[j]] + 1
                    else:
                        used_e[flat_hl[j]] = 1

                total_entries += 1

            y_idx += num_perms
        if empty_article:
            empty_articles += 1

    if not args.use_root:
        print 'Empty articles :', empty_articles

    print 'Original Total Entities :', len(used_e)

    total_count = 0

    for k, v in used_e.iteritems():
        total_count += v
        if v >= cutoff:
            new_map[k] = v

    print 'New Total Entites :', len(new_map)
    print 'Top 20 frequencies :'
    i = 0
    for key, value in sorted(used_e.iteritems(), key=lambda (k, v): (v, k), reverse=True):
        if i == 20:
            break
        print "  %s : %s (%s perc)" % (entity_map[key][0], value, value / float(total_count) * 100.0)
        i += 1

    return new_map


def re_map_entities(e, new_map):
    length = len(e)

    for i in xrange(length):
        for j in xrange(len(e[i])):
            e[i][j] = new_map[e[i][j]]


def load_json(args, type):
    f_name = type if args.full_test else "small_" + type
    f_name = args.source + '_' + str(args.vocab_size) + '_' + f_name

    ifp = open(f_name, 'rb')

    data = json.load(ifp)
    ifp.close()

    if 'test' in type:
        return data['x'], data['y'], data['e'], data['clean_y'], data['raw_x'], data['mask'], data['sha'], data['chunk']
    elif 'dev' in type:
        return data['x'], data['y'], data['e'], data['valid_e'], data['clean_y'], data['raw_x'], data['sha'], data['mask'], data['chunk']
    else:
        return data['x'], data['y'], data['e'], data['valid_e'], data['clean_y'], data['sha'], data['mask'], data['chunk']


def get_entities(args):
    filename = 'entities.json' if args.full_test else "small_entities.json"
    filename = args.source + '_' + str(args.vocab_size) + '_' + filename

    ifp = open(filename, 'rb')
    data = json.load(ifp)
    ifp.close()

    inv_map = invert_ents(data['entities'])
    return inv_map


def generate_valid_entity_types(args):
    e_ls = []

    if args.use_all:
        return set(['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC', 'ROOT'])

    if args.use_person:
        e_ls.append('PERSON')
    if args.use_org:
        e_ls.append('ORGANIZATION')
    if args.use_misc:
        e_ls.append('MISC')
    if args.use_root:
        e_ls.append('ROOT')
    if args.use_location:
        e_ls.append('LOCATION')

    return set(e_ls)


def re_map(used_e):
    new_e = dict()
    counter = 0

    for item in used_e:
        new_e[item] = counter
        counter += 1
    print len(new_e), 'TOTAL NEW E'
    return new_e


def save_updated_e(args, e_map_new, entity_map):
    e_ls = []

    for old_idx, new_idx in e_map_new.iteritems():
        word = entity_map[old_idx][0]
        ner = entity_map[old_idx][1]

        e_for_json = [word, [new_idx, ner]]
        e_ls.append(e_for_json)

    filename_entities = 'entities_model.json' if args.full_test else "small_entities_model.json"
    filename_entities = args.source + '_' + str(args.vocab_size) + '_' + filename_entities

    ofp_entities = open(filename_entities, 'w+')
    final_json_entities = dict()
    final_json_entities['entities'] = e_ls

    json.dump(final_json_entities, ofp_entities)
    ofp_entities.close()


def invert_ents(entities):
    inv_map = dict()

    for entity in entities:
        name = entity[0]
        index = entity[1][0]
        type = entity[1][1]

        inv_map[index] = (name, type)

    return inv_map


def keep_perm_type(perm, restricted_types, entities):
    corresponding_e = entities[perm]

    return corresponding_e[1] in restricted_types


def is_root(perm, entities):
    corresponding_e = entities[perm]

    return corresponding_e[1] == 'ROOT'


def get_perms(highlight):
    t = 0
    for item in highlight:
        for _ in item:
            t += 1

    return t


def print_chunk_info(chunk_freq):
    print 'Chunk frequency :'

    total_ch = float(np.sum(chunk_freq))

    for i in xrange(len(chunk_freq)):
        print '  total : ' + str(chunk_freq[i]) + ', size ' + str(i+1) + ': ' + str(chunk_freq[i] / total_ch)


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
