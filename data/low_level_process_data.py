import json
import numpy as np

import data_args


def process_data(args):
    train, dev = prune_hl(args)
    write_model_ready(args, train, dev)


def prune_hl(args):
    train_x, train_y, train_e, train_ve, train_cly, train_sha = load_json(args, args.train)
    dev_x, dev_y, dev_e, dev_ve, dev_cly, dev_rx, dev_sha = load_json(args, args.dev)

    entity_map = get_entities(args)

    updated_train_y, updated_train_e, updated_train_x, updated_train_ve, updated_train_cly, updated_train_sha = prune_type(
        train_x, train_y, train_e, train_ve, train_cly, None, train_sha, entity_map)
    updated_dev_y, updated_dev_e, updated_dev_x, updated_dev_ve, updated_dev_cly, updated_dev_rx, updated_dev_sha = prune_type(
        dev_x, dev_y, dev_e, dev_ve, dev_cly, dev_rx, dev_sha, entity_map)

    return (
           updated_train_x, updated_train_y, updated_train_e, updated_train_ve, updated_train_cly, updated_train_sha), (
           updated_dev_x, updated_dev_y, updated_dev_e, updated_dev_ve, updated_dev_cly, updated_dev_rx,
           updated_dev_sha)


def write_model_ready(args, train, dev):
    filename_train = args.train_model if args.full_test else "small_" + args.train_model
    filename_train = args.source + '_' + str(args.vocab_size) + '_' + filename_train

    ofp_train = open(filename_train, 'w+')
    final_json_train = dict()

    final_json_train['x'] = train[0]
    final_json_train['y'] = train[1]
    final_json_train['e'] = train[2]
    final_json_train['valid_e'] = train[3]
    final_json_train['clean_y'] = train[4]
    final_json_train['sha'] = train[5]

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
    final_json_dev['sha'] = dev[6]

    json.dump(final_json_dev, ofp_dev)
    ofp_dev.close()


def prune_type(x, y, e, ve, cy, rx, sha, entity_map):
    length = len(y)
    updated_y = []
    updated_cy = []
    updated_e = []
    updated_ve = []
    updated_x = []
    updated_raw_x = []
    updated_sha = []

    restricted_types = generate_valid_entity_types(args)
    invalid_articles = 0

    for i in xrange(length):

        valid_entities = set(ve[i])

        y_idx = 0
        total_entries = 0

        updated_y_ls = []
        updated_e_ls = []

        for highlight in e[i]:

            if total_entries >= args.n:
                break

            num_perms = len(highlight)

            if args.use_root or num_perms == 1:
                updated_y_ls.append(y[i][y_idx])
                updated_e_ls.append(highlight[0])

                total_entries += 1
            else:
                if num_perms == 2:
                    rand_e_idx = 1
                else:
                    rand_e_idx = np.random.randint(1, num_perms - 1)

                updated_y_ls.append(y[i][y_idx + rand_e_idx])
                updated_e_ls.append(highlight[rand_e_idx])

                total_entries += 1

            y_idx += num_perms

        if total_entries == 0:
            invalid_articles += 1
            continue

        if total_entries < args.n and total_entries != 0:

            original_y = updated_y_ls[:]
            original_e = updated_e_ls[:]

            added_entries = len(original_y)

            while total_entries < args.n:
                updated_e_ls.extend(original_e[:])
                updated_y_ls.extend(original_y[:])
                total_entries += added_entries

        updated_y.append(updated_y_ls[:args.n])
        updated_e.append(updated_e_ls[:args.n])
        updated_ve.append(ve[i])
        updated_cy.append(cy[i])
        updated_sha.append(sha[i])
        updated_x.append([w for sent in x[i] for w in sent])

        if rx is not None:
            updated_raw_x.append([w for sent in rx[i] for w in sent])

    print invalid_articles, "Invalid Articles"
    print length, "of Articles"

    if rx is None:
        return updated_y, updated_e, updated_x, updated_ve, updated_cy, updated_sha
    else:
        return updated_y, updated_e, updated_x, updated_ve, updated_cy, updated_raw_x, updated_sha


def load_json(args, type):
    f_name = type if args.full_test else "small_" + type
    f_name = args.source + '_' + str(args.vocab_size) + '_' + f_name

    ifp = open(f_name, 'rb')

    data = json.load(ifp)
    ifp.close()

    if 'test' in type:
        return data['x'], data['y'], data['clean_y'], data['raw_x']
    elif 'dev' in type:
        return data['x'], data['y'], data['e'], data['valid_e'], data['clean_y'], data['raw_x'], data['sha']
    else:
        return data['x'], data['y'], data['e'], data['valid_e'], data['clean_y'], data['sha']


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


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
