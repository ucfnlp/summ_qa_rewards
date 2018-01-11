import json

import data_args


def process_data(args):
    train, dev = prune_hl(args)
    write_model_ready(args, train, dev)


def prune_hl(args):
    train_x, train_y, train_e, train_ve = load_json(args, args.train)
    dev_x, dev_y, dev_e, dev_ve = load_json(args, args.dev)

    entity_map = get_entities(args)

    updated_train_y, updated_train_e, updated_train_x = prune_type(train_x, train_y, train_e, train_ve, entity_map)
    updated_dev_y, updated_dev_e, updated_dev_x = prune_type(dev_x, dev_y, dev_e, dev_ve, entity_map)

    return (updated_train_x, updated_train_y, updated_train_e, train_ve), \
           (updated_dev_x, updated_dev_y, updated_dev_e, dev_ve)


def write_model_ready(args, train, dev):
    filename_train = args.train_model if args.full_test else "small_" + args.train_model
    filename_train = str(args.vocab_size) + '_' + filename_train

    ofp_train = open(filename_train, 'w+')
    final_json_train = dict()

    final_json_train['x'] = train[0]
    final_json_train['y'] = train[1]
    final_json_train['e'] = train[2]
    final_json_train['valid_e'] = train[3]

    json.dump(final_json_train, ofp_train)
    ofp_train.close()

    filename_dev = args.dev_model if args.full_test else "small_" + args.dev_model
    filename_dev = str(args.vocab_size) + '_' + filename_dev

    ofp_dev = open(filename_dev, 'w+')
    final_json_dev = dict()

    final_json_dev['x'] = dev[0]
    final_json_dev['y'] = dev[1]
    final_json_dev['e'] = dev[2]
    final_json_dev['valid_e'] = dev[3]

    json.dump(final_json_dev, ofp_dev)
    ofp_dev.close()


def prune_type(x, y, e, ve, entity_map):
    length = len(y)
    updated_y = []
    updated_e = []
    updated_x = []

    restricted_types = generate_valid_entity_types(args)

    for i in xrange(length):

        valid_entities = set(ve[i])

        y_idx = 0
        total_entries = 0

        updated_y_ls = []
        updated_e_ls = []

        for highlight in e[i]:

            if total_entries >= args.n:
                break

            for perm in highlight:

                if total_entries >= args.n:
                    break

                if perm in valid_entities and keep_perm_type(perm, restricted_types, entity_map):
                    updated_y_ls.append(y[i][y_idx])
                    updated_e_ls.append(perm)
                    total_entries += 1

                y_idx += 1

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
        updated_x.append([w for sent in x[i] for w in sent])

    return updated_y, updated_e, updated_x


def load_json(args, type):
    f_name = type if args.full_test else "small_" + type
    f_name = str(args.vocab_size) + '_' + f_name

    ifp = open(f_name, 'rb')

    data = json.load(ifp)
    ifp.close()

    if 'test' in type:
        return data['x'], data['y']
    else:
        return data['x'], data['y'], data['e'], data['valid_e']


def get_entities(args):
    filename = 'entities.json' if args.full_test else "small_entities.json"
    filename = str(args.vocab_size) + '_' + filename

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


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
