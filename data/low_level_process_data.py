import json
import os
import sys

import hashlib

import data_args


def process_data(args):

    train, dev, test = prune_hl(args)


def prune_hl(args):

    print 'Loading data from JSON'
    train_x, train_y, train_e, train_ve = load_json(args, args.train)
    dev_x, dev_y, dev_e, dev_ve = load_json(args, args.dev)
    test_x, test_y = load_json(args, args.test)

    updated_train_y, updated_train_e = prune_type(train_y, train_e,train_ve)
    updated_dev_y, updated_dev_e = prune_type(dev_y, dev_e, dev_ve)

    return (train_x, updated_train_y, updated_train_e, train_ve), \
           (dev_x, updated_dev_y, updated_dev_e, dev_ve), \
           (test_x, test_y)


def prune_type(y, e, ve):
    len_train = len(y)
    updated_y = []
    updated_e = []

    for i in xrange(len_train):

        valid_entities = set(ve[i])
        y_idx = 0
        updated_y_ls = []
        updated_e_ls = []

        for highlight in e[i]:
            updated_hl = []

            for perm in highlight:

                if perm in valid_entities:
                    updated_y_ls.append(y[y_idx])
                    updated_hl.append(perm)
                y_idx += 1

            updated_e_ls.append(updated_hl)

        updated_y.append(updated_y_ls)
        updated_e.append(updated_e_ls)

    return updated_y, updated_e


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


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)