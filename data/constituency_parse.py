from nltk import ParentedTree

import sys
import os
import hashlib
import json
import time
import codecs
from nltk.tokenize import RegexpTokenizer

import parse_args

reload(sys)
sys.setdefaultencoding('utf8')


def get_set(file_in, train_urls, dev_urls, test_urls):
    if file_in in train_urls:
        return 1
    elif file_in in dev_urls:
        return 2
    elif file_in in test_urls:
        return 3
    else:
        return -1


def get_url_ls():
    sha1 = hashlib.sha1

    train_urls = set()
    dev_urls = set()
    test_urls = set()

    train_ofp = open('lists/all_train.txt', 'r')
    dev_ofp = open('lists/all_val.txt', 'r')
    test_ofp = open('lists/all_test.txt', 'r')

    for line in train_ofp:
        train_urls.add(sha1(line.rstrip()).hexdigest())

    for line in dev_ofp:
        dev_urls.add(sha1(line.rstrip()).hexdigest())

    for line in test_ofp:
        test_urls.add(sha1(line.rstrip()).hexdigest())

    train_ofp.close()
    dev_ofp.close()
    test_ofp.close()

    return train_urls, dev_urls, test_urls


def sha_ls(args):
    ofp_train = open('all_' + args.source + '_sha_train.out', 'w+')
    ofp_dev = open('all_' + args.source + '_sha_dev.out', 'w+')
    ofp_test = open('all_' + args.source + '_sha_test.out', 'w+')

    train_urls, dev_urls, test_urls = get_url_ls()

    for subdir, dirs, files in os.walk(args.raw_data):
        for file_in in files:

            sha = file_in.split('.')[0]
            catg = get_set(sha, train_urls, dev_urls, test_urls)

            if catg == 1:
                ofp_train.write(sha + '\n')
            elif catg == 2:
                ofp_dev.write(sha + '\n')
            else:
                ofp_test.write(sha + '\n')

    ofp_train.close()
    ofp_dev.close()
    ofp_test.close()


def process_data(args):
    raw_data = args.raw_data

    counter = 1
    start_time = time.time()

    highlights = []
    mapping = []
    output_file_count = 1

    ofp_filelist_art = open('list_art.txt', 'w+')
    ofp_filelist_hl = open('list_hl.txt', 'w+')

    intermediate_dir_art = args.parsed_output_loc + '/articles/'
    intermediate_dir_hl = args.parsed_output_loc + '/highlights/'

    if not os.path.exists(intermediate_dir_art):
        os.makedirs(intermediate_dir_art)

    if not os.path.exists(intermediate_dir_hl):
        os.makedirs(intermediate_dir_hl)

    for subdir, dirs, files in os.walk(raw_data):
        for file_in in files:

            current_highlights = []
            current_articel_sent = []

            if file_in.startswith('.'):
                continue

            if counter % 1000 == 0:
                print 'Processing', counter
                print (time.time() - start_time), 'seconds'
                start_time = time.time()

            counter += 1

            sha = file_in.split('.')[0]
            file_in = codecs.open(subdir + file_in, 'r', 'utf-8-sig')
            incoming_hl = False

            for line in file_in:
                line = line.rstrip()

                if len(line) == 0:
                    continue

                if '@highlight' in line:
                    incoming_hl = True
                    continue

                if incoming_hl:
                    current_highlights.append(line)
                    incoming_hl = False
                else:
                    current_articel_sent.append(line)

            if len(current_highlights) == 0:
                continue

            sha = str(sha)

            fname_hl = intermediate_dir_hl + sha + '.txt'
            fname_art = intermediate_dir_art + sha + '.txt'

            cm = (len(current_articel_sent), len(current_highlights), sha)

            mapping.append(cm)
            highlights.append(current_highlights)

            ofp_art = open(fname_art, 'w+')
            ofp_hl = open(fname_hl, 'w+')

            ofp_filelist_art.write(fname_art + '\n')
            ofp_filelist_hl.write(fname_hl + '\n')

            for i in xrange(cm[0]):
                ofp_art.write(current_articel_sent[i] + '\n')

            for i in xrange(cm[1]):
                ofp_hl.write(current_highlights[i] + ' .\n')

            ofp_art.close()
            ofp_hl.close()
            output_file_count += 1

    ofp_filelist_art.close()
    ofp_filelist_hl.close()


def recombine_scnlp_data(args):
    combined_dir = args.parsed_output_loc + '/processed/'

    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    sha_ls = get_sha()

    counter = 1

    for sha in sha_ls:

        if counter % 1000 == 0:
            print 'at', counter

        ifp_article = open(args.parsed_output_loc + '/articles_scnlp/' + sha + '.txt.json', 'rb')
        ifp_hl = open(args.parsed_output_loc + '/highlights_scnlp/' + sha + '.txt.json', 'rb')

        ofp_combined = open(combined_dir + sha + '.json', 'w+')

        document = json.load(ifp_hl)['sentences']
        cur_hl = json.load(ifp_article)['sentences']

        ifp_article.close()
        ifp_hl.close()

        combined_json_out = dict()

        combined_json_out['highlights'] = cur_hl
        combined_json_out['document'] = document

        json.dump(combined_json_out, ofp_combined)
        ofp_combined.close()

        counter += 1


def tree2dict(tree):
    return {tree.label(): [tree2dict(t) if isinstance(t, ParentedTree) else t.lower() for t in tree]}


def get_url_sets(args):

    sha1 = hashlib.sha1

    train_urls = set()
    dev_urls = set()
    test_urls = set()

    train_ofp = open(args.train_urls, 'r')
    dev_ofp = open(args.dev_urls, 'r')
    test_ofp = open(args.test_urls, 'r')

    for line in train_ofp:
        train_urls.add(sha1(line.rstrip()).hexdigest())

    for line in dev_ofp:
        dev_urls.add(sha1(line.rstrip()).hexdigest())

    for line in test_ofp:
        test_urls.add(sha1(line.rstrip()).hexdigest())

    train_ofp.close()
    dev_ofp.close()
    test_ofp.close()

    return train_urls, dev_urls, test_urls


def get_sha():
    ifp = open('list_hl.txt', 'r')

    ls = []

    for item in ifp:
        ls.append(item.rstrip().split('.')[0].split('/')[-1])

    return ls


if __name__ == '__main__':
    args = parse_args.get_args()

    if args.process:
        process_data(args)
    else:
        recombine_scnlp_data(args)
