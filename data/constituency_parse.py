from nltk import Tree

import sys
import os
import hashlib
import json
import time
import codecs
import numpy as np
import re

import parse_args

reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):

    raw_data = args.raw_data_cnn

    counter = 1
    start_time = time.time()

    highlights = []
    mapping = []
    output_file_count = 1

    ofp_filelist = open(args.parsed_output_loc + 'list_art.txt', 'w+')

    for subdir, dirs, files in os.walk(raw_data):
        for file_in in files:

            current_highlights = []
            current_article = []

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
                    current_article.append(line)

            if len(current_article) == 0:
                continue

            sha = str(sha)
            fname = 'articles/' + sha + '.txt'
            cm = (len(current_article), len(current_highlights), sha, fname)

            mapping.append(cm)
            highlights.append(current_highlights)

            ofp_articles = open(args.parsed_output_loc + fname, 'w+')
            ofp_filelist.write(fname +'\n')

            for i in xrange(cm[0]):
                ofp_articles.write(current_article[i] + '\n')
            ofp_articles.close()
            output_file_count += 1

    _len = len(highlights)
    ofp_filelist.close()

    total_hl = 0
    file_count = 1
    ofp_highlights = open(args.parsed_output_loc + 'highlights'+str(file_count)+'.txt', 'w+')
    ofp_lookup = open(args.parsed_output_loc + 'lookup.txt', 'w+')
    ofp_list_hl = open(args.parsed_output_loc + 'list_hl.txt', 'w+')

    for i in xrange(_len):
        num_s_in_art = mapping[i][0]
        num_s_in_hl = mapping[i][1]
        sha = mapping[i][2]

        ofp_lookup.write(str(num_s_in_art) + ' ' + str(num_s_in_hl) + ' ' + sha + '\n')

        for j in xrange(num_s_in_hl):
            ofp_highlights.write(highlights[i][j] + ' .\n')
            total_hl += 1

        if total_hl > 30000:
            ofp_highlights.close()

            ofp_list_hl.write('highlights'+str(file_count)+'.txt'+'\n')

            file_count += 1
            total_hl = 0

            ofp_highlights = open(args.parsed_output_loc + 'highlights' + str(file_count) + '.txt', 'w+')

    ofp_list_hl.write('highlights' + str(file_count) + '.txt' + '\n')

    ofp_list_hl.close()
    ofp_highlights.close()
    ofp_lookup.close()


def recombine_scnlp_data(args):
    ifp_lookup = open(args.parsed_output_loc + 'lookup.txt', 'r')

    sha_ls = []
    hl_ls = []

    print 'Input sha1, hl idx'
    for line in ifp_lookup:
        items = line.rstrip().split()

        sha_ls.append(items[2])
        hl_ls.append(int(items[1]))

    ifp_lookup.close()

    print 'Load SCNLP..'
    file_count = 1

    ifp_hl = open(args.parsed_output_loc + 'highlights'+str(file_count)+'.txt.json', 'rb')
    high_lights = json.load(ifp_hl)
    high_lights = high_lights['sentences']
    ifp_hl.close()

    print 'Combining..'

    hl_idx_end = 0
    counter = 0

    for i in xrange(len(hl_ls)):
        hl_idx_start = hl_idx_end
        hl_idx_end += hl_ls[i]

        ofp_combined = open(args.parsed_output_loc + 'processed/' + sha_ls[i] + '.json', 'w+')
        ifp_article = open(args.parsed_output_loc + 'scnlp/' + sha_ls[i] + '.txt.json', 'rb')

        document = json.load(ifp_article)['sentences']
        ifp_article.close()

        combined_json_out = dict()

        combined_json_out['highlights'] = high_lights[hl_idx_start:hl_idx_end]
        combined_json_out['document'] = document

        json.dump(combined_json_out, ofp_combined)
        ofp_combined.close()

        counter += 1

        if hl_idx_end > 30000:

            print 'Load SCNLP.. (more) ..'
            file_count += 1
            hl_idx_end = 0

            ifp_hl = open(args.parsed_output_loc + 'highlights' + str(file_count) + '.txt.json', 'rb')
            high_lights = json.load(ifp_hl)
            high_lights = high_lights['sentences']

            ifp_hl.close()



def tree2dict(tree):
    return {tree.label(): [tree2dict(t) if isinstance(t, Tree) else t.lower() for t in tree]}


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


def get_set(file_in, train_urls, dev_urls, test_urls):
    if file_in in train_urls:
        return 'train/'
    elif file_in in dev_urls:
        return 'dev/'
    elif file_in in test_urls:
        return 'test/'
    else:
        return ''


if __name__ == '__main__':
    args = parse_args.get_args()

    if args.process:
        process_data(args)
    else:
        recombine_scnlp_data(args)
