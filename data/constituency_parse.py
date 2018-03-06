from nltk import Tree

import sys
import os
import hashlib
import json
import time
import codecs

import parse_args


reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):

    raw_data = args.raw_data_cnn

    counter = 1
    start_time = time.time()

    articles = []
    highlights = []
    mapping = []

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

            cm = (len(current_article), len(current_highlights), sha)

            mapping.append(cm)
            articles.append(current_article)
            highlights.append(current_highlights)

    ofp_articles = open(args.parsed_output_loc + 'articles.txt', 'w+')
    ofp_highlights = open(args.parsed_output_loc + 'highlights.txt', 'w+')
    ofp_lookup = open(args.parsed_output_loc + 'lookup.txt', 'w+')

    _len = len(articles)

    total_hl = 0
    total_art_s = 0

    for i in xrange(_len):
        num_s_in_art = mapping[i][0]
        num_s_in_hl = mapping[i][1]
        sha = mapping[i][2]

        ofp_lookup.write(str(num_s_in_art) + ' ' + str(num_s_in_hl) + ' ' + sha + '\n')

        for j in xrange(num_s_in_art):
            ofp_articles.write(articles[i][j].lower() + '\n')
            total_art_s += 1

        for j in xrange(num_s_in_hl):
            ofp_highlights.write(highlights[i][j].lower() + '\n')
            total_hl += 1

    ofp_articles.close()
    ofp_highlights.close()
    ofp_lookup.close()

    print 'Total article sentences', total_art_s
    print 'Total hl', total_hl


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
    process_data(args)
