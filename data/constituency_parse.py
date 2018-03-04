from nltk.parse import stanford
from nltk import Tree

import sys
import os
import hashlib
import json

import parse_args


reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    os.environ['CLASSPATH'] = args.cp
    os.environ['STANFORD_PARSER'] = args.sp
    os.environ['STANFORD_MODEL'] = args.sm

    raw_data = args.raw_data_cnn
    train_urls, dev_urls, test_urls = get_url_sets(args)

    parser = stanford.StanfordParser()

    counter = 0

    for subdir, dirs, files in os.walk(raw_data):
        for file_in in files:

            current_article = []
            current_highlights = []

            if file_in.startswith('.'):
                continue

            if counter % 10 == 0:
                print 'Processing', counter + 1
            counter += 1

            sha = file_in.split('.')[0]
            file_in = open(subdir + file_in, 'r')
            incoming_hl = False

            for line in file_in:
                if len(line.strip()) == 0:
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

            catg = get_set(sha, train_urls, dev_urls, test_urls)
            sha = str(sha)

            ofp = open(args.parsed_output_loc + catg + sha + '.json', 'w+')
            output_json = dict()

            p_ls = [current_article, current_highlights]

            for i in xrange(len(p_ls)):
                sent_ls = []
                sentences = parser.raw_parse_sents(p_ls[i])

                for line in sentences:
                    for sentence in line:
                        sent_ls.append(tree2dict(sentence))

                if i == 0:
                    output_json['article'] = sent_ls
                else:
                    output_json['highlights'] = sent_ls

            json.dump(output_json,ofp)
            ofp.close()


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
