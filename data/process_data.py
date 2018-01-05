import json
import os
import sys

import hashlib

import data_args


reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    train, dev, test, unique_w = split_data(args)
    # w2v_model = utils.create_w2v_model(args, [item for sublist in articles[512:] for item in sublist])
    word_counts = [50000, 20000, 10000]

    for count in word_counts:
        vocab = create_vocab_map(unique_w, count)
        machine_ready(args, train, dev, test, vocab)


def split_data(args):
    small_size_counter = 0

    unique_words = dict() # word : count

    highlights_train = []
    articles_train = []

    highlights_dev = []
    articles_dev = []

    highlights_test = []
    articles_test = []

    train_urls, dev_urls, test_urls = get_url_sets(args)

    for subdir, dirs, files in os.walk(args.raw_data):
        for file_in in files:

            current_article = []
            current_highlights = []

            if file_in.startswith('.'):
                continue

            sha = file_in.split('.')[0]
            file_in = open(subdir + '/' + file_in, 'r')
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

            current_article, current_highlights = tokenize(args, current_article, current_highlights, unique_words)

            if len(current_article) == 0:
                continue

            catg = get_set(sha, train_urls, dev_urls, test_urls)

            if catg < 0:
                print 'Problem with : ' + str(sha)
                continue

            if catg == 1: #TRAIN
                highlights_train.append(current_highlights)
                articles_train.append(current_article)
            elif catg == 2: #DEV
                highlights_dev.append(current_highlights)
                articles_dev.append(current_article)
            else:#TEST
                highlights_test.append(current_highlights)
                articles_test.append(current_article)

            small_size_counter += 1

            if not args.full_test and small_size_counter >= args.small_limit:
                return (highlights_train, articles_train), (highlights_dev, articles_dev), (
                highlights_test, articles_test), unique_words

    return (highlights_train, articles_train), (highlights_dev, articles_dev), (highlights_test, articles_test), unique_words


def pad_seqs(args, inp, vocab):
    input_seqs = []

    for i in xrange(len(inp)):

        single_inp = []

        for j in xrange(args.max_sentences):
            for k in xrange(args.sentence_length):

                if j < len(inp[i]) and k < len(inp[i][j]):
                    if inp[i][j][k] in vocab:
                        single_inp.append(vocab[inp[i][j][k]])
                    else:
                        single_inp.append(vocab['<unk>'])
                else:
                    single_inp.append(vocab['<padding>'])

        input_seqs.append(single_inp)

    return input_seqs


def machine_ready(args, train, dev, test, vocab):
    seqs_train_articles = pad_seqs(args, train[0], vocab)
    seqs_train_hl = pad_seqs(args, train[1], vocab)

    seqs_dev_articles = pad_seqs(args, dev[0], vocab)
    seqs_dev_hl = pad_seqs(args, dev[1], vocab)

    seqs_test_articles = pad_seqs(args, test[0], vocab)
    seqs_test_hl = pad_seqs(args, test[1], vocab)

    filename_train = args.train if args.full_test else "small_" + args.train
    filename_dev = args.dev if args.full_test else "small_" + args.dev
    filename_test = args.dev if args.full_test else "small_" + args.test

    ofp_train = open(filename_train, 'w+')
    ofp_dev = open(filename_dev, 'w+')
    ofp_test = open(filename_test, 'w+')

    final_json_train = dict()
    final_json_dev = dict()
    final_json_test = dict()

    final_json_train['x'] = seqs_train_articles
    final_json_train['y'] = seqs_train_hl

    final_json_dev['x'] = seqs_dev_articles
    final_json_dev['y'] = seqs_dev_hl

    final_json_test['x'] = seqs_test_articles
    final_json_test['y'] = seqs_test_hl

    json.dump(final_json_train, ofp_train)
    json.dump(final_json_dev, ofp_dev)
    json.dump(final_json_test, ofp_test)

    ofp_train.close()
    ofp_dev.close()
    ofp_test.close()


def tokenize(args, current_article, current_highlights, unique_w):
    article = []
    highlights = []

    for item in current_article:
        sentence = str(item.encode('utf-8'))

        words = sentence.rstrip().split(' ')
        s = []
        for w in words:
            w = w.lower()

            if w in unique_w:
                unique_w[w] += 1
            else:
                unique_w[w] = 1
            s.append(w)

        article.append(s)

    for item in current_highlights:
        sentence = str(item.encode('utf-8'))

        words = sentence.rstrip().split(' ')
        s = []
        for w in words:
            w = w.lower()

            if w in unique_w:
                unique_w[w] += 1
            else:
                unique_w[w] = 1
            s.append(w)

        highlights.append(s)

    return article, highlights


def create_vocab_map(unique_w, count):
    ofp = open('vocab_' + str(count) + '.txt', 'w+')
    vocab_map = dict()
    index = 2

    inv_map = dict()

    for k, v in unique_w.iteritems():
        if v in inv_map:
            inv_map[v].append(k)
        else:
            inv_map[v] = [k]

    ofp.write('<padding>\n<unk>\n')

    vocab_map['<padding>'] = 0
    vocab_map['<unk>'] = 1

    for c in sorted(inv_map.iterkeys(), reverse=True):
        words = inv_map[c]

        for w in words:

            ofp.write(w + '\n')
            vocab_map[w] = index
            index += 1

            if index > count:
                ofp.close()
                return vocab_map

    ofp.close()
    return vocab_map


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
        return 1
    elif file_in in dev_urls:
        return 2
    elif file_in in test_urls:
        return 3
    else:
        return -1

if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
