import json
import os
import sys

import hashlib
from nltk.tag import StanfordNERTagger

import data_args


reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    train, dev, test, unique_w = split_data(args)
    word_counts = [50000, 20000, 10000]

    for count in word_counts:
        print 'Building dataset for vocab size : ' + str(count)
        vocab = create_vocab_map(unique_w, count)
        machine_ready(args, train, dev, test, vocab, count)


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


def seqs(args, inp, vocab, tagger, test=False):
    input_seqs = []
    input_hl_seqs = []
    entity_set = dict()
    entity_counter = 1

    tag_ls = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC']

    total_samples = len(inp[0])
    print total_samples, 'total samples'

    for sample in xrange(total_samples):
        if sample % (total_samples / 10) == 0:
            print '..', sample

        single_inp_hl = []
        single_inp_art = []

        highlight = inp[0][sample]
        article = inp[1][sample]

        entities_used = []

        for sentence in highlight:
            no_entity = True

            s = []
            tagged_text = tagger.tag(sentence[:args.inp_len_hl])

            for i in xrange(len(tagged_text)):
                tag_token = tagged_text[i]

                if tag_token[1] in tag_ls and no_entity: # word is an entity, and highlight still needs entity

                    if tag_token[0] in entity_set:
                        s.append(entity_set[tag_token[0]])
                    else:
                        placeholder = '@entity' + str(entity_counter)
                        s.append(placeholder)
                        entity_set[tag_token[0]] = placeholder
                        entity_counter += 1

                    entities_used.append(tag_token[0])

                    no_entity = False
                else:
                    index = vocab[sentence[i]] if sentence[i] in vocab else 1
                    s.append(index)

            single_inp_hl.append(s)

        input_hl_seqs.append(single_inp_hl)

        for sentence in article:

            s = []
            max_len = args.inp_len if args.inp_len < len(sentence) else len(sentence)

            for i in xrange(max_len):
                word = sentence[i]
                if word in entities_used:
                    s.append(entity_set[word])
                else:
                    index = vocab[word] if word in vocab else 1
                    s.append(index)

            single_inp_art.append(s)

        input_seqs.append(single_inp_art)

    return input_seqs, input_hl_seqs, entity_set.items()


def machine_ready(args, train, dev, test, vocab, count):
    st = StanfordNERTagger(args.stgz, args.stjar, encoding='utf-8')

    print 'Train data NER and indexing'
    seqs_train_articles, seqs_train_hl, train_items = seqs(args, train, vocab, st)

    filename_train = args.train if args.full_test else "small_" + args.train
    filename_train = str(count) + '_' + filename_train

    ofp_train = open(filename_train, 'w+')
    final_json_train = dict()

    final_json_train['x'] = seqs_train_articles
    final_json_train['y'] = seqs_train_hl
    final_json_train['entities'] = train_items

    json.dump(final_json_train, ofp_train)
    ofp_train.close()
    del seqs_train_articles
    del seqs_train_hl
    del train_items

    print 'Dev data NER and indexing'
    seqs_dev_articles, seqs_dev_hl, dev_items = seqs(args, dev, vocab, st)

    filename_dev = args.dev if args.full_test else "small_" + args.dev
    filename_dev = str(count) + '_' + filename_dev

    ofp_dev = open(filename_dev, 'w+')
    final_json_dev = dict()

    final_json_dev['x'] = seqs_dev_articles
    final_json_dev['y'] = seqs_dev_hl
    final_json_dev['entities'] = dev_items

    json.dump(final_json_dev, ofp_dev)
    ofp_dev.close()
    del seqs_dev_articles
    del seqs_dev_hl
    del dev_items

    print 'Test data indexing'
    seqs_test_articles, seqs_test_hl, _ = seqs(args, test[0], vocab, st, test=True)

    filename_test = args.dev if args.full_test else "small_" + args.test
    filename_test = str(count) + '_' + filename_test

    ofp_test = open(filename_test, 'w+')
    final_json_test = dict()

    final_json_test['x'] = seqs_test_articles
    final_json_test['y'] = seqs_test_hl

    json.dump(final_json_test, ofp_test)
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
