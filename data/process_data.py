import json
import os
import sys

import nltk.data
from nltk.tokenize import RegexpTokenizer

import data_args
from util import data_utils as utils

reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    highlights, articles = split_data(args)
    w2v_model = utils.create_w2v_model(args, [item for sublist in articles[512:] for item in sublist])

    machine_ready(args, highlights, articles, w2v_model.wv)


def split_data(args):
    small_size_counter = 0
    highlights = []
    articles = []

    for subdir, dirs, files in os.walk(args.raw_data):
        for file_in in files:

            current_article = []
            current_highlights = []

            if file_in.startswith('.'):
                continue

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

            current_article, current_highlights = tokenize(args, current_article, current_highlights)

            if len(current_article) == 0:
                continue
            highlights.append(current_highlights)
            articles.append(current_article)

            small_size_counter += 1

            if not args.full_test and small_size_counter >= args.small_limit + 512:
                return highlights, articles

    return highlights, articles


def machine_ready(args, highlights, articles, w2v_model):
    input_seqs = []
    input_seqs_hl = []
    utils.write_embeddings_to_file(args, w2v_model)

    for i in xrange(len(articles)):

        single_inp = []

        for j in xrange(args.max_sentences):
            for k in xrange(args.sentence_length):

                if j < len(articles[i]) and k < len(articles[i][j]):
                    single_inp.append(articles[i][j][k])
                else:
                    single_inp.append('<padding>')

        input_seqs.append(single_inp)

    for i in xrange(len(highlights)):

        single_inp = []

        for j in xrange(args.max_sentences_hl):
            for k in xrange(args.sentence_length_hl):

                if j < len(highlights[i]) and k < len(highlights[i][j]):
                    single_inp.append(highlights[i][j][k])
                else:
                    single_inp.append('<padding>')

        input_seqs_hl.append(single_inp)

    filename_train = args.train if args.full_test else "small_" + args.train
    filename_dev = args.dev if args.full_test else "small_" + args.dev

    ofp_train = open(filename_train, 'w+')
    ofp_dev = open(filename_dev, 'w+')

    final_json_train = dict()
    final_json_dev = dict()

    final_json_train['x'] = input_seqs[:512]
    final_json_train['y'] = input_seqs_hl[:512]
    final_json_dev['x'] = input_seqs[512:]
    final_json_dev['y'] = input_seqs_hl[512:]

    json.dump(final_json_train, ofp_train)
    json.dump(final_json_dev, ofp_dev)
    ofp_train.close()
    ofp_dev.close()


def tokenize(args, current_article, current_highlights):
    article = []
    highlights = []
    tokenizer = RegexpTokenizer(r'\w+')

    for item in current_article:
        item_strip = str(item.encode('utf-8'))
        sentences = nltk.sent_tokenize(item_strip.decode("utf8"))

        for sentence in sentences:
            article.append([w.lower() for w in tokenizer.tokenize(sentence)])

    for item in current_highlights:
        item_strip = str(item.encode('utf-8'))
        sentences = nltk.sent_tokenize(item_strip.decode("utf8"))

        for sentence in sentences:
            highlights.append([w.lower() for w in nltk.word_tokenize(sentence)])

    return article, highlights


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
