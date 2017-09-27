import json
import os
import sys

import nltk.data

import data_args
from util import data_utils as utils

reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    highlights, articles = split_data(args)
    w2v_model = utils.create_w2v_model(args, [item for sublist in articles for item in sublist])

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

            if not args.full_test and small_size_counter >= args.small_limit:
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

    filename = "small" + args.train if args.full_test else args.train
    ofp = open(filename, 'w+')

    final_json = dict()

    final_json['x'] = input_seqs
    final_json['y'] = input_seqs_hl

    json.dump(final_json, ofp)
    ofp.close()


def tokenize(args, current_article, current_highlights):
    article = []
    highlights = []

    for item in current_article:
        item_strip = str(item.encode('utf-8'))
        sentences = nltk.sent_tokenize(item_strip.decode("utf8"))

        for sentence in sentences:
            article.append([w.lower() for w in nltk.word_tokenize(sentence)])

    for item in current_highlights:
        item_strip = str(item.encode('utf-8'))
        sentences = nltk.sent_tokenize(item_strip.decode("utf8"))

        for sentence in sentences:
            highlights.append([w.lower() for w in nltk.word_tokenize(sentence)])

    return article, highlights


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
