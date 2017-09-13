import os
import nltk.data
import numpy as np

from util import data_utils as utils
import data_args


def process_data(args):

    highlights, articles = split_data(args)
    w2v_model = utils.create_w2v_model(args, [item for sublist in articles for item in sublist])

    input_seqs = machine_ready(args, highlights, articles, w2v_model.wv)


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

            highlights.append(current_highlights)
            articles.append(current_article)

            small_size_counter += 1

            if not args.full_test and small_size_counter >= args.small_limit:
                return highlights, articles

    return highlights, articles


def machine_ready(args, highlights, articles, w2v_model):
    input_seqs = np.zeros((len(articles), args.max_sentences * args.sentence_length, args.embedding_dim), dtype='float64')

    for i in xrange(len(articles)):

        for j in xrange(args.max_sentences):

            if j >= len(articles[i]):
                break

            k = 0

            for word in articles[i][j]:
                if k == args.sentence_length:
                    break

                if word in w2v_model.vocab:
                    word_idx = w2v_model.vocab[word].index
                    input_seqs[i, j * args.sentence_length + k, :] = w2v_model.syn0[word_idx]

                k += 1

    return input_seqs


def tokenize(args, current_article, current_highlights):
    article = []
    highlights = []

    for item in current_article:
        sentences = nltk.sent_tokenize(item)

        for sentence in sentences:
            article.append([w.lower() for w in nltk.word_tokenize(sentence)])

    for item in current_highlights:
        sentences = nltk.sent_tokenize(item)

        for sentence in sentences:
            highlights.append([w.lower() for w in nltk.word_tokenize(sentence)])

    return article, highlights


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
