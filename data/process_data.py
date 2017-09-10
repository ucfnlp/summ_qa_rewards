import json
import os
import nltk.data

from util import data_utils as utils
import data_args


def process_data(args):

    highlights, articles = split_data(args)
    utils.create_w2v_model(args, [item for sublist in articles for item in sublist])

    machine_ready(highlights, articles)


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
                if '@highlight' in line:
                    incoming_hl = True
                    continue

                if incoming_hl and len(line.strip()) > 0:
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


def machine_ready(highlights, articles):


def tokenize(args, current_article, current_highlights):
    article = []
    highlights = []

    for item in current_article:
        article.append([w.lower() for w in nltk.word_tokenize(item)])

    for item in current_highlights:
        highlights.append([w.lower() for w in nltk.word_tokenize(item)])

    return article, highlights


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
