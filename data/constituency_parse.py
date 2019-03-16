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

    intermediate_dir_art_scnlp = args.parsed_output_loc + '/articles_scnlp/'
    intermediate_dir_hl_scnlp = args.parsed_output_loc + '/highlights_scnlp/'

    if not os.path.exists(intermediate_dir_art):
        os.makedirs(intermediate_dir_art)

    if not os.path.exists(intermediate_dir_hl):
        os.makedirs(intermediate_dir_hl)

    if not os.path.exists(intermediate_dir_art_scnlp):
        os.makedirs(intermediate_dir_art_scnlp)

    if not os.path.exists(intermediate_dir_hl_scnlp):
        os.makedirs(intermediate_dir_hl_scnlp)

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
                ofp_hl.write(current_highlights[i] + '\n')

            ofp_art.close()
            ofp_hl.close()
            output_file_count += 1

    ofp_filelist_art.close()
    ofp_filelist_hl.close()


def recombine_scnlp_data(args):
    ifp_lookup = open(args.parsed_output_loc + 'lookup.txt', 'r')

    article_info = dict()

    stopwords = create_stopwords(args)

    print 'Input sha1, hl idx'
    for line in ifp_lookup:
        items = line.rstrip().split()

        article_info[items[2]] = items[1]

    ifp_lookup.close()

    print 'Combining..'

    train_order, dev_order, test_order = get_order(args)
    all_order = [train_order, dev_order, test_order]
    types = ['train', 'dev', 'test']

    for ls, type_ in zip(all_order, types):
        print 'Loading HL for', type_

        annotated_hl_fp = open(args.intermediate + '_' + type_ + '.txt.json', 'r')
        annotated_hl_json = json.load(annotated_hl_fp)
        sentences = annotated_hl_json['sentences']

        annotated_hl_fp.close()

        print 'loaded..'

        counter = 0

        for sha in ls:
            if sha not in article_info:
                continue

            if counter % 1000 == 0:
                print 'at', counter

            ifp_article = open(args.parsed_output_loc + 'scnlp/' + sha + '.txt.json', 'rb')
            ofp_combined = open(args.parsed_output_loc + 'processed/' + sha + '.json', 'w+')

            document = json.load(ifp_article)['sentences']
            ifp_article.close()

            cur_hl = sentences[hl_idx_start:hl_idx_end]

            process_pos(cur_hl, document, stopwords)

            combined_json_out = dict()

            combined_json_out['highlights'] = cur_hl
            combined_json_out['document'] = document

            json.dump(combined_json_out, ofp_combined)
            ofp_combined.close()

            counter += 1


def tree2dict(tree):
    return {tree.label(): [tree2dict(t) if isinstance(t, ParentedTree) else t.lower() for t in tree]}


def get_order(args):
    train_ls, dev_ls, test_ls = [], [], []

    ifp = open('all_' + args.source + '_sha_train.out', 'rb')

    for line in ifp:
        train_ls.append(line.rstrip())

    ifp.close()
    ifp = open('all_' + args.source + '_sha_dev.out', 'rb')

    for line in ifp:
        dev_ls.append(line.rstrip())

    ifp.close()
    ifp = open('all_' + args.source + '_sha_test.out', 'rb')

    for line in ifp:
        test_ls.append(line.rstrip())

    return train_ls, dev_ls, test_ls


def process_pos(cur_hl, document, stopwords):
    num_sent = len(document)
    hl_token_set = set()
    get_unigrams(cur_hl, hl_token_set, stopwords)

    for i in xrange(num_sent):

        sentence = document[i]
        tokens = sentence['tokens']
        num_tok = len(tokens)

        # tree = ParentedTree.fromstring(sentence['parse'])

        paths = []
        leaves = []
        # mask = [0]*num_tok
        #
        # dfs_nltk_tree(tree, paths, leaves, i + 1)
        # leaves = leaves[::-1]
        #
        # create_subtree_mask(leaves, mask, hl_token_set)
        #
        # assert num_tok == len(paths)
        # paths = paths[::-1]

        for j in xrange(num_tok):
            tokens[j]['trace'] = paths
            tokens[j]['pretrain'] = 0


def dfs_nltk_tree(tree, paths, leaves, s_idx):
    stack = []

    stack.append(tree)
    cur_path = ['S' + str(s_idx)]

    while len(stack) > 0:

        item = stack.pop()

        if type(item) == int:
            cur_path.pop()
            continue

        if type(item[0]) == ParentedTree:
            cur_path.append(item.label())

            stack.append(-1)

            for sub_t in item:
                stack.append(sub_t)
        else:
            cur_path.append(item.label())
            paths.append((item[0], cur_path[:]))
            cur_path.pop()

            leaves.append(item)


def create_subtree_mask(leaves, mask, hl_token_set):
    i = 0
    l = len(mask)

    while i < l:
        w = leaves[i][0].lower()

        if w in hl_token_set:
            cur_root = leaves[i].parent()

            begin, end = get_subtree_bounds(cur_root, leaves, i)

            for k in range(begin,  end):
                if k >= len(mask):
                    print ''
                mask[k] = 1

            i = end
        else:
            i += 1


def get_subtree_bounds(cur_root, leaves, i):
    c_idx = 0

    for c in cur_root:
        if c == leaves[i]:
            break
        c_idx += len(c.leaves())

    begin = i - c_idx
    end = begin + len(cur_root.leaves())

    return begin, end


def get_unigrams(cur_hl, hl_token_set, stopwords):
    sw = stopwords[0]
    punct = stopwords[1]

    for highlight in cur_hl:
        tokens = highlight['tokens']

        for token in tokens:
            word = token['originalText'].lower()

            if word not in sw and word not in punct:
                hl_token_set.add(word)


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


def create_stopwords(args):
    stopwords = set()
    punctuation = set()

    lst_words = []

    ifp = open(str(args.source) + '_vocab_' + str(args.vocab_size) + '.txt', 'r')

    for line in ifp:
        w = line.rstrip()
        lst_words.append(w)

    ifp.close()
    ifp = open(args.stopwords, 'r')

    # Add stopwords
    for line in ifp:
        w = line.rstrip()
        stopwords.add(w)

    ifp.close()

    tokenizer = RegexpTokenizer(r'\w+')

    # add punctuation
    for i in xrange(len(lst_words)):
        w = lst_words[i]

        if len(tokenizer.tokenize(w)) == 0:
            punctuation.add(w)

    return stopwords, punctuation


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
        sha_ls(args)
        recombine_scnlp_data(args)
