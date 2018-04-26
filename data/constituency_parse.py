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
                print 'Problem with :', sha
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

    article_info = dict()
    pos_set = set()

    stopwords = create_stopwords(args)

    print 'Input sha1, hl idx'
    for line in ifp_lookup:
        items = line.rstrip().split()

        article_info[items[2]] = items[1]

    ifp_lookup.close()

    print 'Combining..'

    train_order, dev_order, test_order = get_order()
    all_order = [train_order, dev_order, test_order]
    types = ['train', 'dev', 'test']
    limits = [512, 128, 128]

    for ls, type_, limit in zip(all_order, types, limits):
        print 'Loading HL for', type_

        annotated_hl_fp = open(args.intermediate + '_' + type_ + '.txt.json', 'r')
        annotated_hl_json = json.load(annotated_hl_fp)
        sentences = annotated_hl_json['sentences']

        annotated_hl_fp.close()

        print 'loaded..'

        counter = 0
        hl_idx_end = 0

        for sha in ls:
            hl_idx_start = hl_idx_end

            if sha not in article_info:
                continue

            hl_idx_end += int(article_info[sha])

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

            if counter == limit:
                break


def tree2dict(tree):
    return {tree.label(): [tree2dict(t) if isinstance(t, ParentedTree) else t.lower() for t in tree]}


def get_order():
    train_ls, dev_ls, test_ls = [], [], []

    ifp = open('all_cnn_sha_train.out', 'rb')

    for line in ifp:
        train_ls.append(line.rstrip())

    ifp.close()
    ifp = open('all_cnn_sha_dev.out', 'rb')

    for line in ifp:
        dev_ls.append(line.rstrip())

    ifp.close()
    ifp = open('all_cnn_sha_test.out', 'rb')

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

        tree = ParentedTree.fromstring(sentence['parse'])

        paths = []
        leaves = []
        mask = [0]*num_tok

        dfs_nltk_tree(tree, paths, leaves, i + 1)
        leaves = leaves[::-1]

        create_subtree_mask(leaves, mask, hl_token_set)

        assert num_tok == len(paths)
        paths = paths[::-1]

        for j in xrange(num_tok):
            tokens[j]['trace'] = paths[j]
            tokens[j]['pretrain'] = mask[j]


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
        recombine_scnlp_data(args)
