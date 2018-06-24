import json
import os
import sys

import hashlib
from nltk import ParentedTree
import numpy as np

import data_args


reload(sys)
sys.setdefaultencoding('utf8')


def process_data(args):
    train, dev, test, unique_w, parse_labels = split_data(args)

    prepare_rouge(args, test[0], 'test')
    prepare_rouge(args, dev[0], 'dev')

    word_counts = [args.vocab_size]

    create_pt_map(args, parse_labels)

    for count in word_counts:
        print 'Building dataset for vocab size : ' + str(count)
        vocab, placeholder, unk = create_vocab_map(args, unique_w, count)
        machine_ready(args, train, dev, test, vocab, count, placeholder, unk, parse_labels)


def split_data(args):
    small_size_counter = 0
    unique_words = dict()
    parse_labels = dict()

    highlights_train = []
    articles_train = []
    hashes_train = []

    highlights_dev = []
    articles_dev = []
    hashes_dev = []

    highlights_test = []
    articles_test = []
    hashes_test = []

    sentence_lengths = []

    train_urls, dev_urls, test_urls = get_url_sets(args)

    data_dirs = [args.raw_data_cnn if args.source == 'cnn' else args.raw_data_dm]

    for raw_data in data_dirs:
        for subdir, dirs, files in os.walk(raw_data):
            for file_in in files:

                if file_in.startswith('.'):
                    continue

                sha = file_in.split('.')[0]
                file_in = open(subdir + file_in, 'r')

                inp_json = json.load(file_in)

                # Here current_article is a list containing tuples (sentence, parse_paths, mask, chunk_ls)
                current_article = extract_tokens(args, inp_json['document'], inp_json['highlights'], unique_words,
                                                 parse_labels)
                for c in current_article:
                    sentence_lengths.append(len(c[0]))
                current_highlights = inp_json['highlights']

                if len(current_article) == 0:
                    continue

                catg = get_set(sha, train_urls, dev_urls, test_urls)
                sha = str(sha)

                if catg < 0:
                    print 'Problem with : ' + str(sha)
                    continue

                if catg == 1: #TRAIN
                    highlights_train.append(current_highlights)
                    articles_train.append(current_article)
                    hashes_train.append(sha)
                elif catg == 2: #DEV
                    highlights_dev.append(current_highlights)
                    articles_dev.append(current_article)
                    hashes_dev.append(sha)
                else:#TEST
                    highlights_test.append(current_highlights)
                    articles_test.append(current_article)
                    hashes_test.append(sha)

                small_size_counter += 1

                if not args.full_test and small_size_counter >= args.small_limit:
                    return (highlights_train, articles_train, hashes_train), (highlights_dev, articles_dev, hashes_dev), (
                    highlights_test, articles_test, hashes_test), unique_words, parse_labels

    print np.percentile(sentence_lengths,95)
    return (highlights_train, articles_train, hashes_train), (highlights_dev, articles_dev, hashes_dev), (
    highlights_test, articles_test, hashes_test), unique_words, parse_labels


def prepare_rouge(args, inp, type):
    file_part = args.model_summ_path + type + '_'
    rouge_counter = 0

    for item in inp:
        ofp = open(file_part + args.source + '_' + str(rouge_counter).zfill(6) + '.txt', 'w+')

        for i in xrange(len(item)):

            text = ' '.join(item[i]) + ' .'

            ofp.write(text)

            if i <= len(item) - 1:
                ofp.write(' ')

        ofp.close()
        rouge_counter += 1


def seqs_art(args, inp, vocab, entity_set, raw_entity_mapping, first_word_map, unk, parse_labels, return_r=False):
    inp_seqs = []
    inp_s_chunks = []
    inp_seqs_mask = []
    inp_seqs_raw = []
    inp_ents = []
    inp_parse_paths = []

    counter = 0

    total_samples = len(inp)
    print total_samples, 'total samples'

    for article in inp:
        if (total_samples / 10) > 0 and counter % (total_samples / 10) == 0:
            print '..', counter
        counter += 1

        single_inp_seqs = []
        single_inp_s_chunks = []
        single_inp_seqs_mask = []
        single_inp_seqs_raw = []
        single_inp_parse_paths = []
        entities_in_article = set()

        for i in xrange(len(article)):

            sent = article[i][0]
            parse_paths = article[i][1]
            masks = article[i][2]
            chunk_ls = article[i][3]

            single_inp_sent = []
            single_inp_sent_pp = []
            single_inp_sent_raw = []

            for w in xrange(len(sent)):

                # 1.) check if word in vocab or not
                word = sent[w].lower()

                index = vocab[word] if word in vocab else unk
                single_inp_sent.append(index)

                if return_r:
                    single_inp_sent_raw.append(word)

                # 2.) check if word starts NER
                if word in first_word_map:

                    originals = first_word_map[word]

                    for raw_text_entity in originals:

                        text_ls = raw_text_entity.split(' ')
                        ent_len = len(text_ls)
                        entity_found = True

                        for i in xrange(ent_len):
                            if w + i >= len(sent):
                                entity_found = False
                                break

                            if sent[w + i] != text_ls[i]:
                                entity_found = False
                                break

                        if entity_found:
                            entity = entity_set[raw_entity_mapping[raw_text_entity]]
                            entities_in_article.add(entity[0])
                            break

                single_inp_sent_pp.append([parse_labels[item] for item in parse_paths[w]])

            single_inp_seqs_mask.append(masks)
            single_inp_s_chunks.append(chunk_ls)
            single_inp_seqs.append(single_inp_sent)
            single_inp_parse_paths.append(single_inp_sent_pp)

            if return_r:
                single_inp_seqs_raw.append(single_inp_sent_raw)

        inp_ents.append(list(entities_in_article))
        inp_seqs.append(single_inp_seqs)
        inp_s_chunks.append(single_inp_s_chunks)
        inp_seqs_mask.append(single_inp_seqs_mask)
        inp_parse_paths.append(single_inp_parse_paths)

        if return_r:
            inp_seqs_raw.append(single_inp_seqs_raw)

    if return_r:
        return inp_seqs, inp_ents, inp_seqs_raw, inp_parse_paths, inp_seqs_mask, inp_s_chunks
    else:
        return inp_seqs, inp_ents, inp_parse_paths, inp_seqs_mask, inp_s_chunks


def seqs_hl(args, inp, vocab, entity_set, entity_counter, raw_entity_mapping, first_word_map,  type, placeholder, unk):
    input_hl_seqs = []
    input_hl_entities = []
    input_hl_clean = []

    tag_ls = ['PERSON', 'LOCATION', 'ORGANIZATION', 'MISC']
    dep_ls = ['']

    total_samples = len(inp)

    for sample in xrange(total_samples):
        if (total_samples / 10) > 0 and sample % (total_samples / 10) == 0:
            print '..', sample

        single_inp_hl = []
        single_inp_clean = []
        single_inp_hl_entity_ls = []

        highlights = inp[sample]
        num_sentences = len(highlights)
        for h in xrange(num_sentences):

            single_sent_hl_entity_ls = []

            # 1.) find sentence root
            working_anno_hl = highlights[h]

            basic_dep = working_anno_hl['basicDependencies']
            enhanced_dep = working_anno_hl['enhancedDependencies']
            tokens_ls = working_anno_hl['tokens']

            root_basic_dep = basic_dep[0]
            root_idx = root_basic_dep['dependent']

            root_token = tokens_ls[root_idx - 1]
            root_lemma = root_token['lemma'].lower()
            root_org = root_token['originalText']
            root_first_word = root_org

            start_r = end_r = root_idx - 1

            if root_lemma not in entity_set: # previously not found @entity
                entity_info = [entity_counter, 'ROOT']
                entity_set[root_lemma] = entity_info
                entity_counter += 1

            if root_org not in raw_entity_mapping:
                raw_entity_mapping[root_org] = root_lemma

            if root_first_word not in first_word_map:
                first_word_map[root_first_word] = [root_org]
            else:
                originals = first_word_map[root_first_word]

                if root_org not in originals:
                    first_word_map[root_first_word].append(root_org)

            clean_hl_vec = create_hl_vector(args, vocab, tokens_ls, unk)
            single_inp_clean.append(clean_hl_vec)

            if start_r == end_r:
                hl_vec = clean_hl_vec[:]
                hl_vec[root_idx - 1] = placeholder
            else:
                hl_vec = clean_hl_vec[:start_r] + [placeholder] + clean_hl_vec[end_r + 1:]

            single_inp_hl.append(hl_vec)
            single_sent_hl_entity_ls.append(entity_set[root_lemma.lower()][0])

            # 2.) find all xobj, xsubj
            usable_question_dependencies = find_dependencies(basic_dep, tokens_ls)

            for (tok, type_, t_idx) in usable_question_dependencies:
                tok_lemma = tok['lemma'].lower()
                tok_org = tok['originalText']

                if tok_lemma not in entity_set:  # previously not found @entity
                    entity_info = [entity_counter, type_]
                    entity_set[tok_lemma] = entity_info
                    entity_counter += 1

                if tok_org not in raw_entity_mapping:
                    raw_entity_mapping[tok_org] = tok_lemma

                if tok_org not in first_word_map:
                    first_word_map[tok_org] = [tok_org]
                else:
                    originals = first_word_map[tok_org]

                    if tok_org not in originals:
                        first_word_map[tok_org].append(tok_org)

                hl_vec = clean_hl_vec[:]
                hl_vec[t_idx] = placeholder

                single_inp_hl.append(hl_vec)
                single_sent_hl_entity_ls.append(entity_set[tok_lemma.lower()][0])

            # 3.) find all instances of tags
            # named entities in the form : (entity name, start, end, type, raw name, first word)
            entities = find_ner_tokens(args, tokens_ls, tag_ls)

            for entity_name, start, end, e_type, raw_name, first_word in entities:
                if entity_name not in entity_set:
                    entity_info = [entity_counter, e_type]
                    entity_set[entity_name] = entity_info
                    entity_counter += 1

                hl_vec_complete = clean_hl_vec[:start] + [placeholder] + clean_hl_vec[end + 1:]

                single_inp_hl.append(hl_vec_complete)
                single_sent_hl_entity_ls.append(entity_set[entity_name][0])

                if raw_name not in raw_entity_mapping:
                    raw_entity_mapping[raw_name] = entity_name

                if first_word not in first_word_map:
                    first_word_map[first_word] = [raw_name]
                else:
                    originals = first_word_map[first_word]

                    if raw_name not in originals:
                        first_word_map[first_word].append(raw_name)

            single_inp_hl_entity_ls.append(single_sent_hl_entity_ls)

        input_hl_seqs.append(single_inp_hl)
        input_hl_entities.append(single_inp_hl_entity_ls)
        input_hl_clean.append(single_inp_clean)

    return input_hl_seqs, input_hl_entities, input_hl_clean, entity_counter


def machine_ready(args, train, dev, test, vocab, count, placeholder, unk, parse_labels):
    entity_set = dict()
    raw_entity_mapping = dict()
    first_word_map = dict()

    entity_counter = 0

    print 'Train data NER HL proc..'
    seqs_train_hl, seqs_train_e, seqs_clean_train, entity_counter = seqs_hl(args, train[0], vocab, entity_set, entity_counter, raw_entity_mapping,
                                          first_word_map, 'train', placeholder, unk)
    print 'Dev data NER HL proc..'
    seqs_dev_hl, seqs_dev_e, seqs_clean_dev, entity_counter = seqs_hl(args, dev[0], vocab, entity_set, entity_counter, raw_entity_mapping,
                                      first_word_map, 'dev', placeholder, unk)
    print 'Test data NER HL proc..'
    seqs_test_hl, seqs_test_e, seqs_clean_test, entity_counter = seqs_hl(args, test[0], vocab, entity_set, entity_counter, raw_entity_mapping,
                                        first_word_map, 'test', placeholder, unk)

    sorted_first_word_map = sort_entries(first_word_map)

    print 'Train data indexing..'
    seqs_train_articles, seq_train_art_ents, seq_train_art_parse, seq_train_art_m, seq_train_chunks = seqs_art(args, train[1], vocab, entity_set, raw_entity_mapping,
                                                       sorted_first_word_map, unk, parse_labels)
    print 'Dev data indexing..'
    seqs_dev_articles, seq_dev_art_ents, seq_dev_art_raw, seq_dev_art_parse, seq_dev_art_m, seq_dev_chunks = seqs_art(args, dev[1], vocab, entity_set, raw_entity_mapping,
                                                   sorted_first_word_map, unk, parse_labels, return_r=True)
    print 'Test data indexing..'
    seqs_test_articles, seq_test_art_ents, seq_test_art_raw, seq_test_art_parse, seq_test_art_m, seq_test_chunks = seqs_art(args, test[1], vocab, entity_set, raw_entity_mapping,
                                                     sorted_first_word_map, unk, parse_labels, return_r=True)

    filename_train = args.train if args.full_test else "small_" + args.train
    filename_train = args.source + '_' + str(count) + '_' + filename_train

    ofp_train = open(filename_train, 'w+')
    final_json_train = dict()

    final_json_train['x'] = seqs_train_articles
    final_json_train['y'] = seqs_train_hl
    final_json_train['e'] = seqs_train_e
    final_json_train['sha'] = train[2]
    final_json_train['valid_e'] = seq_train_art_ents
    final_json_train['clean_y'] = seqs_clean_train
    final_json_train['parse'] = seq_train_art_parse
    final_json_train['mask'] = seq_train_art_m
    final_json_train['chunk'] = seq_train_chunks

    json.dump(final_json_train, ofp_train)
    ofp_train.close()

    filename_dev = args.dev if args.full_test else "small_" + args.dev
    filename_dev = args.source + '_' + str(count) + '_' + filename_dev

    ofp_dev = open(filename_dev, 'w+')
    final_json_dev = dict()

    final_json_dev['x'] = seqs_dev_articles
    final_json_dev['raw_x'] = seq_dev_art_raw
    final_json_dev['y'] = seqs_dev_hl
    final_json_dev['e'] = seqs_dev_e
    final_json_dev['sha'] = dev[2]
    final_json_dev['valid_e'] = seq_dev_art_ents
    final_json_dev['clean_y'] = seqs_clean_dev
    final_json_dev['parse'] = seq_dev_art_parse
    final_json_dev['mask'] = seq_dev_art_m
    final_json_dev['chunk'] = seq_dev_chunks

    json.dump(final_json_dev, ofp_dev)
    ofp_dev.close()

    filename_test = args.test if args.full_test else "small_" + args.test
    filename_test = args.source + '_' + str(count) + '_' + filename_test

    ofp_test = open(filename_test, 'w+')
    final_json_test = dict()

    final_json_test['x'] = seqs_test_articles
    final_json_test['y'] = seqs_test_hl
    final_json_test['e'] = seqs_test_e
    final_json_test['sha'] = test[2]
    final_json_test['raw_x'] = seq_test_art_raw
    final_json_test['clean_y'] = seqs_clean_test
    final_json_test['parse'] = seq_test_art_parse
    final_json_test['mask'] = seq_test_art_m
    final_json_test['chunk'] = seq_test_chunks

    json.dump(final_json_test, ofp_test)
    ofp_test.close()

    filename_entities = 'entities.json' if args.full_test else "small_entities.json"
    filename_entities = args.source + '_' + str(count) + '_' + filename_entities

    ofp_entities = open(filename_entities, 'w+')
    final_json_entities = dict()
    final_json_entities['entities'] = entity_set.items()

    json.dump(final_json_entities, ofp_entities)
    ofp_entities.close()


def extract_tokens(args, document, hl, unique_words, parse_labels):
    article = []

    top_sent = True

    for sent in document:
        tokens = sent['tokens']
        tree = ParentedTree.fromstring(sent['parse'])
        s, parse_paths, mask, chunk_ls = [], [], [], []

        for token in tokens:
            text = token['originalText'].lower()
            pt = token['trace'][1]
            pretrain = token['pretrain']

            s.append(text)
            parse_paths.append(pt)
            mask.append(pretrain)

            process_labels(parse_labels, pt)

            if text in unique_words:
                unique_words[text] += 1
            else:
                unique_words[text] = 1

        chunk_ls, num_l = dfs_nltk_tree(args, tree, args.chunk_threshold)

        assert num_l == len(s)

        if top_sent:

            top_sent = False

            if '-RRB-' in s:
                k = s.index('-RRB-')
                if k < 10:
                    s = s[k + 1:]
                    parse_paths = parse_paths[k + 1:]
                    mask = mask[k + 1:]

                    chunk_ls = trim_chunk_ls(k + 1, chunk_ls)

            if '--' in s:
                k = s.index('--')
                if k < 10:
                    s = s[k + 1:]
                    parse_paths = parse_paths[k + 1:]
                    mask = mask[k + 1:]

                    chunk_ls = trim_chunk_ls(k + 1, chunk_ls)

            if '-rrb-' in s:
                k = s.index('-rrb-')
                if k < 10:
                    s = s[k + 1:]
                    parse_paths = parse_paths[k + 1:]
                    mask = mask[k + 1:]

                    chunk_ls = trim_chunk_ls(k + 1, chunk_ls)

        assert sum(chunk_ls) == len(s)

        article.append((s, parse_paths, mask, chunk_ls))

    for sent in hl:
        tokens = sent['tokens']

        for token in tokens:
            text = token['originalText'].lower()

            if text in unique_words:
                unique_words[text] += 1
            else:
                unique_words[text] = 1

    return article


def dfs_nltk_tree(args, tree, threshold=5):
    stack = []
    subtrees = []
    total = 0

    stack.append(tree)

    while len(stack) > 0:

        item = stack.pop()

        if type(item) == ParentedTree:
            num_l = len(item.leaves())

            if num_l <= threshold:
                total += num_l
                subtrees.append(num_l)
                continue

            for sub_t in item[::-1]:
                stack.append(sub_t)
        else:
            subtrees.append(item)

    return subtrees, total


def create_pt_map(args, parse_labels):
    ofp = open(args.source + '_' + 'vocab_pt_.txt', 'w+')

    for label, value in parse_labels.iteritems():
        ofp.write(label + ' ' + str(value) + '\n')

    ofp.write('<padding> ' + str(len(parse_labels)) + '\n')
    ofp.close()


def trim_chunk_ls(k, chunk_ls):
    new_ls = []

    for item in chunk_ls:
        if k <= 0:
            new_ls.append(item)

        elif item <= k:
            k -= item
        elif item > k:
            new_ls.append(item - k)
            k = 0

    return new_ls


def create_vocab_map(args, unique_w, count):
    ofp = open(args.source + '_' + 'vocab_' + str(count) + '.txt', 'w+')
    vocab_map = dict()
    index = 0

    inv_map = dict()

    for k, v in unique_w.iteritems():
        if v in inv_map:
            inv_map[v].append(k)
        else:
            inv_map[v] = [k]

    for c in sorted(inv_map.iterkeys(), reverse=True):
        words = inv_map[c]

        if index >= count:
            break

        for w in words:
            ofp.write(w + '\n')
            vocab_map[w] = index
            index += 1

            if index >= count:
                break

    ofp.write('<padding>\n<unk>\n<placeholder>\n')

    vocab_map['<padding>'] = index
    vocab_map['<unk>'] = unk = index + 1
    vocab_map['<placeholder>'] = placeholder = index + 2

    index += 3

    ofp.close()
    return vocab_map, placeholder, unk


def process_labels(parse_labels, path):

    for label in path:
        if label not in parse_labels:
            parse_labels[label] = len(parse_labels)


def get_embedding_set(args):
    ifp = open(args.embedding_file, 'r')

    embs = set()

    for line in ifp:
        word = line.split(' ')[0]
        embs.add(word)

    ifp.close()
    return embs


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


def create_hl_vector(args, vocab, tokens_ls, unk):
    vector = []

    for token in tokens_ls:

        word = token['originalText'].lower()

        if word in vocab:
            vector.append(vocab[word])
        else:
            vector.append(unk)

    return vector


def sort_entries(first_word_map):
    new_first_word_map = dict()

    for word, ls in first_word_map.iteritems():

        new_ents = []
        originals_as_ls = []

        for entity in ls:
            originals_as_ls.append(entity.split(' '))

        sorted_ls = sorted(originals_as_ls, key=len, reverse=True)

        for entity_ls in sorted_ls:
            new_ents.append(' '.join(entity_ls))

        new_first_word_map[word] = new_ents

    return new_first_word_map


def find_dependencies(basic_dep, tokens):
    found_ls = []

    for item in basic_dep:
        dep = item['dep']

        if 'obj' in dep or 'subj' in dep:
            index = int(item['dependent']) -1
            token = tokens[index]

            found_ls.append((token, dep, index))

    return found_ls


def find_ner_tokens(args, tokens_ls, tag_ls):
    ner_set = set()
    current_ner = None
    start_idx = end_idx = -1

    for i in xrange(len(tokens_ls)):

        item = tokens_ls[i]

        if item['ner'] in tag_ls:

            if current_ner is None:
                start_idx = i
                current_ner = item['ner']
            elif current_ner != item['ner']:
                end_idx = i - 1

                name = ''
                name_raw = ''

                for j in range(start_idx, i):
                    name += tokens_ls[j]['lemma'].lower()
                    name += '' if j == i - 1 else ' '

                    name_raw += tokens_ls[j]['originalText'].lower()
                    name_raw += '' if j == i - 1 else ' '

                # (entity name, start, end, type, raw_name, first word)
                fw = tokens_ls[start_idx]['originalText'].lower()
                ner = (name, start_idx, end_idx, current_ner, name_raw, fw)
                ner_set.add(ner)

                start_idx = i
                current_ner = item['ner']

        elif current_ner is not None:
            end_idx = i - 1

            name = ''
            name_raw = ''

            for j in range(start_idx, i):
                name += tokens_ls[j]['lemma'].lower()
                name += '' if j == i - 1 else ' '

                name_raw += tokens_ls[j]['originalText'].lower()
                name_raw += '' if j == i - 1 else ' '

            fw = tokens_ls[start_idx]['originalText'].lower()
            ner = (name, start_idx, end_idx, current_ner, name_raw, fw)
            ner_set.add(ner)

            start_idx = -1
            current_ner = None

    return ner_set


if __name__ == '__main__':
    args = data_args.get_args()
    process_data(args)
